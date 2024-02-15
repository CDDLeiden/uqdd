# Imports
import os
from datetime import date
import warnings

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

warnings.filterwarnings("ignore")  # Turn off Graphein warnings

from botorch import fit_gpytorch_model
from typing import List, Union
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import gpytorch
from gpytorch.models import ExactGP, VariationalGP, ApproximateGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from gpytorch.kernels import (
    RBFKernel,
    Kernel,
    SpectralDeltaKernel,
    SpectralMixtureKernel,
)

from gauche.kernels.fingerprint_kernels.braun_blanquet_kernel import BraunBlanquetKernel
from gauche.kernels.fingerprint_kernels.dice_kernel import DiceKernel
from gauche.kernels.fingerprint_kernels.rogers_tanimoto_kernel import (
    RogersTanimotoKernel,
)
from gauche.kernels.fingerprint_kernels.sokal_sneath_kernel import SokalSneathKernel
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import wandb

from gauche.dataloader import MolPropLoader
from gauche.dataloader.data_utils import transform_data
from uqdd.models.baselines import MTBaselineDNN
from uqdd.models.models_utils import (
    get_datasets,
    get_model_config,
    get_sweep_config,
    build_loader,
    build_optimizer,
    save_models,
    calc_regr_metrics,
    set_seed,
    MultiTaskLoss,
)

from uqdd.models.models_utils import set_seed, get_model_config, get_datasets, get_tasks
from uqdd.models.models_utils import (
    build_loader,
    build_optimizer,
    MultiTaskLoss,
    save_models,
)
from uqdd.models.models_utils import UCTMetricsTable, process_preds
from uqdd.models.baselines import train_model

num_tasks = 20  # number of tasks i.e. labels
# rank = 1 # increasing the rank hyperparameter allows the model to learn more expressive
# correlations between objectives at the expense of increasing the number of
# model hyperparameters and potentially overfitting.


# TODO using SVGP? --> it is stochastic - with batches
#  https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html

today = date.today()
today = today.strftime("%Y%m%d")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
print(torch.version.cuda) if device == "cuda" else None

LOG_DIR = os.environ.get("LOG_DIR")
DATA_DIR = os.environ.get("DATA_DIR")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
CONFIG_DIR = os.environ.get("CONFIG_DIR")
FIGS_DIR = os.environ.get("FIGS_DIR")


class MultitaskExactGP(ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        num_tasks=20,
        rank=1,
        mean_kernel=None,
        covar_kernel=None,
    ):
        super(MultitaskExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = (
            gpytorch.means.ConstantMean() if mean_kernel is None else mean_kernel
        )
        self.covar_module = TanimotoKernel() if covar_kernel is None else covar_kernel

        # If We learn an IndexKernel for 4 tasks
        # (so we'll actually learn 4x4=16 tasks with correlations)
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=num_tasks, rank=rank
        )

    def forward(self, x, i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


class MultitaskSVGP(ApproximateGP):
    def __init__(
        self, inducing_points, num_tasks, rank=1, mean_kernel=None, covar_kernel=None
    ):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(MultitaskSVGP, self).__init__(variational_strategy)

        mean_kernel = (
            gpytorch.means.ConstantMean() if mean_kernel is None else mean_kernel
        )
        covar_kernel = TanimotoKernel() if covar_kernel is None else covar_kernel

        self.mean_module = gpytorch.means.MultitaskMean(
            mean_kernel, num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            covar_kernel, num_tasks=num_tasks, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BaselineDNNwithGP(nn.Module):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        mean_kernel: Kernel = None,
        covar_kernel: Kernel = None,
        input_dim: int = 2048,
        hidden_dims: List[int] = [2048, 256, 256],
        num_tasks: int = 20,
        use_spectral_norm: bool = False,
    ):
        super(BaselineDNNwithGP, self).__init__()
        # TODO: make sure hidden_dims have the correct length

        # Define the DNN part
        baseline_dnn = MTBaselineDNN(
            input_dim=input_dim,
            hidden_dim_1=hidden_dims[0],
            hidden_dim_2=hidden_dims[1],
            hidden_dim_3=hidden_dims[2],
        )
        self.feature_extractor = (
            baseline_dnn.feature_extractor
        )  # existing DNN architecture without the last layer
        # Define the GP part
        self.gp_layer = MultitaskSVGP(
            inducing_points,
            num_tasks,
            mean_kernel=mean_kernel,
            covar_kernel=covar_kernel,
        )

        if use_spectral_norm:
            self.gp_layer = spectral_norm(self.gp_layer)

    def forward(self, x):
        features = self.feature_extractor(x)
        # If using spectral normalization, apply it to features here
        gp_output = self.gp_layer(features)
        return gp_output


def get_mll(model, likelihood, loader=None, gp_type="exact", dnn_featurizer=False):
    if dnn_featurizer:
        model_ = model.gp_layer
    else:
        model_ = model
    if gp_type.lower() == "exact":
        mll = ExactMarginalLogLikelihood(likelihood, model_)
    elif gp_type.lower() == "variational":
        mll = VariationalELBO(likelihood, model_, num_data=len(loader))
    else:
        raise ValueError(f"Unknown type: {gp_type}")
    return mll


def get_kernel(kernel_name: str, config=None):
    kernels_dict = {
        "mean": gpytorch.means.ConstantMean(
            batch_shape=torch.Size([config.batch_size])
        ),
        "linear_mean": gpytorch.means.LinearMean(
            input_size=config.input_dim, batch_shape=torch.Size([config.batch_size])
        ),  # TODO: Fix BAtch shape Size
        "braun_blanquet": BraunBlanquetKernel(),
        "dice": DiceKernel(),
        "rbf": RBFKernel(),
        "rogers_tanimoto": RogersTanimotoKernel(),
        "sokal_sneath": SokalSneathKernel(),
        "tanimoto": TanimotoKernel(),
    }

    if kernel_name.lower() in kernels_dict.keys():
        return kernels_dict[kernel_name.lower()]
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")


def get_gp_model(config, train_loader, likelihood):
    # Initialize the GP model
    model_kwargs = dict(
        mean_kernel=get_kernel(config.mean_kernel),
        covar_kernel=get_kernel(config.covar_kernel),
        num_tasks=config.num_tasks,
        rank=config.rank,
        use_spectral_norm=config.use_spectral_norm,
    )

    if config.gp_type.lower() == "exact":
        model = MultitaskExactGP(
            train_loader.dataset.x_data,
            train_loader.dataset.y_data,
            likelihood,
            **model_kwargs,
        )

    elif config.gp_type.lower() == "variational":
        model = MultitaskSVGP(
            train_loader.dataset.x_data[: config.num_inducing_points],
            num_tasks=config.num_tasks,
            **model_kwargs,
        )

    else:
        raise ValueError(f"Unknown type: {config.gp_type}")

    return model


# Train the model
def train(model, likelihood, loader, optimizer, config):
    # , gp_type="exact", dnn_featurizer=False):
    # Get into training mode
    model.train()
    likelihood.train()
    mll = get_mll(
        model,
        likelihood,
        loader=loader,
        gp_type=config.gp_type,
        dnn_featurizer=config.dnn_featurizer,
    )
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # "Loss" for GPs - the marginal log likelihood
        loss = -mll(outputs, targets)
        # TODO check this line below
        loader.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# Evaluation loop
def evaluate(
    model,
    likelihood,
    loader,
    config=wandb.config,
    # gp_type="exact",
    # dnn_featurizer=False,
    return_targets=False,
):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    mll = get_mll(
        model,
        likelihood,
        loader=loader,
        gp_type=config.gp_type,
        dnn_featurizer=config.dnn_featurizer,
    )
    total_loss = 0.0
    outputs_all = []
    targets_all = []
    means_all = []
    lowers_all = []
    uppers_all = []

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = likelihood(model(inputs))
            loss = -mll(outputs, targets)
            total_loss += loss.item()
            means = outputs.mean
            lower, upper = outputs.confidence_region()
            outputs_all.append(outputs.cpu().detach())
            means_all.append(means.cpu().detach())
            lowers_all.append(lower.cpu().detach())
            uppers_all.append(upper.cpu().detach())

            if return_targets:
                targets_all.append(targets.cpu().detach())

    outputs_all = torch.cat(outputs_all, dim=0)
    means_all = torch.cat(means_all, dim=0)
    lowers_all = torch.cat(lowers_all, dim=0)
    uppers_all = torch.cat(uppers_all, dim=0)
    total_loss /= len(loader)
    rmse, r2, evs = calc_regr_metrics(means_all, outputs_all)

    if return_targets:
        targets_all = torch.cat(targets_all, dim=0)
        return (
            total_loss,
            rmse,
            r2,
            evs,
            outputs_all,
            means_all,
            lowers_all,
            uppers_all,
            targets_all,
        )

    return total_loss, rmse, r2, evs, outputs_all, means_all, lowers_all, uppers_all


def predict(
    model,
    likelihood,
    loader,
    return_targets=False,
):
    model.eval()
    likelihood.eval()
    outputs_all = []
    targets_all = []
    means_all = []
    lowers_all = []
    uppers_all = []
    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = likelihood(model(inputs))
            means = outputs.mean
            lower, upper = outputs.confidence_region()

            outputs_all.append(outputs.cpu().detach())
            means_all.append(means.cpu().detach())
            lowers_all.append(lower.cpu().detach())
            uppers_all.append(upper.cpu().detach())

            if return_targets:
                targets_all.append(targets.cpu().detach())

    outputs_all = torch.cat(outputs_all, dim=0)
    if return_targets:
        targets_all = torch.cat(targets_all, dim=0)
        return outputs_all, targets_all

    return outputs_all


def initial_evaluation(
    model,
    likelihood,
    train_loader,
    val_loader,
    config=wandb.config,
    # gp_type="exact",
    # dnn_featurizer=False,
    # return_targets=False
):
    (
        val_loss,
        val_rmse,
        val_r2,
        val_evs,
        outputs_all,
        means_all,
        lowers_all,
        uppers_all,
    ) = evaluate(model, likelihood, val_loader, config)
    train_loss, _, _, _, _, _, _, _ = evaluate(model, likelihood, train_loader, config)
    return train_loss, val_loss, val_rmse, val_r2, val_evs
    # return train_loss, val_loss, val_rmse, val_r2, val_evs


def run_epoch(
    model,
    likelihood,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler=None,
    epoch=0,
    config=wandb.config,
    # gp_type="exact",
    # dnn_featurizer=False,
):
    if epoch == 0:
        # Perform evaluation before training starts (epoch 0)
        train_loss, val_loss, val_rmse, val_r2, val_evs = initial_evaluation(
            model, likelihood, train_loader, val_loader, config
        )

    else:
        train_loss = train(model, likelihood, train_loader, optimizer, config)
        val_loss, val_rmse, val_r2, val_evs, _, _, _, _ = evaluate(
            model, likelihood, val_loader, config
        )
        if lr_scheduler is not None:
            # Update the learning rate
            lr_scheduler.step(val_loss)

    return epoch, train_loss, val_loss, val_rmse, val_r2, val_evs


def train_gp_model(
    train_loader,
    val_loader,
    config=wandb.config,
):
    try:
        set_seed(config.seed)
        # deterministic cuda algorithms
        torch.backends.cudnn.deterministic = True

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_tasks
        )

        model = get_gp_model(config, train_loader, likelihood)

        # move them to device
        model = model.to(device)
        likelihood = likelihood.to(device)
        # Temporarily initialize best_model
        best_model = model

        # Initialize the optimizer
        optimizer = build_optimizer(
            model, config.optimizer, config.learning_rate, config.weight_decay
        )

        # Define Learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_factor,
            patience=config.lr_patience,
            verbose=True,
        )

        # Train the model
        best_val_loss = float("inf")
        early_stop_counter = 0

        for epoch in tqdm(range(config.num_epochs + 1)):
            try:
                epoch, train_loss, val_loss, val_rmse, val_r2, val_evs = run_epoch(
                    model,
                    likelihood,
                    train_loader,
                    val_loader,
                    optimizer,
                    lr_scheduler,
                    epoch=epoch,
                    config=config,
                )

                # Log the metrics
                wandb.log(
                    data={
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/rmse": val_rmse,
                        "val/r2": val_r2,
                    }
                )
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    # Save the best model - dropped to avoid memory issues
                    # Update the best model and its performance
                    best_model = model
                else:
                    early_stop_counter += 1
                    if early_stop_counter > config.early_stop:
                        break
            except Exception as e:
                raise RuntimeError(
                    f"The following exception occurred inside the epoch loop {e}"
                )

        # Save the best model
        save_models(config, best_model, None, onnx=True)

        return best_model, likelihood

    except Exception as e:
        raise RuntimeError(f"The following exception occurred in train_model {e}")


def run_gp_model(
    datasets=None,
    config=None,
    activity="xc50",
    split="random",
    wandb_project_name=f"{today}-gp-test",
    **kwargs,
):
    # Load the config
    config = get_model_config(
        config=config,
        activity=activity,
        split=split,
        **kwargs,
    )

    # Load the dataset
    if datasets is None:
        datasets = get_datasets(activity=activity, split=split)

    # Get tasks names:
    tasks = get_tasks(activity=activity, split=split)

    # Initialize wandb
    with wandb.init(
        dir=LOG_DIR,
        project=f"{wandb_project_name}",
        config=config,
        reinit=True,
        save_code=True,
    ) as run:
        config = wandb.config
        run.tags = (
            [
                f"{activity}",
                f"{split}",
                f"{config.gp_type}",
                f"featurizer_{'with_dnn' if config.dnn_featurizer else 'gp_direct'}",
            ],
        )
        run.name = (
            f"{today}_gp_{config.gp_type}-{'with_dnn' if config.dnn_featurizer else ''}_{activity}_{split}_{wandb_project_name}",
        )  # group

        # Initialize the table to store the metrics
        uct_metrics_logger = UCTMetricsTable(
            model_type="gaussian_processes", config=config
        )

        # load the data
        train_loader, val_loader, test_loader = build_loader(
            datasets, config.batch_size, config.input_dim
        )

        # Train the model
        best_model, likelihood = train_gp_model(
            train_loader,
            val_loader,
            config=config,
        )

        # Testing metrics on the best model using evaluate
        test_loss, test_rmse, test_r2, test_evs, _, _, _, _ = evaluate(
            best_model, likelihood, test_loader, config
        )

        # Log the final test metrics
        wandb.log(
            {
                "test/loss": test_loss,
                "test/rmse": test_rmse,
                "test/r2": test_r2,
                "test/evs": test_evs,
            }
        )

        # Predictions
        gp_preds, targets = predict(
            best_model, likelihood, test_loader, return_targets=True
        )

        y_pred, y_std, y_true = process_preds(gp_preds, targets, None)

        # Calculate and log the metrics
        metrics = uct_metrics_logger(
            y_pred=y_pred, y_std=y_std, y_true=y_true, task_name="All 20 Targets"
        )

        for task_idx in range(len(tasks)):
            task_y_pred, task_y_std, task_y_true = process_preds(
                gp_preds, targets, task_idx=task_idx
            )

            # Calculate and log the metrics
            task_name = tasks[task_idx]
            metrics = uct_metrics_logger(
                y_pred=task_y_pred,
                y_std=task_y_std,
                y_true=task_y_true,
                task_name=task_name,
            )

        uct_metrics_logger.wandb_log()


def _train_gp_model(
    datasets=None,
    config: Union[str, dict] = "uqdd/config/gp/gp.json",
    activity: str = "xc50",
    split: str = "random",
    wandb_project_name: str = "gp-test",
    gp_type: str = "exact",
    dnn_featurizer: bool = False,
    seed: int = 42,
    **kwargs,
):
    # Load the config
    config = get_model_config(
        config=config,
        activity=activity,
        split=split,
        gp_type=gp_type,
        dnn_featurizer=dnn_featurizer,
        **kwargs,
    )

    # Load the dataset
    if datasets is None:
        datasets = get_datasets(activity=activity, split=split)

    # Get tasks names:
    tasks = get_tasks(activity=activity, split=split)

    # Initialize wandb
    with wandb.init(
        dir=LOG_DIR,
        project=f"{wandb_project_name}",
        config=config,
        tags=[f"{activity}", f"{split}", f"{gp_type}", f"{dnn_featurizer}"],
        name=f"{today}_gp_{gp_type}-{'with_dnn' if dnn_featurizer else ''}_{activity}_{split}_{wandb_project_name}",  # group
        reinit=True,
        save_code=True,
    ):  # as run:
        # Set the seed
        set_seed(seed)
        config = wandb.config

        # Initialize the table to store the metrics
        uct_metrics_logger = UCTMetricsTable(
            model_type="gaussian_process", config=config
        )

        train_loader, val_loader, test_loader = build_loader(
            datasets, config.batch_size, config.input_dim
        )

        # move them to device
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_tasks
        )

        model = get_gp_model(config, train_loader, likelihood)
        model = model.to(device)
        likelihood = likelihood.to(device)

        # Temporarily initialize best_model
        best_model = model

        # Initialize the optimizer
        optimizer = build_optimizer(
            model, config.optimizer, config.learning_rate, config.weight_decay
        )

        # Define Learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_factor,
            patience=config.lr_patience,
            verbose=True,
        )

        # Train the model
        best_val_loss = float("inf")
        early_stop_counter = 0
        # Train the model
        for epoch in tqdm(range(config.num_epochs + 1)):
            try:
                epoch, train_loss, val_loss, val_rmse, val_r2, val_evs = run_epoch(
                    model,
                    likelihood,
                    train_loader,
                    val_loader,
                    optimizer,
                    lr_scheduler,
                    epoch=epoch,
                    gp_type=config.gp_type,
                    dnn_featurizer=config.dnn_featurizer,
                )
                # Log the metrics
                wandb.log(
                    data={
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/rmse": val_rmse,
                        "val/r2": val_r2,
                    }
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    # Save the best model - dropped to avoid memory issues
                    # Update the best model and its performance
                    best_model = model

                else:
                    early_stop_counter += 1
                    if early_stop_counter > config.early_stop:
                        break
            except Exception as e:
                raise RuntimeError(
                    f"The following exception occurred inside the epoch loop {e}"
                )


def run_gp():
    pass

    #
    # # Calculate and log the metrics
    # # task_name =
    # metrics = uct_metrics_logger(
    #     y_pred=y_pred,
    #     y_std=y_std,
    #     y_true=y_true,
    #     task_name="All 20 Targets"
    # )
    # for task_idx in range(len(tasks)):
    #     task_y_pred, task_y_std, task_y_true = process_preds(ensemble_preds, targets, task_idx=task_idx)
    #
    #     # Calculate and log the metrics
    #     task_name = tasks[task_idx]
    #     metrics = uct_metrics_logger(
    #         y_pred=task_y_pred,
    #         y_std=task_y_std,
    #         y_true=task_y_true,
    #         task_name=task_name
    #     )
    #
    # uct_metrics_logger.wandb_log()

    # pass


# def plot_uncertainties():


# class MultitaskGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = TanimotoKernel()
#
#         # We learn an IndexKernel for 4 tasks
#         # (so we'll actually learn 4x4=16 tasks with correlations)
#         self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank)
#
#     def forward(self, x, i):
#         mean_x = self.mean_module(x)
#
#         # Get input-input covariance
#         covar_x = self.covar_module(x)
#         # Get task-task covariance
#         covar_i = self.task_covar_module(i)
#         # Multiply the two together to get the covariance we want
#         covar = covar_x.mul(covar_i)
#
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar)
#
#
# # Regression experiment parameters, number of random splits and train/test split size
# n_trials = 20
# test_set_size = 0.2

#
# # Another example of a GP Multimodel with SVGP - inspired from : https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html
# class MultitaskSVGP_2(ApproximateGP):
#     def __init__(self):
#         # Let's use a different set of inducing points for each latent function
#         inducing_points = torch.rand(num_latents, 16, 1) # TODO Check size
#
#         # We have to mark the CholeskyVariationalDistribution as batch
#         # so that we learn a variational distribution for each task
#         variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
#             inducing_points.size(-2), batch_shape=torch.Size([num_latents])
#         )
#
#         # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
#         # so that the output will be a MultitaskMultivariateNormal rather than a batch output
#         variational_strategy = gpytorch.variational.LMCVariationalStrategy(
#             gpytorch.variational.VariationalStrategy(
#                 self, inducing_points, variational_distribution, learn_inducing_locations=True
#             ),
#             num_tasks=4,
#             num_latents=3,
#             latent_dim=-1
#         )
#
#         super().__init__(variational_strategy)
#
#         # The mean and covariance modules should be marked as batch
#         # so we learn a different set of hyperparameters
#         self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
#             batch_shape=torch.Size([num_latents])
#         )
#
#     def forward(self, x):
#         # The forward function should be written as if we were dealing with each output
#         # dimension in batch
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
