# Imports
import os
from datetime import date
import warnings
warnings.filterwarnings("ignore") # Turn off Graphein warnings

from botorch import fit_gpytorch_model
from typing import List
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import gpytorch
from gpytorch.models import ExactGP, VariationalGP, ApproximateGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import RBFKernel, ScaleKernel, Kernel, MultitaskKernel, SpectralDeltaKernel, SpectralMixtureKernel
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import wandb

from gauche.dataloader import MolPropLoader
from gauche.dataloader.data_utils import transform_data
from uqdd.models.baselines import BaselineDNN
from uqdd.models.models_utils import set_seed, get_config, get_datasets, get_tasks
from uqdd.models.models_utils import build_loader, build_optimizer, MultiTaskLoss, save_models
from uqdd.models.models_utils import UCTMetricsTable, process_preds
from uqdd.models.baselines import train_model

num_tasks = 20 # number of tasks i.e. labels
# rank = 1 # increasing the rank hyperparameter allows the model to learn more expressive
# correlations between objectives at the expense of increasing the number of
# model hyperparameters and potentially overfitting.


# TODO using SVGP? --> it is stochastic - with batches
#  https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html

today = date.today()
today = today.strftime("%Y%m%d")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
print(torch.version.cuda) if device == 'cuda' else None

LOG_DIR = os.environ.get('LOG_DIR')
DATA_DIR = os.environ.get('DATA_DIR')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
CONFIG_DIR = os.environ.get('CONFIG_DIR')
FIGS_DIR = os.environ.get('FIGS_DIR')


class MultitaskExactGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks=20, rank=1, mean_kernel=None, covar_kernel=None):

        super(MultitaskExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() if mean_kernel is None else mean_kernel
        self.covar_module = TanimotoKernel() if covar_kernel is None else covar_kernel

        # We learn an IndexKernel for 4 tasks
        # (so we'll actually learn 4x4=16 tasks with correlations)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank)

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
    def __init__(self, inducing_points, num_tasks, rank=1, mean_kernel=None, covar_kernel=None):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(MultitaskSVGP, self).__init__(variational_strategy)

        mean_kernel = gpytorch.means.ConstantMean() if mean_kernel is None else mean_kernel
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
            mean_kernel: Kernel=None,
            covar_kernel: Kernel=None,
            input_dim: int =2048,
            hidden_dims: List[int]=[2048, 256, 256],
            num_tasks: int=20,
            use_spectral_norm: bool=False
    ):
        super(BaselineDNNwithGP, self).__init__()
        # TODO: make sure hidden_dims have the correct length

        # Define the DNN part
        baseline_dnn = BaselineDNN(input_dim=input_dim, hidden_dim_1=hidden_dims[0], hidden_dim_2=hidden_dims[1], hidden_dim_3=hidden_dims[2])
        self.feature_extractor = baseline_dnn.feature_extractor # existing DNN architecture without the last layer
        # Define the GP part
        self.gp_layer = MultitaskSVGP(inducing_points, num_tasks, mean_kernel=mean_kernel, covar_kernel=covar_kernel)

        if use_spectral_norm:
            # self.feature_extractor = spectral_norm(self.feature_extractor)
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


# Train the model
def train(model, loader, likelihood, optimizer, gp_type="exact", dnn_featurizer=False):
    # Get into training mode
    model.train()
    likelihood.train()
    mll = get_mll(model, likelihood, loader=loader, gp_type=gp_type, dnn_featurizer=dnn_featurizer)
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
def evaluate(model, loader, likelihood, gp_type="exact", dnn_featurizer=False, return_targets=False):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    mll = get_mll(model, likelihood, loader=loader, gp_type=gp_type, dnn_featurizer=dnn_featurizer)
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
            loss = - mll(outputs, targets)
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
    if return_targets:
        targets_all = torch.cat(targets_all, dim=0)
        return outputs_all, targets_all

    return outputs_all


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


def initial_evaluation(model, train_loader, val_loader, loss_fn):
    val_loss, val_rmse, val_r2, val_evs = evaluate(model, val_loader, loss_fn)
    train_loss, _, _, _ = evaluate(model, train_loader, loss_fn)
    return train_loss, val_loss, val_rmse, val_r2, val_evs



def run_epoch():
    pass



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

