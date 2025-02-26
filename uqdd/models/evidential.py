import logging

import wandb

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from tqdm import tqdm

from uqdd import DEVICE
from uqdd.models.pnn import PNN
from uqdd.models.loss import nig_nll
from uqdd.utils import create_logger

from uqdd.models.utils_train import (
    train_model_e2e,
    evaluate_predictions,
    recalibrate_model,
    get_dataloader,
)

from uqdd.models.utils_models import (
    get_model_config,
    get_sweep_config,
)

from typing import Tuple, Optional


def ev_predict(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = DEVICE,
    set_on_eval: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs evidential prediction using the provided model and dataloader.

    Parameters
    ----------
    model : nn.Module
        The trained evidential deep learning model.
    dataloader : torch.utils.data.DataLoader
        The dataloader containing the test dataset.
    device : torch.device, optional
        The device to run inference on, by default DEVICE.
    set_on_eval : bool, optional
        Whether to set the model in evaluation mode, by default True.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Model outputs, targets, aleatoric uncertainties, and epistemic uncertainties.
    """
    if set_on_eval:
        model.eval()
    outputs_all, targets_all = [], []
    alea_all, epistemic_all = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(
            dataloader, total=len(dataloader), desc="Evidential prediction"
        ):
            inputs = tuple(x.to(device) for x in inputs)
            outputs = model(inputs)  # , alea_vars

            mu, v, alpha, beta = outputs  # (d.squeeze() for d in outputs)
            alea_vars = beta / (alpha - 1)  # aleatoric
            epist_var = torch.sqrt(beta / (v * (alpha - 1)))  # epistemic
            outputs = mu

            outputs_all.append(outputs)
            targets_all.append(targets)
            alea_all.append(alea_vars)
            epistemic_all.append(epist_var)

    outputs_all = torch.cat(outputs_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    alea_all = torch.cat(alea_all, dim=0)
    epistemic_all = torch.cat(epistemic_all, dim=0)

    return outputs_all, targets_all, alea_all, epistemic_all


def ev_predict_params_nll(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = DEVICE,
    set_on_eval: bool = True,
) -> torch.Tensor:
    """
    Calculates the negative log-likelihood (NLL) of the Normal Inverse Gamma (NIG) distribution.

    Parameters
    ----------
    model : nn.Module
        The trained evidential deep learning model.
    dataloader: torch.utils.data.DataLoader
        The dataloader containing the test dataset.
    device: torch.device, optional
        The device to run inference on, by default DEVICE.
    set_on_eval: bool, optional
        Whether to set the model in evaluation mode, by default True.

    Returns
    -------

    """
    if set_on_eval:
        model.eval()
    test_nll = 0.0
    # mus, vs, alphas, betas = [], [], [], []
    # all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(
            dataloader, total=len(dataloader), desc="Evidential prediction"
        ):
            inputs = tuple(x.to(device) for x in inputs)
            outputs = model(inputs)  # , alea_vars

            mu, v, alpha, beta = outputs  # (d.squeeze() for d in outputs)

            nll = nig_nll(mu, v, alpha, beta, targets)
            test_nll += nll.item()
            # mus.append(mu)
            # vs.append(v)
            # alphas.append(alpha)
            # betas.append(beta)
            # all_targets.append(targets)
    test_nll /= len(dataloader)

    return test_nll
    # mus = torch.cat(mus, dim=0)
    # vs = torch.cat(vs, dim=0)
    # alphas = torch.cat(alphas, dim=0)
    # betas = torch.cat(betas, dim=0)
    # all_targets = torch.cat(all_targets, dim=0)

    # return mus.cpu(), vs.cpu(), alphas.cpu(), betas.cpu(), all_targets.cpu()


# Adopted and Modified from https://github.com/teddykoker/evidential-learning-pytorch
class NormalInvGamma(nn.Module):
    """
    Normal Inverse Gamma layer for evidential regression.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_units : int
        Number of output units.
    """

    def __init__(self, in_features: int, out_units: int) -> None:
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units

    def evidence(self, x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
        """
        Computes evidence using softplus activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        eps : float, optional
            Small epsilon to ensure numerical stability, by default 1e-2.

        Returns
        -------
        torch.Tensor
            Transformed evidence.
        """
        return F.softplus(x) + eps

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Normal Inverse Gamma layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Predicted mean, variance, shape, and scale parameters.
        """
        out = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)

        return mu, v, alpha, beta


class Dirichlet(nn.Module):
    """
    Dirichlet layer for evidential classification.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_units : int
        Number of output units.
    """

    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units)
        self.out_units = int(out_units)

    def evidence(self, x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
        """
        Computes evidence using softplus activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        eps : float, optional
            Small epsilon to ensure numerical stability, by default 1e-2.

        Returns
        -------
        torch.Tensor
            Transformed evidence.
        """
        return F.softplus(x) + eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Dirichlet layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Dirichlet concentration parameters.
        """
        out = self.dense(x)
        alpha = self.evidence(out) + 1
        return alpha


class EvidentialDNN(PNN):
    """
    Evidential Deep Neural Network (DNN) for regression and classification.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary.
    logger : logging.Logger, optional
        Logger instance.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> None:
        super(EvidentialDNN, self).__init__(
            config,
            logger,
            aleavar_layer_included=False,
            **kwargs,
        )

        if self.task_type == "regression":
            self.output_layer = NormalInvGamma(
                self.config["regressor_layers"][-1], self.output_dim
            )
        elif self.task_type == "classification":
            self.output_layer = Dirichlet(
                self.config["regressor_layers"][-1], self.output_dim
            )
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        self.apply(self.init_wt)

    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the EvidentialDNN model.

        Parameters
        ----------
        inputs : Tuple[torch.Tensor, torch.Tensor]
            Tuple containing protein and chemical input tensors.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Model output containing either Normal Inverse Gamma or Dirichlet parameters.
        """
        prot_input, chem_input = inputs
        chem_features = self.chem_feature_extractor(chem_input)
        if not self.MT:
            prot_features = self.prot_feature_extractor(prot_input)
            combined_features = torch.cat((chem_features, prot_features), dim=1)
        else:
            combined_features = chem_features
        _output = self.regressor_or_classifier(combined_features)
        # output, var_ = self.output_layer(_output)
        output = self.output_layer(_output)

        return output


def run_evidential(
    config: Optional[dict] = None,
) -> Tuple[nn.Module, Optional[nn.Module], Optional[dict], Optional[dict]]:
    """
    Trains and evaluates an Evidential Deep Neural Network.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary, by default None.

    Returns
    -------
    Tuple[nn.Module, Optional[nn.Module], Optional[dict], Optional[dict]]
        The trained model, recalibration model, evaluation metrics, and plots.
    """
    device = DEVICE
    best_model, config, _, _ = train_model_e2e(
        config,
        model=EvidentialDNN,
        model_type="evidential",
        logger=LOGGER,
        device=device,
    )

    sweep = config.get("sweep", False)

    if not sweep:
        #
        dataloaders = get_dataloader(config, device=device, logger=LOGGER)
        preds, labels, alea_vars, epi_vars = ev_predict(
            best_model, dataloaders["test"], device=device
        )
        # Then comes the predict metrics part
        metrics, plots, uct_logger = evaluate_predictions(
            config,
            preds,
            labels,
            alea_vars,
            "evidential",
            logger=LOGGER,
            epi_vars=epi_vars,
            wandb_push=False,
        )
        # RECALIBRATION
        preds_val, labels_val, alea_vars_val, epi_vars_vals = ev_predict(
            best_model, dataloaders["val"], device=DEVICE
        )
        iso_recal_model = recalibrate_model(
            preds_val,
            labels_val,
            alea_vars_val,
            preds,
            labels,
            alea_vars,
            config,
            epi_val=epi_vars_vals,
            epi_test=epi_vars,
            uct_logger=uct_logger,
        )
        uct_logger.wandb_log()
    else:
        # we need to calculate val metrics with lambda = 1.0
        metrics, plots, iso_recal_model = None, None, None
    # wandb.finish()
    return best_model, iso_recal_model, metrics, plots


def run_evidential_wrapper(**kwargs):
    """
    Wrapper function for running Evidential Deep Learning.

    Parameters
    ----------
    kwargs : dict
        Additional configuration parameters.

    Returns
    -------
    Tuple[nn.Module, Optional[nn.Module], Optional[dict], Optional[dict]]
        The trained model, recalibration model, evaluation metrics, and plots.
    """
    global LOGGER
    LOGGER = create_logger(name="evidential", file_level="debug", stream_level="info")
    config = get_model_config(model_type="evidential", **kwargs)
    return run_evidential(config)


def run_evidential_hyperparam(**kwargs):
    """
    Runs hyperparameter optimization for Evidential Deep Learning using Weights & Biases sweeps.

    Parameters
    ----------
    kwargs: dict
        Hyperparameter configuration options.
    """
    global LOGGER
    LOGGER = create_logger(
        name="evidential-sweep", file_level="debug", stream_level="info"
    )
    sweep_count = kwargs.pop("sweep_count")
    wandb_project_name = kwargs.pop("wandb_project_name")
    sweep_config = get_sweep_config(
        "evidential", **kwargs, wandb_project_name=wandb_project_name
    )
    # print(f"{sweep_config=}")
    sweep_config["project"] = wandb_project_name

    sweep_id = wandb.sweep(sweep_config, project=wandb_project_name)
    print(f"Running sweep with SWEEP_ID: {sweep_id}")

    wandb.agent(sweep_id, function=run_evidential, count=sweep_count)


# if __name__ == "__main__":
#     run_evidential_wrapper(
#         data_name="papyrus",
#         n_targets=-1,
#         task_type="regression",
#         activity_type="kx",
#         split_type="random",
#         descriptor_protein="ankh-large",
#         descriptor_chemical="ecfp2048",
#         median_scaling=False,
#         ext="pkl",
#         wandb_project_name="evidential-test",
#         # epochs=5,
#         seed=42,
#         # device="cuda:1",
#     )
#     pass
#     run_evidential_hyperparam(
#         data_name="papyrus",
#         n_targets=-1,
#         task_type="regression",
#         activity_type="xc50",
#         split_type="random",
#         descriptor_protein="ankh-large",
#         descriptor_chemical="ecfp2048",
#         median_scaling=False,
#         ext="pkl",
#         wandb_project_name="evidential-test",
#         epochs=5,
#         seed=50,
#     )
#     print("Done")
