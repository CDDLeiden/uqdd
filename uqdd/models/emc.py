"""
Evidential Monte Carlo (EMC) utilities.

This module provides helpers to run Monte Carlo dropout on evidential models,
compute NLL, and orchestrate training/evaluation with recalibration.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import wandb
from torch import Tensor

from uqdd import DEVICE
from uqdd.models.evidential import (
    EvidentialDNN,
    ev_uncertainty,
    ev_predict_params,
)
from uqdd.models.loss import nig_nll
from uqdd.models.mcdropout import enable_dropout
from uqdd.models.utils_models import get_model_config, calculate_means, stack_vars
from uqdd.models.utils_train import (
    train_model_e2e,
    evaluate_predictions,
    recalibrate_model,
    get_dataloader,
)
from uqdd.utils import create_logger


def emc_predict_params(
        ev_model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_mc_samples: int = 10,
        device: torch.device = DEVICE,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Perform Monte Carlo dropout sampling for an evidential model and average parameters.

    Parameters
    ----------
    ev_model : nn.Module
        Evidential model with dropout layers enabled during sampling.
    dataloader : torch.utils.data.DataLoader
        DataLoader for the dataset to be evaluated.
    num_mc_samples : int, optional
        Number of Monte Carlo forward passes. Default is 10.
    device : torch.device, optional
        Device for inference. Default is DEVICE.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        (mu, v, alpha, beta, targets) averaged across MC samples.
    """
    mus, vs, alphas, betas = [], [], [], []
    targets: Tensor | None = None
    for _ in range(num_mc_samples):
        ev_model.eval()
        enable_dropout(ev_model)
        mu, v, alpha, beta, targets = ev_predict_params(
            ev_model, dataloader, device=device, set_on_eval=False
        )
        mus.append(mu)
        vs.append(v)
        alphas.append(alpha)
        betas.append(beta)
    mus, vs, alphas, betas = stack_vars(mus, vs, alphas, betas)
    mus, vs, alphas, betas = calculate_means(mus, vs, alphas, betas)

    # targets is set during the last iteration above
    assert targets is not None
    return mus, vs, alphas, betas, targets

    # alea_vars, epist_vars = ev_uncertainty(vs, alphas, betas)
    #
    # return mus.cpu(), targets.cpu(), alea_vars.cpu(), epist_vars.cpu()


def emc_predict(
        ev_model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_mc_samples: int = 10,
        device: torch.device = DEVICE,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Perform Monte Carlo dropout sampling and return averaged outputs and uncertainties.

    Parameters
    ----------
    ev_model : nn.Module
        Evidential model with dropout layers.
    dataloader : torch.utils.data.DataLoader
        DataLoader for the dataset to be evaluated.
    num_mc_samples : int, optional
        Number of Monte Carlo forward passes. Default is 10.
    device : torch.device, optional
        Device for inference. Default is DEVICE.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        (outputs, targets, aleatoric_vars, epistemic_vars), shapes following
        [batch, tasks] and [batch, tasks, num_mc_samples] where applicable.
    """
    mus, vs, alphas, betas, targets = emc_predict_params(
        ev_model, dataloader, num_mc_samples, device
    )
    alea_vars, epist_vars = ev_uncertainty(vs, alphas, betas)

    return mus, targets, alea_vars, epist_vars


def emc_nll(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_mc_samples: int = 10,
        device: torch.device = DEVICE,
) -> float:
    """
    Calculate the negative log-likelihood (NLL) of the Normal Inverse Gamma distribution.

    Parameters
    ----------
    model : nn.Module
        Evidential model.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the dataset.
    num_mc_samples : int, optional
        Number of Monte Carlo forward passes. Default is 10.
    device : torch.device, optional
        Device for inference. Default is DEVICE.

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    mus, vs, alphas, betas, targets_all = emc_predict_params(
        model, dataloader, num_mc_samples, device
    )
    nll = nig_nll(mus, vs, alphas, betas, targets_all).item()
    return nll


#
# def emc_predict_params(
#     ev_model: nn.Module,
#     dataloader: torch.utils.data.DataLoader,
#     num_mc_samples: int = 10,
#     device: torch.device = DEVICE,
#     return_params: bool = False,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Performs Monte Carlo dropout sampling for an Evidential Deep Neural Network (EvDNN).
#     While Averaging over the distribution parameters.
#
#     Parameters
#     ----------
#     ev_model : nn.Module
#         The evidential model with dropout layers.
#     dataloader : torch.utils.data.DataLoader
#         DataLoader for the dataset to be evaluated.
#     num_mc_samples : int, optional
#         Number of Monte Carlo forward passes, by default 10.
#     device : torch.device, optional
#         The device to run the model on, by default DEVICE.
#
#     Returns
#     -------
#     Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
#
#     """
#     mus, vs, alphas, betas = [], [], [], []
#     for _ in range(num_mc_samples):
#         ev_model.eval()
#         enable_dropout(ev_model)
#         mu, v, alpha, beta, targets = ev_predict_params(
#             ev_model, dataloader, device=device, set_on_eval=False
#         )
#         mus.append(mu)
#         vs.append(v)
#         alphas.append(alpha)
#         betas.append(beta)
#
#     mus = torch.stack(mus, dim=2)
#     vs = torch.stack(vs, dim=2)
#     alphas = torch.stack(alphas, dim=2)
#     betas = torch.stack(betas, dim=2)
#
#     mus, vs, alphas, betas = calculate_means(mus, vs, alphas, betas)
#     # TODO return parameters if necessary for further analysis
#     if return_params:
#         return mus.cpu(), vs.cpu(), alphas.cpu(), betas.cpu()
#     alea_vars, epist_vars = ev_uncertainty(mus, vs, alphas, betas)
#     outputs = mus
#
#     return outputs.cpu(), targets.cpu(), alea_vars.cpu(), epist_vars.cpu()
#
#
# def emc_predict_params_nll(
#     ev_model: nn.Module,
#     dataloader: torch.utils.data.DataLoader,
#     num_mc_samples: int = 10,
#     device: torch.device = DEVICE,
# ):
#     # mus, vs, alphas, betas = [], [], [], []
#     test_nll = 0.0
#     for _ in range(num_mc_samples):
#         ev_model.eval()
#         enable_dropout(ev_model)
#
#         nll = ev_predict_params_nll(
#             ev_model, dataloader, device=device, set_on_eval=False
#         )
#         test_nll += nll
#     return test_nll / num_mc_samples


def run_emc(config: Optional[dict] = None) -> Tuple[nn.Module, nn.Module, dict, dict]:
    """
    Train and evaluate an Evidential Monte Carlo (EMC) model.

    Parameters
    ----------
    config : dict or None, optional
        Configuration dictionary for training and evaluation. Default is None.

    Returns
    -------
    Tuple[nn.Module, nn.Module, dict, dict]
        (trained_model, recalibration_model, metrics, plots).
    """
    logger = LOGGER
    num_mc_samples = config.get("num_mc_samples", 10)
    # Train EvDNN
    best_model, config, _, _ = train_model_e2e(
        config,
        model=EvidentialDNN,
        model_type="emc",
        logger=LOGGER,
    )
    dataloaders = get_dataloader(config, device=DEVICE, logger=logger)

    # preds, labels, alea_vars, epi_vars = emc_predict(
    #     best_model, dataloaders["test"], num_mc_samples=num_mc_samples, device=DEVICE
    # )
    # preds, alea_vars, epi_vars = calculate_means(preds, alea_vars, epi_vars)

    preds, labels, alea_vars, epi_vars = emc_predict(
        best_model,
        dataloaders["test"],
        num_mc_samples=num_mc_samples,
        device=DEVICE,
    )
    nll = emc_nll(best_model, dataloaders["test"], num_mc_samples, DEVICE)

    # Then comes the predict metrics part
    metrics, plots, uct_logger = evaluate_predictions(
        config,
        preds,
        labels,
        alea_vars,
        "emc",
        LOGGER,
        epi_vars,
        wandb_push=False,
        nll=nll,
    )
    # RECALIBRATION
    # preds_val, labels_val, alea_vars_val, epi_vars_val = emc_predict(
    #     best_model, dataloaders["val"], num_mc_samples=num_mc_samples, device=DEVICE
    # )
    # preds_val, alea_vars_val, epi_vars_val = calculate_means(
    #     preds_val, alea_vars_val, epi_vars_val
    # )

    preds_val, labels_val, alea_vars_val, epi_vars_val = emc_predict(
        best_model,
        dataloaders["val"],
        num_mc_samples=num_mc_samples,
        device=DEVICE,
    )
    iso_recal_model = recalibrate_model(
        preds_val,
        labels_val,
        alea_vars_val,
        preds,
        labels,
        alea_vars,
        config=config,
        epi_val=epi_vars_val,
        epi_test=epi_vars,
        uct_logger=uct_logger,
        nll=nll,
    )
    uct_logger.wandb_log()
    wandb.finish()
    return best_model, iso_recal_model, metrics, plots


def run_emc_wrapper(**kwargs):
    """
    Wrapper for running EMC with a configuration built from keyword arguments.

    Parameters
    ----------
    **kwargs
        Additional configuration parameters.

    Returns
    -------
    Tuple[nn.Module, nn.Module, dict, dict]
        (trained_model, recalibration_model, metrics, plots).
    """
    global LOGGER
    LOGGER = create_logger(name="emc", file_level="debug", stream_level="info")
    config = get_model_config(model_type="emc", **kwargs)
    return run_emc(config=config)

# if __name__ == "__main__":
#     run_emc_wrapper(
#         data_name="papyrus",
#         activity_type="xc50",
#         n_targets=-1,
#         descriptor_protein="ankh-large",
#         descriptor_chemical="ecfp2048",
#         median_scaling=False,
#         # split_type="scaffold_cluster",
#         split_type="random",
#         ext="pkl",
#         task_type="regression",
#         wandb_project_name="emc-test",
#         epochs=5,
#         num_mc_samples=5,
#     )
#     print("done")
