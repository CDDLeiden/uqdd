from typing import Tuple, Optional
import wandb
import torch
import torch.nn as nn

from uqdd import DEVICE
from uqdd.models.evidential import EvidentialDNN, ev_predict
from uqdd.models.mcdropout import enable_dropout
from uqdd.utils import create_logger

from uqdd.models.utils_train import (
    train_model_e2e,
    evaluate_predictions,
    recalibrate_model,
    get_dataloader,
)

from uqdd.models.utils_models import get_model_config, calculate_means


def emc_predict(
    ev_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_mc_samples: int = 10,
    device: torch.device = DEVICE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs Monte Carlo dropout sampling for an Evidential Deep Neural Network (EvDNN).

    Parameters
    ----------
    ev_model : nn.Module
        The evidential model with dropout layers.
    dataloader : torch.utils.data.DataLoader
        DataLoader for the dataset to be evaluated.
    num_mc_samples : int, optional
        Number of Monte Carlo forward passes, by default 10.
    device : torch.device, optional
        The device to run the model on, by default DEVICE.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        - Stacked outputs from MC samples [batch_size, num_tasks, num_mc_samples]
        - Labels [batch_size, num_tasks]
        - Aleatoric uncertainties [batch_size, num_tasks, num_mc_samples]
        - Epistemic uncertainties [batch_size, num_tasks, num_mc_samples]
    """
    outputs_all, aleatoric_all, epistemic_all = [], [], []  # targets_all = []
    for _ in range(num_mc_samples):
        ev_model.eval()
        enable_dropout(ev_model)
        preds, labels, alea, epistemic = ev_predict(
            ev_model, dataloader, device=device, set_on_eval=False
        )
        outputs_all.append(preds)
        aleatoric_all.append(alea)
        epistemic_all.append(epistemic)

    outputs_all = torch.stack(outputs_all, dim=2)  # .mean(dim=2)
    aleatoric_all = torch.stack(aleatoric_all, dim=2)  # .mean(dim=2)
    epistemic_all = torch.stack(epistemic_all, dim=2)  # .mean(dim=2)

    return outputs_all.cpu(), labels.cpu(), aleatoric_all.cpu(), epistemic_all.cpu()


def run_emc(config: Optional[dict] = None) -> Tuple[nn.Module, nn.Module, dict, dict]:
    """
    Trains and evaluates an Evidential Monte Carlo (EMC) model.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary for training and evaluation, by default None.

    Returns
    -------
    Tuple[nn.Module, nn.Module, dict, dict]
        - The trained EMC model.
        - The recalibration model.
        - Evaluation metrics.
        - Generated plots.
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
    preds, labels, alea_vars, epi_vars = emc_predict(
        best_model, dataloaders["test"], num_mc_samples=num_mc_samples, device=DEVICE
    )
    preds, alea_vars, epi_vars = calculate_means(preds, alea_vars, epi_vars)
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
    )
    # RECALIBRATION
    preds_val, labels_val, alea_vars_val, epi_vars_val = emc_predict(
        best_model, dataloaders["val"], num_mc_samples=num_mc_samples, device=DEVICE
    )
    preds_val, alea_vars_val, epi_vars_val = calculate_means(
        preds_val, alea_vars_val, epi_vars_val
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
    )
    uct_logger.wandb_log()
    wandb.finish()
    return best_model, iso_recal_model, metrics, plots


def run_emc_wrapper(**kwargs):
    """
    Wrapper function for running an EMC model.

    Parameters
    ----------
    kwargs : dict
        Additional configuration parameters.

    Returns
    -------
    Tuple[nn.Module, nn.Module, dict, dict]
        - The trained EMC model.
        - The recalibration model.
        - Evaluation metrics.
        - Generated plots.
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
