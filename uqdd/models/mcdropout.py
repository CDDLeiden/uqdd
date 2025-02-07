from typing import Tuple, Optional, Dict, Any

import wandb
import torch
from torch import nn

from uqdd import DEVICE
from uqdd.models.baseline import BaselineDNN
from uqdd.utils import create_logger

from uqdd.models.utils_train import (
    train_model_e2e,
    evaluate_predictions,
    recalibrate_model,
    get_dataloader,
    predict,
)

from uqdd.models.utils_models import (
    get_model_config,
    get_sweep_config,
)


def enable_dropout(model: torch.nn.Module) -> None:
    """
    Enables dropout layers during inference for Monte Carlo Dropout.

    Parameters
    ----------
    model : torch.nn.Module
        The model in which dropout layers should be enabled in training mode.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def mc_predict(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_mc_samples: int = 100,
    device: torch.device = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs Monte Carlo (MC) Dropout prediction with multiple stochastic forward passes.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model with dropout layers.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test set.
    num_mc_samples : int, optional
        Number of MC forward passes to perform, by default 100.
    device : torch.device, optional
        Device to run inference on, by default DEVICE.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - Stacked model predictions across MC samples.
        - True labels.
        - Stacked aleatoric uncertainties across MC samples.
    """
    # model.train()  # Enable dropout
    outputs_all, aleatoric_all = [], []  # targets_all  []
    for _ in range(num_mc_samples):  # Multiple forward passes
        model.eval()
        enable_dropout(model)
        outputs, targets, alea = predict(
            model, test_loader, device=device, set_on_eval=False
        )
        outputs_all.append(outputs)
        aleatoric_all.append(alea)
    # stack on dim 2
    outputs_all = torch.stack(outputs_all, dim=2)
    aleatoric_all = torch.stack(aleatoric_all, dim=2)
    return outputs_all.cpu(), targets.cpu(), aleatoric_all.cpu()


def run_mcdropout(config: Optional[Dict[str, Any]] = None) -> Tuple[
    nn.Module, Optional[Any], Dict[str, Any], Dict[str, Any]
]:
    """
    Trains and evaluates a BaselineDNN model with Monte Carlo Dropout for uncertainty quantification.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary containing model and training settings.

    Returns
    -------
    Tuple[nn.Module (BaselineDNN), Optional[Any], Dict[str, Any], Dict[str, Any]]
        - Trained BaselineDNN model.
        - Isotonic recalibration model (if applicable).
        - Evaluation metrics.
        - Visualization plots.
    """
    if config is None:
        config = get_model_config(
            "mcdropout", split_type="random", activity_type="xc50"
        )  # * Defaulting to random split_type and xc50 activity_type *
    num_mc_samples = config.get("num_mc_samples", 100)
    best_model, config, _, _ = train_model_e2e(
        config,
        model=BaselineDNN,
        model_type="mcdropout",
        logger=LOGGER,
    )
    dataloaders = get_dataloader(config, device=DEVICE, logger=LOGGER)

    preds, labels, alea_vars = mc_predict(
        best_model,
        dataloaders["test"],
        num_mc_samples=num_mc_samples,
        device=DEVICE,
    )
    # Then comes the predict metrics part
    metrics, plots, uct_logger = evaluate_predictions(
        config, preds, labels, alea_vars, "mcdropout", LOGGER
    )
    # RECALIBRATION
    preds_val, labels_val, alea_vars_val = mc_predict(
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
        uct_logger=uct_logger,
    )
    uct_logger.wandb_log()
    wandb.finish()
    return best_model, iso_recal_model, metrics, plots


def run_mcdropout_wrapper(**kwargs: Any):
    """
    Wrapper function to initialize logging and run MC Dropout-based training and evaluation.

    Parameters
    ----------
    **kwargs : Any
        Additional configuration parameters.

    Returns
    -------
        - Trained BaselineDNN model.
        - Isotonic recalibration model (if applicable).
        - Evaluation metrics.
        - Visualization plots.
    """
    global LOGGER
    LOGGER = create_logger(name="mcdropout", file_level="debug", stream_level="info")
    config = get_model_config(model_type="mcdropout", **kwargs)
    return run_mcdropout(config)


def run_mcdropout_hyperparam(**kwargs: Any) -> None:
    """
    Runs a hyperparameter optimization sweep for Monte Carlo Dropout using Weights & Biases.

    Parameters
    ----------
    **kwargs : Any
        Configuration parameters for hyperparameter tuning, including:
        - `sweep_count` (int): Number of sweep iterations.
        - `wandb_project_name` (str): Weights & Biases project name.
    """
    global LOGGER
    LOGGER = create_logger(
        name="mcdropout-sweep", file_level="debug", stream_level="info"
    )
    sweep_count = kwargs.pop("sweep_count")
    wandb_project_name = kwargs.pop("wandb_project_name")

    config = get_sweep_config("mcdropout", **kwargs)
    config["project"] = wandb_project_name
    sweep_id = wandb.sweep(
        config,
        project=wandb_project_name,
    )
    print(f"Running sweep with SWEEP_ID: {sweep_id}")
    wandb.agent(sweep_id, function=run_mcdropout, count=sweep_count)


# if __name__ == "__main__":
#     run_mcdropout_wrapper(
#         data_name="papyrus",
#         activity_type="xc50",
#         n_targets=-1,
#         descriptor_protein="ankh-large",
#         descriptor_chemical="ecfp2048",
#         median_scaling=False,
#         split_type="random",
#         ext="pkl",
#         task_type="regression",
#         wandb_project_name=f"mcdp-test",
#         epochs=5,
#         num_mc_samples=5,
#     )
