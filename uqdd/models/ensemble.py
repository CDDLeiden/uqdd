"""
Ensemble utilities.

This module provides helpers to train and aggregate ensembles, process results,
and compute uncertainty metrics across members.
"""

import logging
from typing import Optional, List, Type, Tuple, Any
from typing import Literal, cast

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import wandb
from numpy import ndarray
from torch.nn import Module

from uqdd import DEVICE, WANDB_DIR, WANDB_MODE, DATASET_DIR
from uqdd.models.pnn import PNN
from uqdd.models.utils_models import (
    get_model_config,
    set_seed,
)
from uqdd.models.utils_train import (
    train_model_e2e,
    evaluate_predictions,
    predict,
    recalibrate_model,
    assign_wandb_tags,
    get_dataloader,
    post_training_save_model,
)
from uqdd.utils import create_logger, save_pickle

mp.set_start_method("spawn", force=True)


class EnsembleDNN(nn.Module):
    """
    Ensemble Deep Neural Network (DNN) consisting of multiple instances of a base model.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary.
    model_class : Type[nn.Module], optional
        The class of the base model to be used in the ensemble, by default PNN.
    model_list : List[nn.Module], optional
        A pre-initialized list of models to be used in the ensemble, by default None.
    kwargs : dict
        Additional parameters for model initialization.
    """

    def __init__(
            self,
            config: Optional[dict] = None,
            model_class: Type[nn.Module] = PNN,
            model_list: Optional[List[nn.Module]] = None,
            **kwargs,
    ) -> None:
        super(EnsembleDNN, self).__init__()
        if config is None:
            config = get_model_config(model_type="ensemble", **kwargs)
        self.config = config
        self.logger = create_logger(name="EnsembleDNN")
        self.ensemble_size = config.get("ensemble_size", 100)
        if model_list is not None:
            models = model_list
        else:
            models = []
            seed = config.get("seed", 42)
            for _ in range(self.ensemble_size):
                set_seed(seed)
                seed += 1
                model = model_class(config, **kwargs)
                models.append(model)
        self.models = nn.ModuleList(models)

    def forward(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the ensemble model.

        Parameters
        ----------
        inputs : Tuple[torch.Tensor, torch.Tensor]
            Tuple containing protein and chemical input tensors.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Stacked model outputs and aleatoric uncertainty predictions from all ensemble members.
        """
        outputs = []
        vars_ = []
        for model in self.models:
            output, var_ = model(inputs)
            outputs.append(output)
            vars_.append(var_)
        outputs = torch.stack(
            outputs, dim=2
        )  # Shape: [batch_size, output_dim, ensemble_size]
        vars_ = torch.stack(
            vars_, dim=2
        )  # Shape: [batch_size, output_dim, ensemble_size]
        return outputs, vars_


def log_wandb_ensemble(results_tensor_avg: np.ndarray, config: dict) -> None:
    """
    Logs the averaged ensemble training results to Weights & Biases (wandb).

    Parameters
    ----------
    results_tensor_avg : np.ndarray
        Averaged training metrics across ensemble members.
    config : dict
        Configuration dictionary containing experiment settings.
    """
    wandb_keys = [
        "epoch",
        "train/loss",
        "train/rmse",
        "train/r2",
        "train/evs",
        "train/alea_mean",
        "train/alea_var",
        "model/pnorm",
        "model/gnorm",
        "val/loss",
        "val/rmse",
        "val/r2",
        "val/evs",
        "val/alea_mean",
        "val/alea_var",
    ]
    n_targets = config.get("n_targets", -1)
    if n_targets > 1:
        raise NotImplementedError(
            "Multitask training not yet supported for Ensemble Models"
        )
        # for task in n_targets:
        #     wandb_keys += [f"val/rmse/{task}", f"val/r2/{task}", f"val/evs/{task}"]

    # iterate over the metrics and log them to wandb
    num_epochs, num_metrics = results_tensor_avg.shape

    for epoch in range(num_epochs):
        wandb.log(
            # all data except epoch
            data=dict(zip(wandb_keys[1:], results_tensor_avg[epoch, 1:])),
            step=epoch,  # int(results_tensor_avg[epoch, 0]),
        )


def log_wandb_test(test_tensor_avg: np.ndarray, config: dict) -> None:
    """
    Logs the averaged test evaluation results to Weights & Biases (wandb).

    Parameters
    ----------
    test_tensor_avg : np.ndarray
        Averaged test metrics across ensemble members.
    config : dict
        Configuration dictionary containing experiment settings.
    """
    wandb_keys = [
        "test/loss",
        "test/rmse",
        "test/r2",
        "test/evs",
        "test/alea_mean",
        "test/alea_var",
    ]
    # multitask = config.get("MT", False)
    n_targets = config.get("n_targets", -1)

    if n_targets > 1:  # MT
        raise NotImplementedError(
            "Multitask training not yet supported for Ensemble Models"
        )
        # for task in n_targets:
        #     wandb_keys += [
        #         f"test/rmse/task_{task}",
        #         f"test/r2/task_{task}",
        #         f"test/evs/task_{task}",
        #     ]

    test_data = dict(zip(wandb_keys, test_tensor_avg[1:]))
    wandb.log(test_data)


def fill_to_max_epochs(array: np.ndarray, max_epochs: int) -> np.ndarray:
    """
    Expands an array by filling missing values with NaNs up to a specified maximum epoch count.

    Parameters
    ----------
    array : np.ndarray
        The input array containing training results.
    max_epochs : int
        The maximum number of epochs to expand the array to.

    Returns
    -------
    np.ndarray
        The padded array with NaN values for missing epochs.
    """
    num_metrics = array.shape[1]
    filled_array = np.full((max_epochs, num_metrics), np.nan)
    filled_array[: array.shape[0], : array.shape[1]] = array
    return filled_array


def process_results_arrs(
        result_arrs: List[np.ndarray | float],
        test_arrs: List[np.ndarray | float],
        config: dict,
        logger: logging.Logger,
        model_type: str = "ensemble",
) -> np.ndarray:
    """
    Aggregate per-member training/test results and compute ensemble-level stats.

    Parameters
    ----------
    result_arrs : list
        List of per-member result arrays.
    test_arrs : list
        List of per-member test arrays.
    config : dict
        Configuration dictionary.
    logger : logging.Logger
        Logger instance.
    model_type : str
        Model type label.

    Returns
    -------
    dict
        Aggregated results and statistics.
    """
    try:
        # get the maximum number of epochs
        max_epochs = max([results_arr.shape[0] for results_arr in result_arrs])
        # fill the results to max epochs
        result_arrs = [
            fill_to_max_epochs(results_arr, max_epochs) for results_arr in result_arrs
        ]
        # now we stack result tensors on dim 2
        result_arrs = np.stack(result_arrs, axis=2)
        logger.debug(f"{result_arrs.shape=}")
        # this should equal to (num_epochs, metrics_collected, ensemble_size)

        # Take average across model metrics
        results_tensor_avg = np.nanmean(result_arrs, 2)
        # HERE we should report to wandb
        log_wandb_ensemble(results_tensor_avg, config)

        # Test Arrs
        test_arrs = np.stack(test_arrs, axis=1)
        test_tensor_avg = np.nanmean(test_arrs, 1)
        logger.debug(f"{test_tensor_avg.shape=}")

        log_wandb_test(test_tensor_avg, config)

    except Exception as e:
        logger.exception(f"Error in stacking results: {e}")

    finally:
        # Here we want to save a pkl file with the results tensor
        save_pickle(
            result_arrs,
            DATASET_DIR
            / config.get("data_specific_path")
            / f"{model_type}_results.pkl",
        )

    return result_arrs


# Function to train one model on a specific GPU
def train_one_ensemble_member(
        config: dict, gpu_id: int, model_idx: int, logger: logging.Logger
) -> tuple[Module, ndarray, float, dict[str, Any]]:
    """
    Trains a single ensemble model instance on a specified GPU.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing training parameters.
    gpu_id : int
        The GPU ID to assign the model training to.
    model_idx : int
        Index of the ensemble model being trained.
    logger : logging.Logger
        Logger instance for debugging and tracking progress.

    Returns
    -------
    Tuple[nn.Module, np.ndarray, np.ndarray, dict]
        The trained model, training results, test results, and updated configuration.
    """
    # Set the appropriate device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    try:
        # Use a different seed for each ensemble model
        config["seed"] += model_idx

        # Train the model on the specified GPU
        best_model, config_, results_arr, test_arr = train_model_e2e(
            config,
            model=PNN,
            model_type="ensemble",
            logger=logger,
            tracker="tensor",
            write_model=False,
            device=device,
        )
        # Ensure GPU operations are completed
        torch.cuda.synchronize(device)
        print(f"Model {model_idx} training completed on GPU {gpu_id} - {device}")
        return best_model, results_arr, test_arr, config_

    finally:
        # Ensure that GPU memory is cleaned up
        torch.cuda.empty_cache()


def run_ensemble(
        config: Optional[dict] = None,
) -> Tuple[nn.Module, nn.Module, dict, dict]:
    """
    Trains and evaluates an ensemble of deep learning models.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary, by default None.

    Returns
    -------
    Tuple[nn.Module, nn.Module, dict, dict]
        The trained ensemble model, recalibration model, evaluation metrics, and plots.
    """
    ensemble_size = config.get("ensemble_size", 100)
    # parallelize = config.get("parallelize", False)
    logger = LOGGER
    best_models = []
    result_arrs = []
    test_arrs = []

    # Here we should init the wandb to track the resources
    # start wandb run
    run = wandb.init(
        config=config,
        dir=WANDB_DIR,
        mode=cast(Literal["online","offline","disabled","shared"], WANDB_MODE),
        project=config.get("wandb_project_name", "ensemble_test"),
        reinit=True,
    )

    assign_wandb_tags(run, config)

    config_ = config
    for idx in range(ensemble_size):
        best_model, config_tmp, results_arr, test_arr = train_model_e2e(
            config,
            model=PNN,
            model_type="ensemble",
            logger=logger,
            tracker="tensor",
            write_model=False,
        )
        best_models.append(best_model)
        config["seed"] += 1
        result_arrs.append(results_arr)
        test_arrs.append(test_arr)
        if idx == 0:
            config_ = config_tmp

    assert config_ is not None
    process_results_arrs(result_arrs, test_arrs, config_, logger)

    logger.debug(f"{len(best_models)=}")
    ensemble_model = EnsembleDNN(config_, model_list=best_models).to(DEVICE)

    # we should save the best_models here
    config_["model_name"] = post_training_save_model(
        ensemble_model,
        config_,
        model_type="ensemble",
        tracker="wandb",
        run=run,
        logger=logger,
        write_model=True,
    )

    dataloaders = get_dataloader(config, device=DEVICE, logger=LOGGER)

    preds, labels, alea_vars = predict(
        ensemble_model, dataloaders["test"], device=DEVICE
    )

    # Then comes the predict metrics part
    metrics, plots, uct_logger = evaluate_predictions(
        config_,
        preds,
        labels,
        alea_vars,
        "ensemble",
        logger,
        wandb_push=False,
        verbose=True,
    )

    # RECALIBRATION # Get Calibration / Validation Set
    preds_val, labels_val, alea_vars_val = predict(
        ensemble_model, dataloaders["val"], device=DEVICE
    )
    iso_recal_model = recalibrate_model(
        preds_val,
        labels_val,
        alea_vars_val,
        preds,
        labels,
        alea_vars,
        config=config_,
        uct_logger=uct_logger,
    )

    uct_logger.wandb_log()
    wandb.finish()

    return ensemble_model, iso_recal_model, metrics, plots


def run_ensemble_wrapper(**kwargs):
    """
    Wrapper function for running an ensemble of deep learning models.

    Parameters
    ----------
    kwargs : dict
        Additional configuration parameters.

    Returns
    -------
    Tuple[nn.Module, nn.Module, dict, dict]
        The trained ensemble model, recalibration model, evaluation metrics, and plots.
    """
    global LOGGER
    LOGGER = create_logger(name="ensemble", file_level="debug", stream_level="info")
    config = get_model_config(model_type="ensemble", **kwargs)
    return run_ensemble(config)

# if __name__ == "__main__":
#     ensemble_model, iso_recal_model, metrics, plots = run_ensemble_wrapper(
#         data_name="papyrus",
#         activity_type="kx",
#         n_targets=-1,
#         descriptor_protein="ankh-large",
#         descriptor_chemical="ecfp2048",
#         median_scaling=False,
#         split_type="time",
#         ext="pkl",
#         task_type="regression",
#         wandb_project_name="ensemble-test",
#         ensemble_size=5,
#         epochs=5,
#         seed=440,
#     )
#     #
#     print("Done!")
