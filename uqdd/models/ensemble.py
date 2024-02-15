# get today's date as yyyy/mm/dd format
import os
from typing import Union
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from uqdd.models.models_utils import set_seed, get_model_config, get_datasets, get_tasks
from uqdd.models.models_utils import (
    build_loader,
    build_optimizer,
    MultiTaskLoss,
    save_models,
)
from uqdd.models.models_utils import UCTMetricsTable, process_preds

from uqdd.models.baselines import MTBaselineDNN, run_epoch, predict

# get today's date as yyyy/mm/dd format
from datetime import date

today = date.today()
today = today.strftime("%Y%m%d")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
print(torch.version.cuda) if device == "cuda" else None

LOG_DIR = os.environ.get("LOG_DIR")
DATA_DIR = os.environ.get("DATA_DIR")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
CONFIG_DIR = os.environ.get("CONFIG_DIR")

wandb_mode = "online"  # 'data/papyrus_filtered_high_quality_xc50_01_standardized.csv'


def build_ensemble(config=wandb.config):
    ensemble_models = []
    try:
        seed = config.seed
    except AttributeError:
        seed = 42
    # deterministic cuda algorithms
    torch.backends.cudnn.deterministic = True

    for _ in range(config.ensemble_size):
        set_seed(seed)
        model = MTBaselineDNN(
            config.input_dim,
            config.hidden_dim_1,
            config.hidden_dim_2,
            config.hidden_dim_3,
            config.output_dim,
            config.dropout,
        )
        ensemble_models.append(model)
        seed += 1

    return ensemble_models


def train_model(
    model_idx,
    ensemble_models,
    train_loader,
    val_loader,
    test_loader,
    config,
    **kwargs
    # config=None,
):
    """
    Train a single model from the ensemble.

    Parameters
    ----------
    model_idx : int
        Index of the model in the ensemble.
    ensemble_models : list
        List of models in the ensemble.
    train_loader : torch.utils.data.DataLoader
        Training data loader.
    val_loader : torch.utils.data.DataLoader
        Validation data loader.
    test_loader : torch.utils.data.DataLoader
        Test data loader.
    config : wandb.config
        Configuration object containing hyperparameters and settings.
    **kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    best_model : torch.nn.Module
        Best trained model based on validation loss.
    """
    # config = wandb.config if config is None else config

    # Get the model for the given model_idx
    model = ensemble_models[model_idx]
    # Move the model to the device
    model.to(device)

    # Define the loss function
    loss_fn = MultiTaskLoss(
        loss_type=config.loss,
        reduction="none",
    )
    # Define the optimizer with weight decay and learning rate scheduler
    optimizer = build_optimizer(
        model, config.optimizer, config.learning_rate, config.weight_decay
    )
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
    best_model = None
    for epoch in tqdm(range(config.num_epochs), desc="Epochs"):
        try:
            epoch, train_loss, val_loss, val_rmse, val_r2, val_evs = run_epoch(
                model,
                train_loader,
                val_loader,
                loss_fn,
                optimizer,
                lr_scheduler,
                epoch=epoch,
            )
            wandb.log(
                data={
                    f"epoch": epoch,
                    f"model{model_idx}/train_loss": train_loss,
                    f"model{model_idx}/val_loss": val_loss,
                    f"model{model_idx}/val_rmse": val_rmse,
                    f"model{model_idx}/val_r2": val_r2,
                    f"model{model_idx}/val_evs": val_evs,
                },
                # step=epoch
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
            raise Exception(
                f"The following exception occurred inside the epoch loop {e}"
            )

        # Save the best model
        # save_models(config, best_model, model_idx)

    predictions = predict(best_model, test_loader, return_targets=False)

    return best_model, predictions


def run_ensemble(
    datasets=None,
    config: Union[str, dict] = "uqdd/config/ensemble/ensemble.json",
    activity: str = "xc50",
    split: str = "random",
    wandb_project_name: str = "multitask-learning-ensemble",
    ensemble_size: int = 100,
    seed: int = 42,
    **kwargs,
):
    # Load the config
    config = get_model_config(
        config=config,
        activity=activity,
        split=split,
        ensemble_size=ensemble_size,
        **kwargs,
    )
    # Load the dataset
    if datasets is None:
        datasets = get_datasets(activity=activity, split=split)
    # Get tasks names:
    tasks = get_tasks(activity=activity, split=split)

    # Initialize wandb for the ensemble models
    with wandb.init(
        dir=LOG_DIR,
        mode=wandb_mode,
        project=wandb_project_name,
        config=config,
        name=f"{today}_ensemble_{activity}_{split}",
    ):
        config = wandb.config

        # Initialize the table to store the metrics
        uct_metrics_logger = UCTMetricsTable(model_type="ensemble", config=config)

        # Define the data loaders
        train_loader, val_loader, test_loader = build_loader(
            datasets, config.batch_size, config.input_dim
        )
        # Define the ensemble models
        config.seed = seed  # TODO FIX THIS
        ensemble_models = build_ensemble(config=config)
        # Initialize lists to store the results
        best_models = []
        predictions = []
        # results = []
        _, targets = predict(
            ensemble_models[0].to(device), test_loader, return_targets=True
        )
        # Train the ensemble models
        for model_idx in tqdm(range(len(ensemble_models)), desc="Ensemble models"):
            # Train the model
            best_model, preds = train_model(
                model_idx,
                ensemble_models,
                train_loader,
                val_loader,
                test_loader,
                config,
            )
            # Store the results of the model
            predictions.append(preds)
            best_models.append(best_model)

        # Save Best Ensemble Models
        best_models = nn.ModuleList(best_models)
        save_models(config, best_models, f"{wandb.run.name}_ensemble_model", onnx=False)

        # Ensemble the predictions
        ensemble_preds = torch.stack(predictions, dim=2)
        # Process ensemble predictions
        y_pred, y_std, y_true = process_preds(ensemble_preds, targets, None)

        # Calculate and log the metrics
        # task_name =
        metrics = uct_metrics_logger(
            y_pred=y_pred, y_std=y_std, y_true=y_true, task_name="All 20 Targets"
        )
        for task_idx in range(len(tasks)):
            task_y_pred, task_y_std, task_y_true = process_preds(
                ensemble_preds, targets, task_idx=task_idx
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


if __name__ == "__main__":
    # datasets = get_datasets('xc50', 'random')
    test_config = {
        "activity": "xc50",
        "batch_size": 64,
        "dropout": 0.1,
        "early_stop": 100,
        "hidden_dim_1": 2048,
        "hidden_dim_2": 256,
        "hidden_dim_3": 256,
        "input_dim": 2048,
        "learning_rate": 0.01,
        "loss": "huber",
        "lr_scheduler": "ReduceLROnPlateau",  # TODO not necessary anymore
        "lr_factor": 0.5,
        "lr_patience": 20,
        "num_epochs": 3,
        "num_tasks": 20,
        "optimizer": "SGD",
        "output_dim": 20,
        "weight_decay": 0.001,
        "seed": 42,
        "split": "random",
        "ensemble_size": 3,
    }

    # test_loss, test_predictions = \
    run_ensemble(
        config=test_config,  # os.path.join(CONFIG_DIR, 'ensemble/ensemble.json'),
        activity="xc50",
        split="random",
        ensemble_size=5,
        # ensemble_method='fusion',
        wandb_project_name="mtl-ensemble-test",
        seed=42,
    )
    #
    # print(test_loss)
    # print(test_predictions)
