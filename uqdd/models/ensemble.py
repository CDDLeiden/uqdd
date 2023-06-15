__author__ = "Bola Khalil"
__supervisor__ = "Kajetan Schweighofer"
__copyright__ = "Copyright 2022, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__license__ = "All rights reserved, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__version__ = "0.0.1"
__maintainer__ = "Bola Khalil"
__email__ = "bkhalil@its.jnj.com"
__status__ = "Development"

# get today's date as yyyy/mm/dd format
import os
import pickle
import sys;sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import wandb
from tqdm import tqdm
import concurrent.futures

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from uqdd.models.models_utils import set_seed, get_config, get_datasets, get_tasks
from uqdd.models.models_utils import build_loader, build_optimizer, MultiTaskLoss, save_models
from uqdd.models.models_utils import make_uct_plots

from uqdd.models.baselines import BaselineDNN, run_epoch, predict
from uncertainty_toolbox.metrics import get_all_metrics

# get today's date as yyyy/mm/dd format
from datetime import date

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

wandb_mode = 'online'  # 'data/papyrus_filtered_high_quality_xc50_01_standardized.csv'


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
        model = BaselineDNN(
            config.input_dim,
            config.hidden_dim_1,
            config.hidden_dim_2,
            config.hidden_dim_3,
            config.output_dim,
            config.dropout
        )
        ensemble_models.append(model)
        seed += 1

    return ensemble_models


def train_model(
        model_idx,
        ensemble_models,
        train_loader,
        val_loader,
        config,
        **kwargs
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
    config : wandb.config
        Configuration object containing hyperparameters and settings.
    **kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    best_model : torch.nn.Module
        Best trained model based on validation loss.
    """
    config = wandb.config if config is None else config

    # Get the model for the given model_idx
    model = ensemble_models[model_idx]
    # Move the model to the device
    model.to(device)

    # Define the loss function
    loss_fn = MultiTaskLoss(
        loss_type=config.loss,
        reduction='none',
    )
    # Define the optimizer with weight decay and learning rate scheduler
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor,
                                     patience=config.lr_patience, verbose=True)

    # Train the model
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model = None
    for epoch in tqdm(range(config.num_epochs), desc='Epochs'):
        try:
            epoch, train_loss, val_loss, val_rmse, val_r2, val_evs = run_epoch(
                model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler, epoch=epoch
            )
            wandb.log(
                data={
                    f'model{model_idx}/epoch': epoch,
                    f'model{model_idx}/train_loss': train_loss,
                    f'model{model_idx}/val_loss': val_loss,
                    f'model{model_idx}/val_rmse': val_rmse,
                    f'model{model_idx}/val_r2': val_r2,
                    f'model{model_idx}/val_evs': val_evs,
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
            raise Exception(f"The following exception occurred inside the epoch loop {e}")

        # Save the best model
        # save_models(config, best_model, model_idx)

    return best_model


def calculate_metrics(y_pred, y_std, y_true, task_name, activity, split):
    """
    Calculate metrics for the predictions.

    Parameters
    ----------
    y_pred : ndarray
        (Mean of) Predicted values.
    y_std : ndarray
        Standard deviation of predicted values.
    y_true : ndarray
        True values.
    task_name : str
        Name of the task.
    activity : str
        Activity name.
    split : str
        Split name.

    Returns
    -------
    metrics : dict
        Dictionary containing calculated metrics.
    img     : wandb.Image
        Image of the UCT plot - ready for logging
    """
    metrics = get_all_metrics(
        y_pred=y_pred,
        y_std=y_std,
        y_true=y_true,
        num_bins=100,
        resolution=99,
        scaled=True,
        verbose=False,
    )

    figures_path = os.path.join(FIGS_DIR, "ensemble", activity, split)
    os.makedirs(figures_path, exist_ok=True)

    metrics_filepath = os.path.join(figures_path, f"{task_name}_metrics.pkl")
    with open(metrics_filepath, "wb") as file:
        pickle.dump(metrics, file)

    fig = make_uct_plots(
        y_pred,
        y_std,
        y_true,
        task_name=task_name,
        n_subset=min(500, len(y_true)),
        ylims=None,
        num_stds_confidence_bound=1.96,
        plot_save_str=os.path.join(figures_path, f"{task_name}_uct"),
        savefig=True,
    )

    img = wandb.Image(fig)

    return metrics, img


def uct_metrics_logger(
        uct_metrics_table,
        task_name,
        config,
        metrics,
        img
):
    """
    Log UCT metrics to the UCT metrics table.

    Parameters
    ----------
    uct_metrics_table : UCTMetricsTable
        UCT metrics table object for logging.
    task_name : str
        Name of the task.
    config : wandb.config
        Configuration object.
    metrics : dict
        Dictionary containing calculated metrics.
    img : wandb.Image
        Image of the UCT plot.

    Returns
    -------
    None
    """
    uct_metrics_table.add_data(
        task_name,
        config.activity,
        config.split,
        metrics["accuracy"]["rmse"],
        metrics["accuracy"]["r2"],
        metrics["accuracy"]["mae"],
        metrics["accuracy"]["mdae"],
        metrics["accuracy"]["marpd"],
        metrics["accuracy"]["corr"],
        metrics["avg_calibration"]["rms_cal"],
        metrics["avg_calibration"]["ma_cal"],
        metrics["avg_calibration"]["miscal_area"],
        metrics["sharpness"]["sharp"],
        metrics["scoring_rule"]["nll"],
        metrics["scoring_rule"]["crps"],
        metrics["scoring_rule"]["check"],
        metrics["scoring_rule"]["interval"],
        img
    )


def process_ensemble_preds(
        ensemble_preds,
        targets,
        task_idx=None
):

    # Get the predictions mean and std
    ensemble_preds_mu = ensemble_preds.mean(dim=2)
    ensemble_preds_std = ensemble_preds.std(dim=2)

    if task_idx is not None:
        ensemble_preds_mu = ensemble_preds_mu[:, task_idx]
        ensemble_preds_std = ensemble_preds_std[:, task_idx]
        targets = targets[:, task_idx]
    else:
        # flatten
        ensemble_preds_mu = torch.flatten(ensemble_preds_mu.transpose(0, 1))
        ensemble_preds_std = torch.flatten(ensemble_preds_std.transpose(0, 1))
        targets = torch.flatten(targets.transpose(0, 1))

    # nan mask filter
    nan_mask = ~torch.isnan(targets)
    ensemble_preds_mu = ensemble_preds_mu[nan_mask]
    ensemble_preds_std = ensemble_preds_std[nan_mask]
    targets = targets[nan_mask]

    # convert to numpy and to cpu
    y_pred = ensemble_preds_mu.cpu().numpy()
    y_std = ensemble_preds_std.cpu().numpy()
    y_true = targets.cpu().numpy()

    return y_pred, y_std, y_true


def run_ensemble(
        datasets=None,
        config='uqdd/config/ensemble/ensemble.json',
        activity='xc50',
        split='random',
        wandb_project_name='multitask-learning-ensemble',
        ensemble_size=100,
        seed=42,
        **kwargs
):
    # Load the config
    config = get_config(config=config, activity=activity, split=split, ensemble_size=ensemble_size, **kwargs)  #

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
        uct_metrics_table = wandb.Table(
            columns=[
                "Target",
                "Activity",
                "Split",
                "RMSE",
                "R2",
                "MAE",
                "MADAE",
                "MARPD",
                "Correlation",
                "RMS Calibration",
                "MA Calibration",
                "Miscalibration Area",
                "Sharpness",
                "NLL",
                "CRPS",
                "Check",
                "Interval",
                "UCT plots"
            ])

        config = wandb.config

        # Define the data loaders
        train_loader, val_loader, test_loader = build_loader(datasets, config.batch_size, config.input_dim)
        # Define the ensemble models
        config.seed = seed  # TODO FIX THIS
        ensemble_models = build_ensemble(config=config)
        #
        best_models = []
        predictions = []
        for model_idx in tqdm(range(len(ensemble_models)), desc='Ensemble models'):
            # Train the model
            best_model = train_model(
                model_idx,
                ensemble_models,
                train_loader,
                val_loader,
                config
            )
            # Test the model
            if model_idx == 0:
                model_y_preds, targets = predict(best_model, test_loader, return_targets=True)
            else:
                model_y_preds = predict(best_model, test_loader, return_targets=False)

            predictions.append(model_y_preds)
            best_models.append(best_model)

        # Save Best Ensemble Models
        best_models = nn.ModuleList(best_models)
        save_models(config, best_models, f"{wandb.run.name}_ensemble_model", onnx=False)

        # Ensemble the predictions
        ensemble_preds = torch.stack(predictions, dim=2)  # torch.Size([datapoints, tasks, ensemble_size])

        # Process ensemble predictions
        task_name = "All 20 Targets"
        y_pred, y_std, y_true = process_ensemble_preds(ensemble_preds, targets, None)

        # Calculate the metrics
        metrics, img = calculate_metrics(
            y_pred=y_pred,
            y_std=y_std,
            y_true=y_true,
            task_name=task_name,
            activity=activity,
            split=split
        )

        uct_metrics_logger(uct_metrics_table, task_name, config, metrics, img)

        for task_idx in range(len(tasks)):
            task_name = tasks[task_idx]
            task_y_pred, task_y_std, task_y_true = process_ensemble_preds(ensemble_preds, targets, task_idx=task_idx)

            # Calculate the metrics
            metrics, img = calculate_metrics(
                y_pred=task_y_pred,
                y_std=task_y_std,
                y_true=task_y_true,
                task_name=task_name,
                activity=activity,
                split=split
            )
            uct_metrics_logger(uct_metrics_table, task_name, config, metrics, img)

        wandb.log(
            data={
                f'uct_metrics': uct_metrics_table,
            }
        )


if __name__ == '__main__':
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
        "ensemble_size": 3
    }

    test_loss, test_predictions = run_ensemble(
        config=test_config,  # os.path.join(CONFIG_DIR, 'ensemble/ensemble.json'),
        activity='xc50',
        split='random',
        ensemble_size=5,
        # ensemble_method='fusion',
        wandb_project_name='mtl-ensemble-test',
        seed=42,
    )

    print(test_loss)
    print(test_predictions)
