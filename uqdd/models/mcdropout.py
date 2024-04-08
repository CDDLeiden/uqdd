import argparse

import wandb
import torch
import torch.nn as nn
from uqdd.models.baseline import BaselineDNN
from uqdd.utils import create_logger, parse_list

from uqdd.models.utils_train import (
    train_model_e2e,
    predict_uc_metrics,
)

from uqdd.models.utils_models import (
    get_model_config,
    get_sweep_config,
)


import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from datetime import date
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from uqdd.models.models_utils import set_seed, get_model_config, get_datasets, get_tasks
from uqdd.models.models_utils import (
    build_loader,
    build_optimizer,
    MultiTaskLoss,
    save_models,
)
from uqdd.models.models_utils import UCTMetricsTable, process_preds
from uqdd.models.baseline import train_model

from functools import partial
import numpy as np
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

# wandb_dir = '../logs/'
wandb_mode = "online"  # 'offline')))))


def predict(
    model,
    test_loader,
    num_samples=100,
    return_targets=False,
):
    model.train()  # Enable dropout
    outputs_all = []
    targets_all = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(
            tqdm(test_loader, desc="MC prediction", total=len(test_loader))
        ):
            inputs = inputs.to(device)
            outputs = torch.stack(
                [model(inputs) for _ in range(num_samples)], dim=2
            )  # Multiple forward passes
            outputs_all.append(outputs.cpu().detach())
            if return_targets:
                targets_all.append(targets.cpu().detach())

    model.eval()  # Disable dropout
    outputs_all = torch.cat(outputs_all, dim=0)
    if return_targets:
        targets_all = torch.cat(targets_all, dim=0)
        return outputs_all, targets_all
    return outputs_all


# def mc_uncertainty_estimate(outputs):
#     outputs = outputs.cpu().detach()
#     y_mean = outputs.mean(dim=2).numpy()
#     y_std = outputs.std(dim=2).numpy()
#     # y_var = outputs.var(dim=0)
#     # y_std = torch.sqrt(y_var)
#
#     return y_mean, y_std # , y_var


def plot_predictions(y_true, y_pred, y_std):
    plt.figure(figsize=(12, 6))
    plt.errorbar(y_true, y_pred, yerr=y_std, fmt="o")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.grid()
    plt.show()


def plot_uncertainty_distribution(y_std):
    plt.figure(figsize=(12, 6))
    plt.hist(y_std, bins=50)
    plt.xlabel("Uncertainty")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


def run_mcdropout(
    datasets=None,
    config=os.path.join(CONFIG_DIR, "baseline", "baseline_xc50_random_best.json"),
    activity="xc50",
    split="random",
    wandb_project_name="multitask-learning-mcdropout",
    num_samples=100,
    seed=42,
    **kwargs,
):
    # load the config
    config = get_model_config(
        config=config, activity=activity, split=split, num_samples=num_samples, **kwargs
    )
    # Load the dataset
    if datasets is None:
        datasets = get_datasets(activity=activity, split=split)

    # Get tasks names:
    tasks = get_tasks(activity=activity, split=split)

    # Initialize wandb
    with wandb.init(
        dir=LOG_DIR,
        mode=wandb_mode,
        project=wandb_project_name,
        config=config,
        name=f"{today}_mcdropout_{activity}_{split}",
    ):
        config = wandb.config

        # Initialize the table to store the metrics
        uct_metrics_logger = UCTMetricsTable(model_type="mcdropout", config=config)

        # Define the data loaders
        train_loader, val_loader, test_loader = build_loader(
            datasets, config.batch_size, config.input_dim
        )

        # Train the baseline model
        set_seed(seed)
        # Train the model
        # TODO : Add the option to load a pretrained model and train it further or just evaluate it
        best_model, loss_fn = train_model(
            train_loader, val_loader, config=config, seed=seed
        )

        # Perform MC Dropout during predictions
        preds, targets = predict(
            best_model.to(device),
            test_loader,
            num_samples=config.num_samples,
            return_targets=True,
        )

        # Process the predictions
        y_pred, y_std, y_true = process_preds(preds, targets, None)

        # Calculate and log the metrics
        metrics = uct_metrics_logger(
            y_pred=y_pred, y_std=y_std, y_true=y_true, task_name="All 20 Targets"
        )

        for task_idx in range(len(tasks)):
            task_y_pred, task_y_std, task_y_true = process_preds(
                preds, targets, task_idx=task_idx
            )

            task_name = tasks[task_idx]
            metrics = uct_metrics_logger(
                y_pred=task_y_pred,  # y_pred[:, task_idx],
                y_std=task_y_std,  # y_std[:, task_idx],
                y_true=task_y_true,  # targets[:, task_idx],
                task_name=task_name,
            )

        uct_metrics_logger.wandb_log()


if __name__ == "__main__":
    run_mcdropout(wandb_project_name="mtl-mcdropout-test")
