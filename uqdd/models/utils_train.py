from functools import partial

import wandb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from uqdd import TODAY, DEVICE, LOGS_DIR, WANDB_MODE

from uqdd.models.utils_models import (
    get_model_config,
    get_sweep_config,
    build_loader,
    build_optimizer,
    build_loss,
    build_lr_scheduler,
    save_models,
    calc_regr_metrics,
    set_seed,
    MultiTaskLoss,
)


def train(model, dataloader, loss_fn, optimizer, device=DEVICE):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, loss_fn, device=DEVICE):
    model.eval()
    total_loss = 0.0
    targets_all = []
    outputs_all = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), targets)
            total_loss += loss.item() * inputs.size(0)

            targets_all.append(targets)
            outputs_all.append(outputs)

        total_loss /= len(dataloader.dataset)
        targets_all = torch.cat(targets_all, dim=0)
        outputs_all = torch.cat(outputs_all, dim=0)
        # Calculate metrics
        rmse, r2, evs = calc_regr_metrics(targets_all, outputs_all)

    return total_loss, rmse, r2, evs


def predict(
    model,
    dataloader,
    return_targets=False,
    device=DEVICE,
):
    model.eval()
    outputs_all = []
    targets_all = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs_all.append(outputs.squeeze())
            if return_targets:
                targets_all.append(targets)
    outputs_all = torch.cat(outputs_all, dim=0).cpu()
    if return_targets:
        targets_all = torch.cat(targets_all, dim=0).cpu()
        return outputs_all, targets_all
    return outputs_all


def initial_evaluation(model, train_loader, val_loader, loss_fn, device=DEVICE):
    val_loss, val_rmse, val_r2, val_evs = evaluate(model, val_loader, loss_fn, device)
    train_loss, _, _, _ = evaluate(model, train_loader, loss_fn, device)
    return train_loss, val_loss, val_rmse, val_r2, val_evs


def run_one_epoch(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    lr_scheduler,
    epoch=0,
    device=DEVICE,
):
    """
    Run a single epoch of training and evaluation.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained and evaluated.
    train_loader : torch.utils.data.DataLoader
        Data Loader for training data.
    val_loader : torch.utils.data.DataLoader
        Data Loader for validation data.
    loss_fn : torch.nn.Module
        Loss function used for training and evaluation.
    optimizer : torch.optim.Optimizer
        Optimizer for model parameter updates.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    epoch : int, optional
        Current epoch number. Default is 0.
    device : torch.device, optional
        Device to run the model on. Default is DEVICE var.

    Returns
    -------
    float
        Validation loss for the epoch.
    """
    if epoch == 0:
        # Perform evaluation before training starts (epoch 0)
        train_loss, val_loss, val_rmse, val_r2, val_evs = initial_evaluation(
            model, train_loader, val_loader, loss_fn, device
        )

    else:
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_rmse, val_r2, val_evs = evaluate(
            model, val_loader, loss_fn, device
        )

        # Update the learning rate
        lr_scheduler.step(val_loss)

    return epoch, train_loss, val_loss, val_rmse, val_r2, val_evs


def wandb_epoch_logger(epoch, train_loss, val_loss, val_rmse, val_r2, val_evs):
    wandb.log(
        data={
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/rmse": val_rmse,
            "val/r2": val_r2,
            "val/evs": val_evs,
        }
    )


def wandb_test_logger(test_loss, test_rmse, test_r2, test_evs):
    wandb.log(
        data={
            "test/loss": test_loss,
            "test/rmse": test_rmse,
            "test/r2": test_r2,
            "test/evs": test_evs,
        }
    )


def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    config,
    n_targets=-1,
    seed=42,
    device=DEVICE,
):
    try:
        set_seed(seed)
        multitask = n_targets > 1
        model = model.to(device)
        best_model = model

        best_val_loss = float("inf")
        early_stop_counter = 0

        optimizer = build_optimizer(
            model, config.optimizer, config.learning_rate, config.weight_decay
        )
        loss_fn = build_loss(config.loss, reduction="none", mt=multitask)
        lr_scheduler = build_lr_scheduler(
            optimizer,
            config.lr_scheduler,
            config.lr_scheduler_patience,
            config.lr_scheduler_factor,
        )

        # "none", "mean", "sum"
        for epoch in tqdm(range(config.epochs + 1)):
            try:
                epoch, train_loss, val_loss, val_rmse, val_r2, val_evs = run_one_epoch(
                    model,
                    train_loader,
                    val_loader,
                    loss_fn,
                    optimizer,
                    lr_scheduler,
                    epoch=epoch,
                    device=device,
                )
                wandb_epoch_logger(
                    epoch, train_loss, val_loss, val_rmse, val_r2, val_evs
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    best_model = model
                else:
                    early_stop_counter += 1
                    if early_stop_counter > config.early_stop:
                        break
            except Exception as e:
                raise RuntimeError(
                    f"The following exception occurred inside the epoch loop {e}"
                )
        save_models(best_model, model_name=model_name, onnx=True)
        return best_model, loss_fn
    except Exception as e:
        raise RuntimeError(f"The following exception occurred in train_model {e}")


def run_model(
    config,
    datasets,
    model,
    model_name,
    n_targets=-1,
    seed=42,
    device=DEVICE,
):
    # Load the dataset
    train_loader, val_loader, test_loader = build_loader(
        datasets, config.batch_size, shuffle=True
    )

    # Train the model
    best_model, loss_fn = train_model(
        model,
        model_name,
        train_loader,
        val_loader,
        config,
        n_targets,
        seed,
        device,
    )

    # Testing metrics on the best model
    test_loss, test_rmse, test_r2, test_evs = evaluate(
        best_model, test_loader, loss_fn, device
    )

    wandb_test_logger(test_loss, test_rmse, test_r2, test_evs)

    return best_model, test_loss


# def _train_model(
#     model, train_loader, val_loader, config=wandb.config, seed=42, device=DEVICE
# ):
#     try:
#         # set a random seed for reproducibility
#         set_seed(seed)
#
#         # Load the model
#         model = model.to(device)
#
#         # Temporarily initialize best_model
#         best_model = model
#
#         # Define the loss function
#         loss_fn = MultiTaskLoss(loss_type=config.loss, reduction="none")
#
#         # Define the optimizer with weight decay and learning rate scheduler
#         optimizer = build_optimizer(
#             model, config.optimizer, config.learning_rate, config.weight_decay
#         )
#
#         # Define Learning rate scheduler
#         lr_scheduler = ReduceLROnPlateau(
#             optimizer,
#             mode="min",
#             factor=config.lr_factor,
#             patience=config.lr_patience,
#             verbose=True,
#         )
#
#         # Train the model
#         best_val_loss = float("inf")
#         early_stop_counter = 0
#         # Train the model
#         for epoch in tqdm(range(config.num_epochs + 1)):
#             try:
#                 epoch, train_loss, val_loss, val_rmse, val_r2, val_evs = run_epoch(
#                     model,
#                     train_loader,
#                     val_loader,
#                     loss_fn,
#                     optimizer,
#                     lr_scheduler,
#                     epoch=epoch,
#                 )
#                 # Log the metrics
#                 wandb.log(
#                     data={
#                         "epoch": epoch,
#                         "train/loss": train_loss,
#                         "val/loss": val_loss,
#                         "val/rmse": val_rmse,
#                         "val/r2": val_r2,
#                     }
#                 )
#                 # Early stopping
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     early_stop_counter = 0
#                     # Save the best model - dropped to avoid memory issues
#                     # Update the best model and its performance
#                     best_model = model
#
#                 else:
#                     early_stop_counter += 1
#                     if early_stop_counter > config.early_stop:
#                         break
#
#             except Exception as e:
#                 raise RuntimeError(
#                     f"The following exception occurred inside the epoch loop {e}"
#                 )
#
#         # Save the best model
#         save_models(config, best_model)
#         return best_model, loss_fn
#
#     except Exception as e:
#         raise Exception(f"The following exception occurred in train_model {e}")
#

#
# def train(
#         model,
#         loader,
#         optimizer,
#         loss_fn,
# ):
#     model.train()
#     total_loss = 0.0
#     for i, (inputs, targets) in enumerate(loader):
#         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
#
#         # Zero the parameter gradients
#         optimizer.zero_grad()
#         # ➡ Forward pass
#         outputs = model(inputs)
#
#         # loss calculation
#         loss = loss_fn(outputs, targets)
#
#         # ⬅ Backward pass + weight update
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     total_loss /= len(loader)
#     return total_loss
