"""
Training utilities for models.

This module provides functions for training, evaluating, and predicting with neural network models,
including support for evidential regression and uncertainty quantification.
"""

import logging
from pathlib import Path
from typing import (
    Tuple,
    Optional,
    Dict,
    Union,
    Callable,
    Any,
    Type,
    List,
    LiteralString,
)

import numpy as np
import torch
import wandb
from numpy import ndarray, dtype
from tqdm import tqdm

from uqdd import DEVICE, WANDB_DIR, WANDB_MODE, TODAY, FIGS_DIR
from uqdd.data.utils_data import get_tasks
from uqdd.models.loss import build_loss
from uqdd.models.utils_metrics import (
    calc_regr_metrics,
    MetricsTable,
    process_preds,
    create_df_preds,
    calc_alea_epi_mean_var_notnan,
    recalibrate,
    get_calib_props,
)
from uqdd.models.utils_models import (
    build_loader,
    build_optimizer,
    build_lr_scheduler,
    save_model,
    set_seed,
    build_datasets,
    get_desc_len,
    compute_pnorm,
    compute_gnorm,
    ckpt,
    load_ckpt,
    get_model_name,
    get_data_specific_path,
)
from uqdd.utils import create_logger


def evidential_processing(
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process outputs of an evidential regression model to compute aleatoric and epistemic uncertainty.

    Parameters
    ----------
    outputs : tuple of torch.Tensor
        Model outputs consisting of ``(mu, v, alpha, beta)``.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        Aleatoric (beta/(alpha-1)) and epistemic (sqrt(beta/(v*(alpha-1)))) uncertainty.
    """
    mu, v, alpha, beta = outputs
    alea_vars = beta / (alpha - 1)  # aleatoric
    epi_vars = torch.sqrt(beta / (v * (alpha - 1)))  # epistemic
    return alea_vars, epi_vars


def model_forward(
        model: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        targets: torch.Tensor,
        lossfname: str = "evidential_regression",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Tuple]:
    """
    Perform a forward pass and prepare arguments for the specified loss function.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model.
    inputs : tuple of torch.Tensor
        Input tensors (single or multi-input).
    targets : torch.Tensor
        Target values tensor.
    lossfname : str, optional
        Loss function name ("evidential_regression", "gaussnll", etc.). Default is ``"evidential_regression"``.

    Returns
    -------
    tuple
        ``(outputs, alea_vars or None, epi_vars or None, loss_args)`` ready for loss computation.
    """
    if lossfname.lower() == "evidential_regression":
        outputs = model(inputs)
        alea_vars, epi_vars = evidential_processing(outputs)
        args = (outputs, targets)
        return outputs, alea_vars, epi_vars, args
    elif lossfname.lower() == "gaussnll":
        outputs, vars_ = model(inputs)
        args = (outputs, targets, vars_)
        return outputs, vars_, None, args
    else:
        outputs, vars_ = model(inputs)
        args = (outputs, targets)
        return outputs, vars_, None, args


def train(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        device: torch.device = DEVICE,
        epoch: int = 0,
        max_norm: Optional[float] = None,
        lossfname: str = "evidential_regression",
        tracker: str = "wandb",
        subset: str = "train",
) -> ndarray[Any, dtype[Any]] | float | Any:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model.
    dataloader : torch.utils.data.DataLoader
        DataLoader for training data.
    loss_fn : callable
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    device : torch.device, optional
        Device for training. Default is ``DEVICE``.
    epoch : int, optional
        Current training epoch. Default is ``0``.
    max_norm : float or None, optional
        Gradient clipping norm. Default is ``None``.
    lossfname : str, optional
        Loss function name. Default is ``"evidential_regression"``.
    tracker : str, optional
        Tracking tool for logging ("wandb" or "tensor"). Default is ``"wandb"``.
    subset : str, optional
        Subset name (e.g., "train"). Default is ``"train"``.

    Returns
    -------
    float or numpy.ndarray
        Average epoch loss (float). If tracker is "tensor", returns a numpy array of tracked values.
    """
    model.train()
    total_loss = 0.0
    targets_all = []
    outputs_all = []
    vars_all = []
    epis_all = []
    num_batches = len(dataloader)

    for inputs, targets in dataloader:
        inputs = tuple(x.to(device) for x in inputs)
        targets = targets.to(device)
        optimizer.zero_grad()

        outputs, alea_vars, epi_vars, args = model_forward(
            model, inputs, targets, lossfname=lossfname
        )
        loss = loss_fn(*args)
        loss.backward()

        vars_all.append(alea_vars.detach())
        epis_all.append(epi_vars.detach()) if epi_vars is not None else None

        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        total_loss += loss.item()
        targets_all.append(targets)
        outputs = (
            outputs[0] if lossfname.lower() == "evidential_regression" else outputs
        )
        outputs_all.append(outputs)

    total_loss /= num_batches
    targets_all = torch.cat(targets_all, dim=0)
    outputs_all = torch.cat(outputs_all, dim=0)

    train_rmse, train_r2, train_evs = calc_regr_metrics(targets_all, outputs_all)
    pnorm = compute_pnorm(model)
    gnorm = compute_gnorm(model)
    vars_all = torch.cat(vars_all, dim=0)
    vars_mean, vars_var = calc_alea_epi_mean_var_notnan(vars_all, targets_all)

    if tracker.lower() == "wandb":
        data = {
            f"{subset}/loss": total_loss,
            f"{subset}/rmse": train_rmse,
            f"{subset}/r2": train_r2,
            f"{subset}/evs": train_evs,
            f"{subset}/alea_mean": vars_mean.item(),
            f"{subset}/alea_var": vars_var.item(),
            "model/pnorm": pnorm,
            "model/gnorm": gnorm,
        }

        if lossfname.lower() == "evidential_regression":
            epis_all = torch.cat(epis_all, dim=0)
            epis_mean, epis_var = calc_alea_epi_mean_var_notnan(epis_all, targets_all)
            data[f"{subset}/epis_mean"] = epis_mean.item()
            data[f"{subset}/epis_var"] = epis_var.item()

        wandb.log(
            data=data,
            step=epoch,
        )

    elif tracker.lower() == "tensor":
        vals = [
            epoch,
            total_loss,
            train_rmse,
            train_r2,
            train_evs,
            vars_mean.item(),
            vars_var.item(),
            pnorm,
            gnorm,
        ]
        if lossfname.lower() == "evidential_regression":
            epis_all = torch.cat(epis_all, dim=0)
            epis_mean, epis_var = calc_alea_epi_mean_var_notnan(epis_all, targets_all)
            vals.extend([epis_mean.item(), epis_var.item()])

        tracked_vals = np.array(vals, dtype=np.float32)
        return tracked_vals

    return total_loss


def evaluate(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        device: torch.device = DEVICE,
        metrics_per_task: bool = False,
        subset: str = "val",
        epoch: Optional[int] = 0,
        lossfname: str = "evidential_regression",
        tracker: str = "wandb",
) -> ndarray[Any, dtype[np.generic | Any]] | ndarray[Any, dtype[Any]] | float | Any:
    """
    Evaluate the model on validation or test data.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model.
    dataloader : torch.utils.data.DataLoader
        DataLoader for evaluation data.
    loss_fn : callable
        Loss function.
    device : torch.device, optional
        Device for evaluation. Default is ``DEVICE``.
    metrics_per_task : bool, optional
        Compute metrics per task in multitask mode. Default is ``False``.
    subset : str, optional
        Subset name ("val" or "test"). Default is ``"val"``.
    epoch : int, optional
        Current evaluation epoch. Default is ``0``.
    lossfname : str, optional
        Loss function name. Default is ``"evidential_regression"``.
    tracker : str, optional
        Tracking tool ("wandb" or "tensor"). Default is ``"wandb"``.

    Returns
    -------
    float or numpy.ndarray
        Average evaluation loss (float). If tracker is "tensor", returns a numpy array of tracked values.
    """
    model.eval()
    total_loss = 0.0
    targets_all = []
    outputs_all = []
    vars_all = []
    epis_all = []
    num_batches = len(dataloader)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = tuple(x.to(device) for x in inputs)
            targets = targets.to(device)

            outputs, alea_vars, epi_vars, args = model_forward(
                model, inputs, targets, lossfname=lossfname
            )

            loss = loss_fn(*args)

            vars_all.append(alea_vars.detach())
            epis_all.append(epi_vars.detach()) if epi_vars is not None else None

            total_loss += loss.item()

            outputs = (
                outputs[0] if lossfname.lower() == "evidential_regression" else outputs
            )
            outputs_all.append(outputs)
            targets_all.append(targets)

        total_loss /= num_batches
        targets_all = torch.cat(targets_all, dim=0)
        outputs_all = torch.cat(outputs_all, dim=0)

        rmse, r2, evs = calc_regr_metrics(targets_all, outputs_all)

        vars_all = torch.cat(vars_all, dim=0)
        vars_mean, vars_var = calc_alea_epi_mean_var_notnan(vars_all, targets_all)

        if tracker.lower() == "wandb":
            data = {
                f"{subset}/loss": total_loss,
                f"{subset}/rmse": rmse,
                f"{subset}/r2": r2,
                f"{subset}/evs": evs,
                f"{subset}/alea_mean": vars_mean.item(),
                f"{subset}/alea_var": vars_var.item(),
            }

            if lossfname.lower() == "evidential_regression":
                epis_all = torch.cat(epis_all, dim=0)
                epis_mean, epis_var = calc_alea_epi_mean_var_notnan(
                    epis_all, targets_all
                )
                data[f"{subset}/epis_mean"] = epis_mean.item()
                data[f"{subset}/epis_var"] = epis_var.item()

            wandb.log(data=data, step=epoch)

        elif tracker.lower() == "tensor":
            vals = [epoch, total_loss, rmse, r2, evs, vars_mean.item(), vars_var.item()]
            (
                vals.extend([compute_pnorm(model), compute_gnorm(model)])
                if epoch == 0 and subset == "train"
                else None
            )

            if lossfname.lower() == "evidential_regression":
                epis_all = torch.cat(epis_all, dim=0)
                epis_mean, epis_var = calc_alea_epi_mean_var_notnan(
                    epis_all, targets_all
                )
                vals.extend([epis_mean.item(), epis_var.item()])
            tracked_vals = np.array(vals, dtype=np.float32)

        if metrics_per_task:
            tasks_rmse, tasks_r2, tasks_evs = calc_regr_metrics(
                targets_all, outputs_all, metrics_per_task
            )
            for task_idx in range(len(tasks_rmse)):
                if tracker.lower() == "wandb":
                    wandb.log(
                        data={
                            f"{subset}/rmse/task_{task_idx}": tasks_rmse[task_idx],
                            f"{subset}/r2/task_{task_idx}": tasks_r2[task_idx],
                            f"{subset}/evs/task_{task_idx}": tasks_evs[task_idx],
                        },
                        step=epoch,
                    )
                elif tracker.lower() == "tensor":
                    tracked_vals = np.append(
                        tracked_vals,
                        [tasks_rmse[task_idx], tasks_r2[task_idx], tasks_evs[task_idx]],
                    )
    if tracker.lower() == "tensor":
        tracked_vals = np.array(tracked_vals, dtype=np.float32)
        return tracked_vals

    return total_loss


def predict(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: Union[torch.device, str] = DEVICE,
        set_on_eval: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform predictions using a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    dataloader : torch.utils.data.DataLoader
        DataLoader for inference data.
    device : torch.device or str, optional
        Device for inference. Default is ``DEVICE``.
    set_on_eval : bool, optional
        Whether to set the model to evaluation mode. Default is ``True``.

    Returns
    -------
    (torch.Tensor, torch.Tensor, torch.Tensor)
        Predicted values, ground truth labels, and estimated uncertainties.
    """
    if set_on_eval:
        model.eval()
    outputs_all = []
    targets_all = []
    vars_all = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, total=len(dataloader), desc="Predicting"):
            inputs = tuple(x.to(device) for x in inputs)
            targets = targets.to(device)

            outputs, vars_ = model(inputs)
            vars_all.append(vars_)
            outputs_all.append(outputs)
            targets_all.append(targets)

    outputs_all = torch.cat(outputs_all, dim=0).cpu()
    targets_all = torch.cat(targets_all, dim=0).cpu()
    vars_all = torch.cat(vars_all, dim=0).cpu()
    return outputs_all, targets_all, vars_all


def evaluate_predictions(
        config: Dict[str, Any],
        preds: torch.Tensor,
        labels: torch.Tensor,
        alea_vars: torch.Tensor,
        model_type: str = "ensemble",
        logger: Optional[logging.Logger] = None,
        epi_vars: Optional[torch.Tensor] = None,
        wandb_push: bool = False,
        run_name: Optional[str] = None,
        project_name: Optional[str] = None,
        figpath: Optional[LiteralString | str | bytes | Path] = None,
        export_preds: bool = True,
        verbose: bool = True,
        csv_path: Optional[str] = None,
        nll: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
    """
    Evaluate predictions, compute UQ metrics, and optionally log results.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model and dataset settings.
    preds : torch.Tensor
        Model predictions.
    labels : torch.Tensor
        Ground truth labels.
    alea_vars : torch.Tensor
        Aleatoric uncertainty estimates.
    model_type : str, optional
        Model type (e.g., "ensemble", "pnn"). Default is ``"ensemble"``.
    logger : logging.Logger or None, optional
        Logger instance.
    epi_vars : torch.Tensor or None, optional
        Epistemic uncertainty estimates. Default is ``None``.
    wandb_push : bool, optional
        Whether to log results to Weights & Biases. Default is ``False``.
    run_name : str or None, optional
        Weights & Biases run name.
    project_name : str or None, optional
        Weights & Biases project name.
    figpath : str or Path or None, optional
        Path to save generated figures. Default is ``None``.
    export_preds : bool, optional
        Whether to export predictions as CSV. Default is ``True``.
    verbose : bool, optional
        Whether to print additional information. Default is ``True``.
    csv_path : str or None, optional
        Path to save prediction CSV file.
    nll : float or None, optional
        Negative log-likelihood value.

    Returns
    -------
    tuple
        ``(metrics_dict, plots_dict, metrics_logger)``.
    """
    data_name = config.get("data_name", "papyrus")
    activity_type = config.get("activity_type", "xc50")
    n_targets = config.get("n_targets", -1)
    multitask = config.get("MT", False)
    data_specific_path = get_data_specific_path(config, logger=logger)

    model_name = get_model_name(config)

    uct_metrics_logger = MetricsTable(
        model_type=model_type,
        config=config,
        add_plots_to_table=True,
        logger=logger,
        run_name=run_name,
        project_name=project_name,
        verbose=verbose,
        csv_path=csv_path,
    )

    y_true, y_pred, y_err, y_alea, y_eps = process_preds(
        preds, labels, alea_vars, epi_vars, None, model_type
    )
    _ = create_df_preds(
        y_true,
        y_pred,
        y_alea,
        y_err,
        y_eps,
        export_preds,
        data_specific_path,
        model_name,
        logger,
    )
    task_name = f"All {n_targets} Targets" if n_targets > 1 else "PCM"
    metrics, plots = uct_metrics_logger(
        y_pred=y_pred,
        y_std=y_alea,
        y_true=y_true,
        y_err=y_err,
        y_eps=y_eps,
        task_name=task_name,
        figpath=figpath,
        nll=nll,
    )

    props_df = get_calib_props(
        y_pred=y_pred,
        y_std=y_alea,
        y_true=y_true,
        vectorized=True,
        prop_type="interval",
        output_folder=figpath,
    )

    submetrics, subplots = uct_metrics_logger(
        y_pred=y_pred,
        y_std=y_alea,
        y_true=y_true,
        y_err=y_err,
        y_eps=y_eps,
        task_name=task_name + "_subset_100",
        n_subset=100,
        figpath=figpath,
    )

    if multitask:
        tasks = get_tasks(data_name, activity_type, n_targets)
        for task_idx in range(len(tasks)):
            task_name = tasks[task_idx]
            y_true, y_pred, y_err, y_alea, y_eps = process_preds(
                preds, labels, alea_vars, epi_vars, task_idx, model_type
            )
            taskmetrics, taskplots = uct_metrics_logger(
                y_pred=y_pred,
                y_std=y_alea,
                y_true=y_true,
                y_err=y_err,
                y_eps=y_eps,
                task_name=task_name,
                figpath=figpath,
            )
            metrics[task_name] = taskmetrics
            plots[task_name] = taskplots

            submetrics[task_name], subplots[task_name] = uct_metrics_logger(
                y_pred=y_pred,
                y_std=y_alea,
                y_true=y_true,
                y_err=y_err,
                y_eps=y_eps,
                task_name=task_name + "_subset_100",
                n_subset=100,
                figpath=figpath,
            )

    if wandb_push:
        uct_metrics_logger.wandb_log()

    return metrics, plots, uct_metrics_logger


def initial_evaluation(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        device: torch.device = DEVICE,
        epoch: int = 0,
        lossfname: str = "",
        tracker: str = "wandb",
) -> Tuple[float, float]:
    """
    Perform an initial evaluation on train and validation sets.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data.
    loss_fn : callable
        Loss function used for evaluation.
    device : torch.device, optional
        Device to run the model on. Default is ``DEVICE``.
    epoch : int, optional
        Current epoch number. Default is ``0``.
    lossfname : str, optional
        Loss function name. Default is ``""``.
    tracker : str, optional
        Tracker for logging results. Default is ``"wandb"``.

    Returns
    -------
    (float, float)
        Training loss and validation loss.
    """
    val_loss = evaluate(
        model,
        val_loader,
        loss_fn,
        device,
        False,
        "val",
        epoch,
        lossfname,
        tracker=tracker,
    )
    train_loss = evaluate(
        model,
        train_loader,
        loss_fn,
        device,
        False,
        "train",
        epoch,
        lossfname,
        tracker=tracker,
    )
    return train_loss, val_loss


def run_one_epoch(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int = 0,
        device: torch.device = DEVICE,
        max_norm: Optional[float] = None,
        lossfname: str = "",
        tracker: str = "wandb",
) -> Tuple[int, float, float]:
    """
    Execute one training epoch, including evaluation and LR scheduling.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be trained and evaluated.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation.
    loss_fn : callable
        Loss function used for training and evaluation.
    optimizer : torch.optim.Optimizer
        Optimizer for parameter updates.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler or None, optional
        Learning rate scheduler.
    epoch : int, optional
        Current epoch number. Default is ``0``.
    device : torch.device, optional
        Device to run the model on. Default is ``DEVICE``.
    max_norm : float or None, optional
        Maximum norm for gradient clipping. Default is ``None``.
    lossfname : str, optional
        Loss function name. Default is ``""``.
    tracker : str, optional
        Tracker for logging. Default is ``"wandb"``.

    Returns
    -------
    (int, float, float)
        Epoch number, training loss, and validation loss.
    """
    if epoch == 0:
        train_loss, val_loss = initial_evaluation(
            model, train_loader, val_loader, loss_fn, device, epoch, lossfname, tracker
        )

    else:
        train_loss = train(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            epoch,
            max_norm=max_norm,
            lossfname=lossfname,
            tracker=tracker,
        )
        val_loss = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            False,
            "val",
            epoch,
            lossfname,
            tracker=tracker,
        )

        if lr_scheduler is not None:
            vloss = val_loss if not isinstance(val_loss, np.ndarray) else val_loss[1]
            lr_scheduler.step(vloss)

    return epoch, train_loss, val_loss


def train_model(
        model: torch.nn.Module,
        config: Dict[str, Any],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        n_targets: int = -1,
        seed: int = 42,
        device: torch.device = DEVICE,
        logger: Optional[logging.Logger] = None,
        max_norm: Optional[float] = None,
        tracker: str = "wandb",
) -> Tuple[torch.nn.Module, Callable, np.ndarray]:
    """
    Train a model with the specified configuration.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be trained.
    config : dict
        Configuration dictionary containing hyperparameters.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data.
    n_targets : int, optional
        Number of targets for multitask learning. Default is ``-1``.
    seed : int, optional
        Random seed for reproducibility. Default is ``42``.
    device : torch.device, optional
        Device to run the model on. Default is ``DEVICE``.
    logger : logging.Logger or None, optional
        Logger instance. Default is ``None``.
    max_norm : float or None, optional
        Maximum norm for gradient clipping. Default is ``None``.
    tracker : str, optional
        Tracker for logging. Default is ``"wandb"``.

    Returns
    -------
    (torch.nn.Module, callable, numpy.ndarray)
        Trained model, loss function, and recorded training statistics.
    """
    try:
        set_seed(seed)
        multitask = n_targets > 1

        model = model.to(device)
        best_val_loss = float("inf")
        early_stop_counter = 0
        early_stop_criteria = int(config.get("early_stop", 10))
        optimizer = build_optimizer(
            model,
            config.get("optimizer", "adam"),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.0),
        )

        loss_fn = build_loss(
            config.get("loss", "mse"),
            reduction=config.get("loss_reduction", "mean"),
            lamb=config.get("lamb", 1e-2),
            mt=multitask,
        )
        lr_scheduler = build_lr_scheduler(
            optimizer,
            config.get("lr_scheduler", None),
            config.get("lr_scheduler_patience", None),
            config.get("lr_scheduler_factor", None),
        )

        results_arr = []
        random_num = np.random.randint(0, 10000000000)

        for epoch in tqdm(range(config.get("epochs", 10)), desc="Epochs"):
            try:
                lossfname = config.get("loss", "mse")
                (
                    epoch,
                    train_loss,
                    val_loss,
                ) = run_one_epoch(
                    model,
                    train_loader,
                    val_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    epoch=epoch,
                    device=device,
                    max_norm=max_norm,
                    lossfname=lossfname,
                    tracker=tracker,
                )
                vloss = val_loss if not isinstance(val_loss, np.ndarray) else val_loss[1]
                if vloss < best_val_loss:
                    best_val_loss = vloss
                    early_stop_counter = 0
                    config = ckpt(model, config)

                else:
                    early_stop_counter += 1
                    if early_stop_counter > early_stop_criteria:
                        break
                if tracker.lower() == "tensor":
                    results_arr.append(np.append(train_loss, val_loss[1:]))

            except Exception as e:
                raise RuntimeError(
                    f"The following exception occurred inside the epoch loop {e}"
                )

        best_model = load_ckpt(model, config)
        if tracker.lower() == "tensor":
            results_arr = np.stack(results_arr, axis=0)

        return best_model, loss_fn, results_arr
    except Exception as e:
        raise RuntimeError(f"The following exception occurred in train_model {e}")


def run_model(
        config: Dict[str, Any],
        model: torch.nn.Module,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        device: torch.device = DEVICE,
        logger: Optional[logging.Logger] = None,
        max_norm: Optional[float] = None,
        tracker: str = "wandb",
) -> Tuple[torch.nn.Module, float, np.ndarray]:
    """
    Run the full training and evaluation cycle.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing hyperparameters.
    model : torch.nn.Module
        Model to train and evaluate.
    dataloaders : dict of str -> torch.utils.data.DataLoader
        Dataloaders for train, validation, and test sets.
    device : torch.device, optional
        Device to run the model on. Default is ``DEVICE``.
    logger : logging.Logger or None, optional
        Logger instance. Default is ``None``.
    max_norm : float or None, optional
        Maximum norm for gradient clipping. Default is ``None``.
    tracker : str, optional
        Tracker for logging. Default is ``"wandb"``.

    Returns
    -------
    (torch.nn.Module, float, numpy.ndarray)
        Best trained model, test loss, and recorded training statistics.
    """
    if logger is None:
        logger = create_logger("run_model")
    seed = config.get("seed", 42)
    n_targets = config.get("n_targets", -1)
    mt = n_targets > 1
    best_model, loss_fn, results_arr = train_model(
        model,
        config,
        dataloaders["train"],
        dataloaders["val"],
        n_targets,
        seed,
        device,
        logger=logger,
        max_norm=max_norm,
        tracker=tracker,
    )

    test_loss = evaluate(
        best_model,
        dataloaders["test"],
        loss_fn,
        device,
        metrics_per_task=mt,
        subset="test",
        epoch=None,
        lossfname=config.get("loss", "mse"),
        tracker=tracker,
    )

    return best_model, test_loss, results_arr


def assign_wandb_tags(run: Any, config: Dict[str, Any]) -> Any:
    """
    Assign metadata tags to a Weights & Biases run.

    Parameters
    ----------
    run : Any
        Weights & Biases run object.
    config : dict
        Configuration dictionary.

    Returns
    -------
    Any
        Updated wandb run object with assigned tags.
    """
    median_scaling = config.get("median_scaling", False)
    m_tag = "median_scaling" if median_scaling else "no_median_scaling"
    mt = config.get("MT", False)
    mt_tag = "MT" if mt else "ST"
    wandb_tags = [
        config.get("model_type", "pnn"),
        config.get("data_name", "papyrus"),
        config.get("activity_type", "xc50"),
        config.get("descriptor_protein", None),
        config.get("descriptor_chemical", None),
        config.get("split_type", "random"),
        config.get("task_type", "regression"),
        m_tag,
        mt_tag,
        f"max_norm={config.get('max_norm', None)}",
        TODAY,
        config.get("tags", None),
        f"ev_lamb={config.get('lamb', None)}",
    ]
    wandb_tags = [tag for tag in wandb_tags if tag]
    run.tags += tuple(wandb_tags)
    return run


def get_dataloader(
        config: Dict[str, Any],
        device: Union[torch.device, str] = DEVICE,
        logger: Optional[logging.Logger] = None,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create dataloaders for training, validation, and testing.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing dataset parameters.
    device : torch.device or str, optional
        Device to load the dataset on. Default is ``DEVICE``.
    logger : logging.Logger or None, optional
        Logger instance.

    Returns
    -------
    dict of str -> torch.utils.data.DataLoader
        Train, validation, and test dataloaders.
    """
    data_name = config.get("data_name", "papyrus")
    activity_type = config.get("activity_type", "xc50")
    n_targets = config.get("n_targets", -1)
    descriptor_protein = config.get("descriptor_protein", None)
    descriptor_chemical = config.get("descriptor_chemical", None)
    median_scaling = config.get("median_scaling", False)
    split_type = config.get("split_type", "random")
    ext = config.get("ext", "pkl")
    task_type = config.get("task_type", "regression")

    datasets = build_datasets(
        data_name,
        n_targets,
        activity_type,
        split_type,
        descriptor_protein,
        descriptor_chemical,
        median_scaling,
        task_type,
        ext,
        logger,
        device=device,
    )

    batch_size = config.get("batch_size", 128)
    wt_resampler = config.get("wt_resampler", False)
    dataloaders = build_loader(
        datasets, batch_size, shuffle=False, wt_resampler=wt_resampler
    )

    return dataloaders


def post_training_save_model(
        model: torch.nn.Module,
        config: Dict[str, Any],
        model_type: str = "pnn",
        onnx: bool = True,
        tracker: str = "wandb",
        run: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        write_model: bool = True,
) -> str:
    """
    Save the trained model and configuration.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    config : dict
        Configuration dictionary.
    model_type : str, optional
        Type of the model. Default is ``"pnn"``.
    onnx : bool, optional
        Whether to save the model in ONNX format. Default is ``True``.
    tracker : str, optional
        Tracker to log model information. Default is ``"wandb"``.
    run : Any or None, optional
        Wandb run instance.
    logger : logging.Logger or None, optional
        Logger instance.
    write_model : bool, optional
        Whether to save the model file. Default is ``True``.

    Returns
    -------
    str
        Name of the saved model.
    """
    config["model_type"] = model_type
    model_name = get_model_name(config, run=run)
    data_specific_path = get_data_specific_path(config, logger=logger)
    config["data_specific_path"] = data_specific_path

    descriptor_protein = config.get("descriptor_protein", None)
    descriptor_chemical = config.get("descriptor_chemical", None)
    desc_prot_len, desc_chem_len = get_desc_len(descriptor_protein, descriptor_chemical)

    if write_model:
        save_model(
            config,
            model,
            model_name,
            data_specific_path,
            desc_prot_len,
            desc_chem_len,
            onnx=onnx,
            tracker=tracker,
        )

    return model_name


def get_tracker(
        config: Optional[Dict[str, Any]], tracker: str = "wandb"
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Initialize and retrieve a tracking instance.

    Parameters
    ----------
    config : dict or None
        Configuration dictionary.
    tracker : str, optional
        Tracker type ("wandb" or other). Default is ``"wandb"``.

    Returns
    -------
    (Any or None, dict or None)
        Tracking run instance and updated config.
    """
    if tracker.lower() == "wandb":
        if config is not None:
            run = wandb.init(
                config=config,
                dir=WANDB_DIR,
                mode=WANDB_MODE,
                project=config.get("wandb_project_name", "test-project"),
                reinit=True,
            )
            config = wandb.config
            run = assign_wandb_tags(run, config)
        else:
            run = wandb.init(dir=WANDB_DIR, mode=WANDB_MODE)
            config = wandb.config
    else:
        run = None

    return run, config


def train_model_e2e(
        config: Dict[str, Any],
        model: Type[torch.nn.Module],
        model_type: str = "pnn",
        model_kwargs: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        seed: int = 42,
        device: torch.device = DEVICE,
        tracker: str = "wandb",
        write_model: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any], np.ndarray, float]:
    """
    Train a model end-to-end: dataset prep, training, and evaluation.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing training parameters.
    model : type of torch.nn.Module
        Model class to instantiate and train.
    model_type : str, optional
        Type of the model. Default is ``"pnn"``.
    model_kwargs : dict or None, optional
        Additional keyword arguments for model initialization.
    logger : logging.Logger or None, optional
        Logger instance.
    seed : int, optional
        Random seed for reproducibility. Default is ``42``.
    device : torch.device, optional
        Device for training. Default is ``DEVICE``.
    tracker : str, optional
        Tracker for logging progress. Default is ``"wandb"``.
    write_model : bool, optional
        Whether to save the trained model. Default is ``True``.

    Returns
    -------
    (torch.nn.Module, dict, numpy.ndarray, float)
        Trained model, updated config, training statistics, and test loss.
    """
    if model_kwargs is None:
        model_kwargs = {}

    run, config = get_tracker(config, tracker=tracker)

    n_targets = config.get("n_targets", -1)
    descriptor_protein = config.get("descriptor_protein", None)
    descriptor_chemical = config.get("descriptor_chemical", None)
    split_type = config.get("split_type", "random")
    max_norm = config.get("max_norm", None)
    seed = config.get("seed", seed)

    set_seed(seed)
    assert split_type in [
        "random",
        "scaffold",
        "time",
        "scaffold_cluster",
    ], "Split type must be either random or scaffold or scaffold_cluster or time"

    mt = n_targets > 1
    config["MT"] = mt
    if mt and descriptor_protein:
        logger.warning(
            "For multitask learning, only chemical descriptors will be used. Setting descriptor_protein to None"
        )
        descriptor_protein = None
    if mt and split_type == "time":
        logger.warning(
            "For multitask learning, only random or scaffold split will be used. Setting split_type to random"
        )
        config["split_type"] = "random"

    config["prot_input_dim"], config["chem_input_dim"] = get_desc_len(
        descriptor_protein, descriptor_chemical, logger=logger
    )

    model_ = model(config=config, **model_kwargs, logger=logger).to(device)

    dataloaders = get_dataloader(config, device=device, logger=logger)
    best_model, test_loss, results_arr = run_model(
        config,
        model_,
        dataloaders,
        device=device,
        logger=logger,
        max_norm=max_norm,
        tracker=tracker,
    )
    config["model_name"] = post_training_save_model(
        best_model,
        config,
        model_type=model_type,
        tracker=tracker,
        run=run,
        logger=logger,
        write_model=write_model,
    )

    return best_model, config, results_arr, test_loss


def recalibrate_model(
        preds_val: Union[np.ndarray, torch.Tensor, List[float]],
        labels_val: Union[np.ndarray, torch.Tensor, List[float]],
        alea_vars_val: Union[np.ndarray, torch.Tensor, List[float]],
        preds_test: Union[np.ndarray, torch.Tensor, List[float]],
        labels_test: Union[np.ndarray, torch.Tensor, List[float]],
        alea_vars_test: Union[np.ndarray, torch.Tensor, List[float]],
        config: Dict[str, Any],
        epi_val: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
        epi_test: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
        uct_logger: Optional[Any] = None,
        figpath: Optional[str | Path | bytes | LiteralString] = None,
        nll: Optional[float] = None,
) -> Any:
    """
    Recalibrate uncertainty estimates using validation and test predictions.

    Parameters
    ----------
    preds_val : array-like
        Predictions on the validation set.
    labels_val : array-like
        True labels for the validation set.
    alea_vars_val : array-like
        Aleatoric uncertainties for validation predictions.
    preds_test : array-like
        Predictions on the test set.
    labels_test : array-like
        True labels for the test set.
    alea_vars_test : array-like
        Aleatoric uncertainties for test predictions.
    config : dict
        Configuration dictionary containing model and dataset settings.
    epi_val : array-like or None, optional
        Epistemic uncertainties for validation predictions.
    epi_test : array-like or None, optional
        Epistemic uncertainties for test predictions.
    uct_logger : Any or None, optional
        Logger for uncertainty quantification metrics.
    figpath : str or Path or None, optional
        Path to save calibration plots.
    nll : float or None, optional
        Negative log-likelihood value.

    Returns
    -------
    Any
        The recalibration model used for adjusting uncertainty estimates.
    """
    model_name = config.get("model_name", "ensemble")
    data_specific_path = config.get("data_specific_path", None)
    model_type = config.get("model_type", None)
    figures_path = figpath or (FIGS_DIR / data_specific_path / model_name)

    y_true_val, y_pred_val, y_err_val, y_alea_val, y_eps_val = process_preds(
        preds_val, labels_val, alea_vars_val, epi_vars=epi_val, model_type=model_type
    )
    y_true_test, y_pred_test, y_err_test, y_alea_test, y_eps_test = process_preds(
        preds_test,
        labels_test,
        alea_vars_test,
        epi_vars=epi_test,
        model_type=model_type,
    )

    iso_recal_model = recalibrate(
        y_true_val,
        y_pred_val,
        y_alea_val,
        y_err_val,
        y_true_test,
        y_pred_test,
        y_alea_test,
        y_err_test,
        n_subset=None,
        savefig=True,
        save_dir=figures_path,
        uct_logger=uct_logger,
    )
    return iso_recal_model
