import logging
from pathlib import Path
from typing import Tuple, Dict, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import uncertainty_toolbox as uct
from uncertainty_toolbox import viz as uct_viz
from uncertainty_toolbox.metrics import get_all_metrics
from tdc import Evaluator
import wandb
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score

from uqdd import FIGS_DIR, TODAY, DATA_DIR
from uqdd.data.utils_data import export_pickle, export_df
from uqdd.utils import create_logger
from uqdd.utils_chem import smi_to_pil_image

import math

import scipy.stats as ss

from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import norm
from scipy.stats import bootstrap
from scipy.integrate import quad
from bisect import bisect_left

string_types = (type(b""), type(""))
sns.set(style="white")


def calc_nanaware_metrics(tensor, nan_mask, all_tasks_agg=False):
    """
    Aggregate a tensor by excluding NaN values based on a nan_mask.

    Calculates the mean of the non-NaN values along the specified dimension.
    Optionally, it can aggregate the mean across all tasks.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be aggregated.
    nan_mask : torch.Tensor
        A boolean mask indicating the NaN values in the tensor.
    all_tasks_agg : bool or str, optional
        Determines whether to aggregate across all tasks. If False (default),
        returns the mean for each task. If 'mean', returns the mean of all tasks.
        If 'sum', returns the sum of all tasks.

    Returns
    -------
    torch.Tensor
        The aggregated tensor based on the specified aggregation method.

    Notes
    -----
    - The nan_mask should have the same shape as tensor_for_agg.
    - The nan_mask should be a boolean tensor with True indicating NaN values.

    Examples
    --------
    >> import torch
    >> tensor_for_agg = torch.tensor([[1, 2, 3], [4, float('nan'), 6]])
    >> nan_mask = torch.isnan(tensor_for_agg)
    >> aggregated_tensor = calc_nanaware_metrics(tensor_for_agg, nan_mask, all_tasks_agg=True)
    >> print(aggregated_tensor)
    tensor(3.3333)

    The above example demonstrates the usage of the `agg_notnan` function.
    The input tensor contains NaN values, and the nan_mask is used to identify those NaN values.
    By specifying `all_tasks_agg=True`, the function calculates the mean of the non-NaN values and then
    returns the mean of all tasks. In this case, the output is `tensor(3.3333)`.
    """
    # Now we only include the non-Nan targets in the mean calc.
    # tensor_means = torch.sum(tensor, dim=0) / torch.sum(~nan_mask, dim=0)

    valid_values = torch.where(
        ~nan_mask, tensor, torch.tensor(0.0, device=tensor.device)
    )
    sum_values = torch.sum(valid_values, dim=0)
    valid_counts = torch.sum(~nan_mask, dim=0)

    tensor_means = sum_values / valid_counts.clamp(min=1)  # Avoid division by zero

    # If we want to aggregate across all tasks, we do so here.
    # if not all_tasks_agg:
    #     return tensor_means
    if all_tasks_agg == "mean":
        return torch.nanmean(tensor_means)
    elif all_tasks_agg == "sum":
        return torch.nansum(tensor_means)
    return tensor_means


def calc_regr_metrics(targets, outputs, metrics_per_task=False):
    """
    Calculates regression metrics between targets and outputs, including RMSE, R^2, and EVS.

    Parameters
    ----------
    targets : torch.Tensor
        The true target values.
    outputs : torch.Tensor
        The predicted output values by the model.
    metrics_per_task : bool, optional
        Whether to calculate the metrics per task in a multi-task learning scenario.
    Returns
    -------
    tuple of np.ndarray
        The calculated metrics (RMSE, R^2, EVS) for each task.
    """
    # Handle extra dimensions (ensembles) by averaging ensemble predictions
    if outputs.dim() > targets.dim():
        outputs = outputs.nanmean(dim=-1)

    targets = targets.detach().cpu()
    outputs = outputs.detach().cpu()

    if metrics_per_task:
        # Calculate metrics per task
        rmse = np.full(targets.shape[1], np.nan)  # Initialize with NaNs
        r2 = np.full(targets.shape[1], np.nan)    # Initialize with NaNs
        evs = np.full(targets.shape[1], np.nan)   # Initialize with NaNs

        for i in range(targets.shape[1]):
            task_t = targets[:, i]
            task_o = outputs[:, i]

            # Filter out NaN values for valid comparison
            valid_mask = ~torch.isnan(task_t)
            if valid_mask.any():  # Check if there are any valid data points
                task_t = task_t[valid_mask].numpy()  # Valid target values
                task_o = task_o[valid_mask].numpy()  # Valid output predictions

                # Compute metrics
                rmse[i] = mean_squared_error(task_t, task_o, squared=False)
                r2[i] = r2_score(task_t, task_o)
                evs[i] = explained_variance_score(task_t, task_o)
        #     # nan_mask = ~torch.isnan(task_t)
        #     task_t, task_o = task_t[valid_mask].numpy(), task_o[valid_mask].numpy()
        #     rmse.append(mean_squared_error(task_t, task_o, squared=False))
        #     r2.append(r2_score(task_t, task_o))
        #     evs.append(explained_variance_score(task_t, task_o))
        # rmse, r2, evs = np.array(rmse), np.array(r2), np.array(evs)
    else:
        # Calculate metrics for all tasks
        nan_mask = ~torch.isnan(targets)
        targets, outputs = targets[nan_mask].numpy().flatten(), outputs[nan_mask].numpy().flatten()
        rmse = mean_squared_error(targets, outputs, squared=False)
        r2 = r2_score(targets, outputs)
        evs = explained_variance_score(targets, outputs)

    return rmse, r2, evs
    # Detect multitask learning scenario
    # is_multitask = targets.shape[1] > 1

    # targets = targets.squeeze().flatten()
    # outputs = outputs.squeeze().flatten()
    # nan_mask = torch.isnan(targets)
    # targets = targets[~nan_mask]
    # outputs = outputs[~nan_mask]
    #
    # targets = targets.detach().cpu().numpy()
    # outputs = outputs.detach().cpu().numpy()
    #
    # # # Adjust dimensions if necessary (for MTL to STL comparison)
    # # if targets.dim() < outputs.dim():
    # #     targets = targets.unsqueeze(-1)
    # # targets.requires_grad_(False)
    # # outputs.requires_grad_(False)
    # # targets = targets.detach()
    # # outputs = outputs.detach()
    # #
    # # targets = targets.cpu().numpy()
    # # outputs = outputs.cpu().numpy()
    # # targets = targets.detach().numpy()
    # # outputs = outputs.detach().numpy()
    #
    # # no reduction here because we want to calc per task metrics
    # rmse = mean_squared_error(
    #     targets,
    #     outputs,
    #     squared=False,
    #     # , multioutput="uniform_average" # it doesn't matter as we take mean over prediction
    # )
    # r2 = r2_score(targets, outputs)  # , multioutput="uniform_average"
    # evs = explained_variance_score(targets, outputs)  # , multioutput="uniform_average"


# def calc_loss_notnan(outputs, targets, nan_mask, loss_fn):
#     """
#     Calculates the loss for non-NaN values between outputs and targets using a given loss function.
#
#     Parameters
#     ----------
#     outputs : torch.Tensor
#         Predicted outputs from the model.
#     targets : torch.Tensor
#         True target values.
#     nan_mask : torch.Tensor
#         A boolean mask indicating NaN values in the targets.
#     loss_fn : function
#         A loss function compatible with torch.Tensors and supports 'none' reduction.
#
#     Returns
#     -------
#     torch.Tensor
#         The aggregated loss value excluding NaNs.
#     """
#     valid_targets = torch.where(
#         ~nan_mask, targets, torch.tensor(0.0, device=targets.device)
#     )
#     valid_outputs = torch.where(
#         ~nan_mask, outputs, torch.tensor(0.0, device=outputs.device)
#     )
#
#     loss_per_task = loss_fn(valid_outputs, valid_targets)
#     loss = calc_nanaware_metrics(loss_per_task, nan_mask, all_tasks_agg="sum")
#
#     return loss

# targets[nan_mask], outputs[nan_mask] = 0.0, 0.0
#
# loss_per_task = loss_fn(outputs, targets)
#
# # Now we only include the non-Nan targets in the mean calc.
# loss = calc_nanaware_metrics(
#     tensor=loss_per_task, nan_mask=nan_mask, all_tasks_agg="sum"
# )
# # task_losses = torch.sum(loss_per_task, dim=1) / torch.sum(~nan_mask, dim=1)
# # loss = torch.sum(task_losses)
# return loss


def process_preds(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    alea_vars: torch.Tensor = None,
    epi_vars: torch.Tensor = None,
    task_idx: Union[int, None] = None
):
    """
    Process predictions to extract mean, standard deviation, and align with targets,
    and compute the absolute error between predictions and targets.

    Parameters
    ----------
    predictions : torch.Tensor
        The model predictions with dimensions [samples, tasks, ensemble members].
    targets : torch.Tensor
        The true target values.
    alea_vars :  torch.Tensor, optional
        Aleatoric part of uncertainty
    epi_vars : torch.Tensor, optional
        Epistemic part of uncertainty - if calculated by probabilistic model (Evidential model)
    task_idx : int, optional
        Index of the specific task to process in a multi-task learning setting.

    Returns
    -------
    tuple
        A tuple containing arrays for predictions mean, standard deviation, targets,
        and absolute error, filtered to exclude NaN values.
    """

    if alea_vars is None:
        vars_ = torch.zeros_like(targets)
        # alea_vars_mean, alea_vars_vars = calc_aleatoric_mean_var_notnan(alea_vars, targets)
    else:
        if epi_vars is None:
            vars_ = alea_vars.mean(dim=-1) #.squeeze()
        else:
            vars_ = alea_vars

    if epi_vars is None:
        epi_vars = predictions.var(dim=-1) #.squeeze()  # (dim=2)
        predictions = predictions.mean(dim=-1) #.squeeze()
    # Get the predictions mean and std
    y_pred = predictions # predictions.mean(dim=-1).squeeze()  # (dim=2)
    y_std = epi_vars.sqrt()
    # y_std = np.minimum(y_std, 1e3) # clip the unc for vis
    y_true = targets #.squeeze()
    # if vars_ is not None:
    #     vars_ = vars_.mean(dim=-1).squeeze()
    # else:  # Empty Tensor
    #     vars_ = torch.zeros_like(targets)
    if task_idx is not None:
        # For MTL, select predictions for the specific task
        y_pred, y_std, y_true, vars_ = (
            y_pred[:, task_idx],
            y_std[:, task_idx],
            y_true[:, task_idx],
            vars_[:, task_idx]
        )

    nan_mask = ~torch.isnan(y_true)
    y_pred, y_std, y_true, vars_ = (
        y_pred[nan_mask],
        y_std[nan_mask],
        y_true[nan_mask],
        vars_[nan_mask],
    )
    # Calculate the error
    y_err = (y_pred - y_true).abs()  # IT IS SOOO HIGH WHY? - Not really only during testing the script was high
    # Convert to numpy arrays
    y_pred, y_std, y_true, y_err, vars_ = map(
        lambda x: x.cpu().numpy(), (y_pred, y_std, y_true, y_err, vars_)
    )

    return y_true, y_pred, y_std, y_err, vars_


def get_preds_export_path(data_specific_path, model_name):
    path = DATA_DIR / "predictions" / Path(data_specific_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path / f"{model_name}_preds.csv"


def create_df_preds(
    y_true,
    y_pred,
    y_std,
    y_err,
    y_alea=None,
    export: bool = True,
    data_specific_path: str = None,
    model_name: str = None,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Create a DataFrame from prediction data and optionally export it.

    Parameters
    ----------
    logger
    y_true : array-like
        The true target values.
    y_pred : array-like
        The predicted values.
    y_std : array-like
        The standard deviation of the predictions.
    y_err : array-like
        The prediction errors.
    y_alea : array-like or None
        Aleatoric uncertainty part.
    export : bool, optional
        Whether to export the DataFrame as a CSV file.
    data_specific_path : str, optional
        The specific data directory path for exporting.
    model_name : str, optional
        The name of the model for the export file name.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the prediction data.
    """
    df = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "y_std": y_std, "y_err": y_err, "y_alea": y_alea}
    )

    if export and data_specific_path and model_name:
        export_path = get_preds_export_path(data_specific_path, model_name)
        export_df(df, export_path)
        logger.debug(f"Exported predictions to {export_path}")

    return df


def plot_true_vs_preds(
    y_pred,
    y_true,
    n_subset=None,
    ax=None,
    **kwargs,
):
    if n_subset is not None and n_subset < len(y_true):
        # Randomly select indices if subset_size is specified and less than the total number of points
        # random generator
        rng = np.random.default_rng(42)
        indices = rng.choice(len(y_true), size=n_subset, replace=False)

        y_true_subset = y_true[indices]
        y_preds_subset = y_pred[indices]
    else:
        # Use all points if no subset_size is specified or if subset_size is larger than available points
        y_true_subset = y_true
        y_preds_subset = y_pred

    # Sort the subset based on y_true for better visualization
    sorted_indices = np.argsort(y_true_subset)
    sorted_y_true = y_true_subset[sorted_indices]
    sorted_y_preds = y_preds_subset[sorted_indices]

    # Plot the graph
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(sorted_y_true, sorted_y_preds, "o", label="Predictions")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    # ax.set_title(f"True vs. Predicted Values - {task_name or 'PCM'}")

    # Calculate the best-fitting line
    best_fit_coeffs = np.polyfit(sorted_y_true, sorted_y_preds, deg=1)
    best_fit_line = np.poly1d(best_fit_coeffs)
    ax.plot(
        sorted_y_true, best_fit_line(sorted_y_true), color="red", label="Best Fit Line"
    )

    # Calculate and display the RMSE for the subset
    rmse = np.sqrt(np.mean((sorted_y_preds - best_fit_line(sorted_y_true)) ** 2))
    ax.text(
        0.95, 0.05, f"RMSE: {rmse:.2f}", transform=ax.transAxes, ha="right", va="bottom"
    )

    # Show the legend and display/save the graph
    ax.legend(loc="upper left")
    #
    # if save_path:
    #     plt.savefig(save_path)
    #     print(f"Saved true_vs_preds_plot to {save_path}")
    # else:
    #     plt.show()
    #
    # return fig

    # Calculate the distances of each point to the best-fitted line
    # distances = np.abs(best_fit_line(sorted_y_true) - sorted_y_preds)
    # normalized_distances = distances / np.max(distances)
    # ax.plot(sorted_y_true, sorted_y_preds, color="gray", alpha=0.2)
    # for i in range(len(sorted_y_true)):
    #     ax.plot([sorted_y_true[i], sorted_y_true[i]], [sorted_y_preds[i], best_fit_line(sorted_y_true[i])],
    #              color='gray', alpha=normalized_distances[i])
    # Plot the grey lines between dots and best fit line
    # for i in range(len(sorted_y_true)):
    #     ax.plot(
    #         [sorted_y_true[i], sorted_y_true[i]],
    #         [sorted_y_preds[i], best_fit_line(sorted_y_true[i])],
    #         color="gray",
    #         alpha=0.2,
    #     )


def plot_pred_intervals(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: int = 100,
    ylims: Tuple[float, float] | None = (-3.0, 3.0),
    num_stds_confidence_bound: float = 2.0,  # 1.96 for 95% CI for normal distribution
    ax: plt.Axes = None,
    ordered=False,
):
    if ax is None:
        fig, ax = plt.subplots()
    if ordered:
        uct_viz.plot_intervals_ordered(
            y_pred=y_pred,
            y_std=y_std,
            y_true=y_true,
            n_subset=n_subset,
            ylims=ylims,
            num_stds_confidence_bound=num_stds_confidence_bound,
            ax=ax,
        )
    else:
        uct_viz.plot_intervals(
            y_pred=y_pred,
            y_std=y_std,
            y_true=y_true,
            n_subset=n_subset,
            ylims=ylims,
            num_stds_confidence_bound=num_stds_confidence_bound,
            ax=ax,
        )

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    ax.text(
        0.05,
        0.95,
        f"RMSE: {rmse:.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )


def plot_sharpness(
    y_std: np.ndarray, n_subset: int = 100, ax: plt.Axes = None, **kwargs
):
    if ax is None:
        fig, ax = plt.subplots()
    uct_viz.plot_sharpness(y_std=y_std, n_subset=n_subset, ax=ax)


def make_uct_plots(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    task_name: str = None,
    n_subset: int = 100,
    ylims: Tuple[float, float] | None = (-3.0, 3.0),
    num_stds_confidence_bound: float = 2.0,  # 1.96 for 95% CI for normal distribution
    plot_save_str: str = "uct_plot",
    savefig: bool = True,
    save_dir: Path = "path/to/figures",
) -> Dict[str, plt.Figure]:
    if not all(isinstance(arr, np.ndarray) for arr in [y_pred, y_std, y_true]):
        raise ValueError("y_preds, y_std, and y_true must be numpy arrays.")

    task_name = task_name if task_name is not None else "PCM"
    plots = {}
    plot_functions = [
        (
            "prediction_intervals",
            plot_pred_intervals,
            {
                "ylims": ylims,
                "num_stds_confidence_bound": num_stds_confidence_bound,
                "ordered": False,
            },
        ),
        (
            "ordered_prediction_intervals",
            plot_pred_intervals,
            {
                "ylims": ylims,
                "num_stds_confidence_bound": num_stds_confidence_bound,
                "ordered": True,
            },
        ),
        (
            "calibration",
            uct_viz.plot_calibration,
            {},
        ),
        (
            "adversarial_group_calibration",
            uct_viz.plot_adversarial_group_calibration,
            {},
        ),
        ("sharpness", plot_sharpness, {}),
        ("residuals_vs_stds", uct_viz.plot_residuals_vs_stds, {}),
        ("true_vs_predictions", plot_true_vs_preds, {}),
    ]

    for plot_name, plot_func, kwargs in plot_functions:
        fig, ax = plt.subplots(figsize=(6, 4))
        # if plot_name == "sharpness":
        #     plot_func(y_std=y_std, n_subset=n_subset, ax=ax)
        # else:
        plot_func(
            y_pred=y_pred,
            y_std=y_std,
            y_true=y_true,
            n_subset=n_subset,
            ax=ax,
            **kwargs,
        )
        ax.set_title(f"{plot_name.replace('_', ' ').title()} - {task_name}")
        plt.tight_layout()
        plots[plot_name] = fig

        if savefig:
            fig_save_path = save_dir / f"{plot_save_str}_{plot_name}"
            uct.viz.save_figure(
                str(fig_save_path), ext_list=["png", "svg"], white_background=True
            )
            # plt.savefig(fig_save_path, format="png", bbox_inches="tight")
    #

    return plots


def calculate_uct_metrics(
    y_pred,
    y_std,
    y_true,
    Nbins=100,
    task_name=None,
    figpath=FIGS_DIR,
):
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
    figpath : Path or str
        Path to save the figures. Default is FIGS_DIR.

    Returns
    -------
    metrics : dict
        Dictionary containing calculated metrics.
    plots : dict
        Dictionary containing the generated plots.
    """
    metrics = get_all_metrics(
        y_pred=y_pred,
        y_std=y_std,
        y_true=y_true,
        num_bins=Nbins,
        resolution=99,
        scaled=True,
        verbose=True,
    )
    # figures_path = FIGS_DIR / data_specific_path / model_name
    # figures_path.mkdir(parents=True, exist_ok=True)
    metrics_filepath = Path(figpath) / f"{task_name}_metrics.pkl"
    export_pickle(metrics, metrics_filepath)

    plots = make_uct_plots(
        y_pred,
        y_std,
        y_true,
        task_name=task_name,
        n_subset=min(200, len(y_true)),
        ylims=None,
        num_stds_confidence_bound=2.0,
        plot_save_str=f"{task_name}_uct",
        savefig=True,
        save_dir=Path(figpath),
    )

    # img = wandb.Image(fig)
    # plots = {"UCT"}
    return metrics, plots


def calculate_tdc_classification_metrics(
    y_pred,
    y_true,
    task_name=None,
    data_specific_path="papyrus/xc50/all/",
):
    metrics = [
        "PR-AUC",
        "range_logAUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "PR@K",
        "RP@K",
    ]

    # thresholds = {"Accuracy":0.5, "Precision":0.5, "Recall":0.5, "F1":0.5, "PR@K":0.9, "RP@K":0.9}
    # thresholds = {"PR@K":0.9, "RP@K":0.9}
    results = {}
    for m in metrics:
        threshold = 0.9 if m in ["PR@K", "RP@K"] else 0.5
        evaluator = Evaluator(name=m)
        results[m] = evaluator(y_true, y_pred, threshold=threshold)

    return results
    # if m in ["PR@K", "RP@K"]:
    #     score = evaluator(y_true, y_pred, threshold=0.9)
    # else:
    #     score = evaluator(y_true, y_pred)
    # results[m] = score
    # print(f"{m} score: {score:.4f}")


def make_uq_plots(
    ordered_df,
    gaus_pred,
    errors_observed,
    mis_cal,
    Nbins=20,
    include_bootstrap=True,
    task_name="PCM",
    figpath=FIGS_DIR,
    logger=logging.Logger("uqtools"),
):
    # generate error-based calibration plot
    fig, _, _, _ = get_slope_metric(
        ordered_df.uq,
        ordered_df.errors,
        Nbins=Nbins,
        include_bootstrap=include_bootstrap,
        logger=logger,
    )
    if figpath is not None:
        fig.savefig(Path(figpath) / f"{task_name}_rmv_vs_rmse.png")
        fig.savefig(Path(figpath) / f"{task_name}_rmv_vs_rmse.svg")


    # Generate Z-score plot and calibration curve
    fig2, _ = plot_Z_scores(ordered_df.errors, ordered_df.uq)
    if figpath is not None:
        fig2.savefig(Path(figpath) / f"{task_name}_Z_scores.png")
        fig2.savefig(Path(figpath) / f"{task_name}_Z_scores.svg")

    fig3 = plot_calibration_curve(gaus_pred, errors_observed, mis_cal)
    if figpath is not None:
        fig3.savefig(Path(figpath) / f"{task_name}_uq_calibration_curve.png")
        fig3.savefig(Path(figpath) / f"{task_name}_uq_calibration_curve.svg")
    plots = {"rmv_vs_rmse": fig, "Z_scores": fig2, "calibration_curve": fig3}
    return plots


def calculate_uqtools_metrics(
    uncertainties,
    errors,
    Nbins=100,
    include_bootstrap=True,
    task_name="PCM",
    figpath=FIGS_DIR,
    logger=logging.Logger("uqtools"),
):
    # Order uncertainties and errors according to uncertainties
    ordered_df = order_sig_and_errors(uncertainties, errors)
    # calculate rho_rank and rho_rank_sim # TODO spearmanr is already implemented in scipy
    # rho_rank, _ = spearman_rank_corr(np.abs(errors), uncertainties)
    rho_rank, _ = spearmanr(np.abs(errors), uncertainties)
    logger.info(f"rho_rank = {rho_rank:.2f}")
    exp_rhos_temp = []
    for i in range(1000):
        exp_rho, _ = expected_rho(uncertainties)
        exp_rhos_temp.append(exp_rho)
    rho_rank_sim = np.mean(exp_rhos_temp)
    rho_rank_sim_std = np.std(exp_rhos_temp)
    logger.info(f"rho_rank_sim = {rho_rank_sim:.2f} +/- {rho_rank_sim_std:.2f}")

    # Calculate the miscalibration area
    gaus_pred, errors_observed = calibration_curve(ordered_df.abs_z)
    mis_cal = calibration_area(errors_observed, gaus_pred)
    logger.info(f"miscalibration area = {mis_cal:.2f}")

    # Calculate NLL and simulated NLL
    _NLL = NLL(uncertainties, errors)
    logger.info(f"NLL = {_NLL:.2f}")
    exp_NLL = []
    rng = np.random.default_rng()
    for i in range(1000):
        sim_errors = np.abs(rng.normal(0, uncertainties))
        # sim_errors = []
        # for sigma in uncertainties:
        #     sim_error = rng.normal(0, sigma)
        #     # sim_error = np.random.normal(0, sigma)
        #     sim_errors.append(sim_error)
        NLL_sim = NLL(uncertainties, sim_errors)
        exp_NLL.append(NLL_sim)
    NLL_sim = np.mean(exp_NLL)
    NLL_sim_std = np.std(exp_NLL)
    logger.info(f"NLL_sim = {NLL_sim:.2f} +/- {NLL_sim_std:.2f}")

    plots = make_uq_plots(
        ordered_df,
        gaus_pred,
        errors_observed,
        mis_cal,
        Nbins=Nbins,
        include_bootstrap=include_bootstrap,
        task_name=task_name,
        figpath=figpath,
        logger=logger,
    )

    # Z-metrics
    Z = errors / uncertainties
    Z_var = np.var(Z)
    interval_var = bootstrap((Z,), np.var)
    logger.info(f"var(Z) = {Z_var:.2f} CI = {interval_var.confidence_interval}")
    Z_mean = np.mean(Z)
    interval_mean = bootstrap((Z,), np.mean)
    logger.info(f"mean(Z) = {Z_mean:.2f} CI = {interval_mean.confidence_interval}")

    metrics = {
        "rho_rank": rho_rank,
        "rho_rank_sim": rho_rank_sim,
        "rho_rank_sim_std": rho_rank_sim_std,
        "uq_mis_cal": mis_cal,
        "uq_NLL": _NLL,
        "uq_NLL_sim": NLL_sim,
        "uq_NLL_sim_std": NLL_sim_std,
        "Z_var": Z_var,
        "Z_var_CI_low": interval_var.confidence_interval.low,
        "Z_var_CI_high": interval_var.confidence_interval.high,
        "Z_mean": Z_mean,
        "Z_mean_CI_low": interval_mean.confidence_interval.low,
        "Z_mean_CI_high": interval_mean.confidence_interval.high,
    }

    return metrics, plots

    # return (
    #     fig,
    #     fig2,
    #     fig3,
    #     rho_rank,
    #     rho_rank_sim,
    #     rho_rank_sim_std,
    #     mis_cal,
    #     _NLL,
    #     NLL_sim,
    #     NLL_sim_std,
    #     Z_var,
    #     interval_var.confidence_interval,
    #     Z_mean,
    #     interval_mean.confidence_interval,
    # )

    # rmvs, rmses, ci_low, ci_high = get_rmvs_and_rmses(
    #     uncertainties, errors, Nbins=Nbins, include_bootstrap=include_bootstrap
    # )
    # fig_z_score, ax = plot_Z_scores(errors, uncertainties)
    #
    # fig, slope, r_sq, intercept = get_slope_metric(
    #     ordered_df.uq,
    #     ordered_df.errors,
    #     Nbins=Nbins,
    #     include_bootstrap=include_bootstrap,
    # )
    #
    # gaus_pred, errors_observed = calibration_curve(ordered_df.abs_z)
    # mis_cal = calibration_area(errors_observed, gaus_pred)
    # print("miscalibration area = ", mis_cal)
    # fig = plot_calibration_curve(gaus_pred, errors_observed, mis_cal)
    #
    # ci_low, ci_high = get_bootstrap_intervals(ordered_df.errors, Nbins=Nbins)
    # rho, sim_errors = expected_rho(ordered_df.uq)

    # raise NotImplementedError


def order_sig_and_errors(sigmas, errors):
    ordered_df = pd.DataFrame()
    ordered_df["uq"] = sigmas
    ordered_df["errors"] = errors
    ordered_df["abs_z"] = np.abs(ordered_df.errors) / ordered_df.uq
    ordered_df = ordered_df.sort_values(by="uq")
    return ordered_df


def rmse(x, axis=None):
    return np.sqrt((x**2).mean())


### SOURCE: UQtools.py https://github.com/jensengroup/UQ_validation_methods/blob/main/UQtools.py ###


def get_bootstrap_intervals(errors_ordered, Nbins=10):
    """
    calculate the confidence intervals at a given p-level.
    """
    ci_low = []
    ci_high = []
    N_total = len(errors_ordered)
    N_entries = math.ceil(N_total / Nbins)

    for i in range(0, N_total, N_entries):
        data = errors_ordered[i : i + N_entries]
        res = bootstrap((data,), rmse, vectorized=False)
        ci_low.append(res.confidence_interval[0])
        ci_high.append(res.confidence_interval[1])
    return ci_low, ci_high


def expected_rho(uncertainties):
    """
    for each uncertainty we draw a random Gaussian error to simulate the expected errors
    the spearman rank coeff. is then calculated between uncertainties and errors.
    """
    # sim_errors = []
    rng = np.random.default_rng()
    sim_errors = np.abs(rng.normal(0, uncertainties))
    # for sigma in uncertainties:
    #     # random normal generator
    #     error = np.abs(rng.normal(0, sigma))
    #     error = np.abs(np.random.normal(0, sigma))
    #     sim_errors.append(error)

    # rho, _ = spearman_rank_corr(uncertainties, sim_errors)
    rho, _ = spearmanr(uncertainties, sim_errors)
    return rho, sim_errors


# def spearman_rank_corr(v1, v2):
#     v1_ranked = ss.rankdata(v1)
#     v2_ranked = ss.rankdata(v2)
#     return pearsonr(v1_ranked, v2_ranked)


def NLL(uncertainties, errors):
    NLL = 0
    for uncertainty, error in zip(uncertainties, errors):
        temp = math.log(2 * np.pi * uncertainty**2) + (error) ** 2 / uncertainty**2
        NLL += temp

    NLL = NLL / (2 * len(uncertainties))
    return NLL


def calibration_curve(errors_sigma):
    N_errors = len(errors_sigma)
    gaus_pred = []
    errors_observed = []
    for i in np.arange(-10, 0 + 0.01, 0.01):
        gaus_int = 2 * norm(loc=0, scale=1).cdf(i)
        gaus_pred.append(gaus_int)
        observed_errors = (errors_sigma > abs(i)).sum()
        errors_frac = observed_errors / N_errors
        errors_observed.append(errors_frac)

    return gaus_pred, errors_observed


def plot_calibration_curve(gaus_pred, errors_observed, mis_cal):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        gaus_pred,
        gaus_pred,
        errors_observed,
        color="purple",
        alpha=0.4,
        label="miscalibration area = {:0.3f}".format(mis_cal),
    )
    ax.plot(gaus_pred, errors_observed, color="purple", alpha=1)
    ax.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), linestyle="dashed", color="k")
    ax.set_xlabel("expected fraction of errors", fontsize=14)
    ax.set_ylabel("observed fraction of errors", fontsize=14)
    ax.legend(fontsize=14, loc="lower right")
    return fig


def plot_Z_scores(errors, uncertainties):
    Z_scores = errors / uncertainties
    N_bins = 29
    xmin, xmax = -7, 7
    y, bin_edges = np.histogram(Z_scores, bins=N_bins, range=(xmin, xmax))
    bin_width = bin_edges[1] - bin_edges[0]
    x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    sy = np.sqrt(y)
    target_values = np.array(
        [len(errors) * bin_width * norm.pdf(x_value) for x_value in x]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(Z_scores, bins=N_bins, range=(xmin, xmax), color="purple", alpha=0.3)
    ax.errorbar(x, y, sy, fmt=".", color="k")
    ax.plot(
        np.arange(-7, 7, 0.1),
        len(errors) * bin_width * norm.pdf(np.arange(-7, 7, 0.1), 0, 1),
        color="k",
    )
    ax.set_xlabel("error (Z)", fontsize=16)
    ax.set_ylabel("count", fontsize=16)
    ax.set_xlim([-7, 7])
    return fig, ax


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0], myList[1], 0, 1
    if pos == len(myList):
        return myList[-2], myList[-1], -2, -1
    before = myList[pos - 1]
    after = myList[pos]
    if myNumber < before or myNumber > after:
        print("problem")
    else:
        return before, after, pos - 1, pos


def f_linear_segment(x, point_list=None, x_list=None):

    x1, x2, x1_idx, x2_idx = take_closest(x_list, x)
    f = point_list[x1_idx] + (x - x1) / (x2 - x1) * (
        point_list[x2_idx] - point_list[x1_idx]
    )

    return f


def area_function(x, observed_list, predicted_list):
    h = np.abs((f_linear_segment(x, observed_list, predicted_list) - x))
    return h


def calibration_area(observed, predicted):
    area = 0
    x = min(predicted)
    while x < max(predicted):
        temp, _ = quad(area_function, x, x + 0.001, args=(observed, predicted))
        area += temp
        x += 0.001
    return area


def chi_squared(x_values, x_sigmas, target_values):
    mask = x_values > 0
    chi_value = ((x_values[mask] - target_values[mask]) / x_sigmas[mask]) ** 2
    chi_value = np.sum(chi_value)

    N_free_cs = len(x_values[mask])
    print(N_free_cs)
    chi_prob = ss.chi2.sf(chi_value, N_free_cs)
    return chi_value, chi_prob


def get_slope_metric(
    uq_ordered,
    errors_ordered,
    Nbins=10,
    include_bootstrap=True,
    logger=logging.Logger("uqtools"),
):
    """
    Calculates the error-based calibration metrices

    uq_ordered: list of uncertainties in increasing order
    error_ordered: list of observed errors corresponding to the uncertainties in uq_ordered
    NBins: integer deciding how many bins to use for the error-based calibration metric
    include_bootstrap: boolean deciding wiether to include 95% confidence intervals on RMSE values from bootstrapping
    """
    rmvs, rmses, ci_low, ci_high = get_rmvs_and_rmses(
        uq_ordered, errors_ordered, Nbins=Nbins, include_bootstrap=include_bootstrap
    )

    x = np.array(rmvs).reshape((-1, 1))
    y = np.array(rmses)
    model = LinearRegression().fit(x, y)

    # Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
    r_sq = model.score(x, y)
    logger.info("R squared:", r_sq)

    # Print the Intercept:
    intercept = model.intercept_
    logger.info("intercept:", intercept)

    # Print the Slope:
    slope = model.coef_[0]
    logger.info("slope:", slope)

    # Predict a Response and print it:
    y_pred = model.predict(x)

    fig, ax = plt.subplots(figsize=(8, 5))
    assymetric_errors = [
        np.array(rmses) - np.array(ci_low),
        np.array(ci_high) - np.array(rmses),
    ]
    ax.errorbar(x, y, yerr=assymetric_errors, fmt="o", linewidth=2)
    ax.plot(
        np.arange(rmvs[0], rmvs[-1], 0.0001),
        np.arange(rmvs[0], rmvs[-1], 0.0001),
        linestyle="dashed",
        color="k",
    )
    ax.plot(
        rmvs,
        y_pred,
        linestyle="dashed",
        color="red",
        label=r"$R^2$ = "
        + "{:0.2f}".format(r_sq)
        + ", slope = {:0.2f}".format(slope)
        + ", intercept = {:0.2f}".format(intercept),
    )

    ax.set_xlabel("RMV", fontsize=14)
    ax.set_ylabel("RMSE", fontsize=14)
    ax.legend(fontsize=14)
    return fig, slope, r_sq, intercept


def get_rmvs_and_rmses(uq_ordered, errors_ordered, Nbins=10, include_bootstrap=True):
    """
    uq orderes should be the list of uncertainties in increasing order and errors should be the corresponding errors
    Nbins determine how many bins the data should be divided into
    """

    N_total = len(uq_ordered)
    N_entries = math.ceil(N_total / Nbins)
    # print(N_entries)
    rmvs = [
        np.sqrt((uq_ordered[i : i + N_entries] ** 2).mean())
        for i in range(0, N_total, N_entries)
    ]
    # print(rmvs)
    rmses = [
        np.sqrt((errors_ordered[i : i + N_entries] ** 2).mean())
        for i in range(0, N_total, N_entries)
    ]
    if include_bootstrap:
        ci_low, ci_high = get_bootstrap_intervals(errors_ordered, Nbins=Nbins)
    else:
        ci_low, ci_high = None, None
    return rmvs, rmses, ci_low, ci_high


class MetricsTable:
    def __init__(
        self,
        config=None,
        model_type=None,
        add_plots_to_table=False,
        logger=None,
    ):
        """
        Initialize the metrics table.

        Returns:
        --------
        None
        """
        self.config = config
        self.logger = logger or create_logger("MetricsTable")
        self.activity = config.get("activity_type", "xc50")
        self.split = config.get("split_type", "time")
        self.model_type = model_type
        self.desc_prot = config.get("descriptor_protein", None)
        self.desc_chem = config.get("descriptor_chemical", None)
        self.mt = config.get("MT", False)
        self.task_type = config.get("task_type", "regression")
        self.data_specific_path = config.get("data_specific_path", None)
        self.model_name = config.get("model_name", "ensemble")
        self.aleatoric = config.get("aleatoric", False)
        self.add_plots_to_table = add_plots_to_table
        cols = []
        if self.model_type:
            cols += ["Model type"]
        if self.mt:
            cols += ["Task"]
        cols += ["Activity", "Split", "desc_prot", "desc_chem"]
        if self.task_type == "regression":
            cols.extend(
                [
                    "RMSE",
                    "R2",
                    "MAE",
                    "MDAE",
                    "MARPD",
                    "PCC",  # Pearson correlation coefficient
                    "RMS Calibration",
                    "MA Calibration",
                    "Miscalibration Area",
                    "Sharpness",
                    "NLL",
                    "CRPS",
                    "Check",
                    "Interval",
                ]
            )
            # UQtools
            cols.extend(
                [
                    "rho_rank",
                    "rho_rank_sim",
                    "rho_rank_sim_std",
                    "uq_mis_cal",
                    "uq_NLL",
                    "uq_NLL_sim",
                    "uq_NLL_sim_std",
                    "Z_var",
                    "Z_var_CI_low",
                    "Z_var_CI_high",
                    "Z_mean",
                    "Z_mean_CI_low",
                    "Z_mean_CI_high",
                ]
            )
            if add_plots_to_table:
                # plots
                cols.extend(
                    [
                        "prediction_intervals",
                        "ordered_prediction_intervals",
                        "calibration",
                        "adversarial_group_calibration",
                        "sharpness",
                        "residuals_vs_pred_std",
                        "true_vs_preds",
                        "rmv_vs_rmse",
                        "Z_scores",
                        "calibration_curve",
                    ]
                )
        elif self.task_type == "classification":
            cols.extend(
                [
                    "PR-AUC",
                    "range_logAUC",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "PR@K",
                    "RP@K",
                ]
            )
        if self.aleatoric:
            cols.extend(
                [
                    "aleatoric_uct_mean",
                    "epistemic_uct_mean",
                    "total_uct_mean"
                ]
            )
        self.table = wandb.Table(columns=cols)

    def __call__(
        self,
        y_pred,
        y_std,
        y_true,
        y_err,
        y_alea=None,
        task_name=None
    ):
        """
        Calculate metrics and add them to the table.

        Parameters
        ----------
        y_pred : ndarray
            (Mean of) Predicted values.
        y_std : ndarray
            Standard deviation of predicted values.
        y_true : ndarray
            True values.
        task_name : str, optional
            Name of the task. The default is None.

        Returns
        -------
        metrics : dict
            Dictionary containing calculated metrics.
        """
        # y_pred, y_std, y_true, y_err, y_alea = self.not_nan_filter(y_pred, y_std, y_true, y_err)
        metrics, plots = self.calculate_metrics(
            y_pred=y_pred,
            y_std=y_std,
            y_true=y_true,
            y_err=y_err,
            y_alea=y_alea,
            task_name=task_name,
            data_specific_path=self.data_specific_path,
        )
        # if vars_ is not None:

        # self.export_plots(imgs, task_name)
        self.add_data(task_name=task_name, metrics=metrics, plots=plots)

        return metrics, plots
    # @staticmethod
    # def not_nan_filter(y_pred, y_std, y_true, y_err, y_alea=None):
    #     valid_mask = ~torch.isnan(y_true) # watch out the shapes Shape []
    #     y_pred = y_pred[valid_mask]
    #     y_std = y_std[valid_mask]
    #     y_true = y_true[valid_mask]
    #     y_err = y_err[valid_mask]
    #     if y_alea is not None:
    #         y_alea = y_alea[valid_mask]
    #
    #     return y_pred, y_std, y_true, y_err, y_alea

    def calculate_metrics(
        self, y_pred, y_std, y_true, y_err, y_alea, data_specific_path, task_name=None  # model_name=None,
    ):
        figures_path = FIGS_DIR / data_specific_path / self.model_name
        figures_path.mkdir(parents=True, exist_ok=True)
        # TODO Deal with NANs
        if self.task_type == "regression":
            uctmetrics, uctplots = calculate_uct_metrics(
                y_pred=y_pred,
                y_std=y_std,
                y_true=y_true,
                Nbins=100,
                task_name=task_name,
                figpath=figures_path,
            )
            # calculate other uqtools metrics
            uqmetrics, uqplots = calculate_uqtools_metrics(
                y_std,
                y_err,
                Nbins=100,
                include_bootstrap=True,
                task_name=task_name,
                figpath=figures_path,
                logger=self.logger,
            )
            metrics = {**uctmetrics, **uqmetrics}
            plots = {**uctplots, **uqplots}

        # elif self.task_type == "classification":
        else:
            metrics = calculate_tdc_classification_metrics(
                y_pred=y_pred,
                y_true=y_true,
                task_name=task_name,
                data_specific_path=data_specific_path,
            )
            # TODO classification plots
            plots = {}

        if self.aleatoric:
            y_alea_mean = y_alea.mean()
            y_std_mean = y_std.mean()
            metrics["aleatoric_uct_mean"] = y_alea_mean
            metrics["epistemic_uct_mean"] = y_std_mean
            metrics["total_uct_mean"] = y_alea_mean + y_std_mean

        plots = {k: wandb.Image(v) for k, v in plots.items()}

        return metrics, plots

    def add_data(self, task_name, metrics, plots):
        """
        Add data to the UCT metrics table.

        Parameters
        ----------
        task_name : str
            Name of the task.
        config : wandb.config
            Configuration object.
        metrics : dict
            Dictionary containing calculated metrics.
        plots : dict
            Dictionary containing UCT plots and other plots. The keys are the plot names.

        Returns
        -------
        None
        """
        vals = [self.model_type] if self.model_type is not None else []
        if self.mt:
            vals.append(task_name)
        vals.extend(
            [
                self.activity,
                self.split,
                self.desc_prot,
                self.desc_chem,
            ]
        )
        if self.task_type == "regression":
            vals.extend(
                [
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
                ]
            )
            # UQtools
            vals.extend(
                [
                    metrics["rho_rank"],
                    metrics["rho_rank_sim"],
                    metrics["rho_rank_sim_std"],
                    metrics["uq_mis_cal"],
                    metrics["uq_NLL"],
                    metrics["uq_NLL_sim"],
                    metrics["uq_NLL_sim_std"],
                    metrics["Z_var"],
                    metrics["Z_var_CI_low"],
                    metrics["Z_var_CI_high"],
                    metrics["Z_mean"],
                    metrics["Z_mean_CI_low"],
                    metrics["Z_mean_CI_high"],
                ]
            )
            if self.add_plots_to_table:
                # plots
                vals.extend(
                    [
                        plots["prediction_intervals"],
                        plots["ordered_prediction_intervals"],
                        plots["calibration"],
                        plots["adversarial_group_calibration"],
                        plots["sharpness"],
                        plots["residuals_vs_stds"],
                        plots["true_vs_predictions"],
                        # UQ tools
                        plots["rmv_vs_rmse"],
                        plots["Z_scores"],
                        plots["calibration_curve"],
                    ]
                )
        elif self.task_type == "classification":
            vals.extend(
                [
                    metrics["PR-AUC"],
                    metrics["range_logAUC"],
                    metrics["Accuracy"],
                    metrics["Precision"],
                    metrics["Recall"],
                    metrics["F1"],
                    metrics["PR@K"],
                    metrics["RP@K"],
                ]
            )
        if self.aleatoric:
            vals.extend(
                [
                    metrics["aleatoric_uct_mean"],
                    metrics["epistemic_uct_mean"],
                    metrics["total_uct_mean"]
                ]
            )

        self.table.add_data(*vals)
        plt.close()

    def wandb_log(self):
        """
        Export the UCT metrics table to wandb.
        """
        wandb.log({f" Uncertainty Metrics Table - {self.model_type}": self.table})


def recalibrate(
        y_true_recal, y_pred_recal, y_std_recal, y_err_recal,
        y_true_test, y_pred_test, y_std_test, y_err_test,
        n_subset=None,
        task_name="PCM",
        savefig: bool = True,
        save_dir: Path = "path/to/figures",
        uct_logger = None
):

    if savefig:
        before_path = Path(save_dir) / "Before_recal"
        after_path = Path(save_dir) / "After_recal"
        before_path.mkdir(exist_ok=True)
        after_path.mkdir(exist_ok=True)
        before_path_m = before_path / "metrics"
        after_path_m = after_path / "metrics"
        before_path_m.mkdir(exist_ok=True)
        after_path_m.mkdir(exist_ok=True)

    else:
        before_path = None
        after_path = None
        before_path_m = None
        after_path_m = None
    # Before Calibration
    # Plot average calibration
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    uct.viz.plot_calibration(y_pred_test, y_std_test, y_true_test, n_subset=n_subset, ax=ax1)
    ax1.set_title("Calibration Curve - Before Recalibration")
    plt.gcf().set_size_inches(4, 4)
    plt.tight_layout()

    if savefig:
        fig_save_path = Path(save_dir) / "Calib_curve_before_recalibration"
        uct.viz.save_figure(
            str(fig_save_path), ext_list=["png", "svg"], white_background=True
        )
    plt.show()
    plt.close()
    # TODO : uct logger here
    # metrics, plots = uct_logger(
    #     y_pred=y_pred,
    #     y_std=y_std,
    #     y_true=y_true,
    #     y_err=y_err,
    #     y_alea=y_alea,
    #     task_name="before_calibration",
    # )
    uctmetrics, _ = calculate_uct_metrics(
                y_pred=y_pred_test,
                y_std=y_std_test,
                y_true=y_true_test,
                Nbins=100,
                task_name=task_name,
                figpath=before_path_m,
            )
    uqmetrics, _ = calculate_uqtools_metrics(
        y_std_test,
        y_err_test,
        Nbins=100,
        include_bootstrap=True,
        task_name=task_name,
        figpath=after_path_m,
    )
    _before = {**uctmetrics, **uqmetrics}
    # wandb.log(data=_before)
    # plots_before = {**uctplots, **uqplots}
    plt.close()

    # Recalibrating
    y_pred_recal = y_pred_recal.flatten()
    y_std_recal = y_std_recal.flatten()
    exp_props, obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(
        y_pred_recal, y_std_recal, y_true_recal,
    )
    # Train a recalibration model.
    recal_model = uct.recalibration.iso_recal(exp_props, obs_props)
    # Get the expected props and observed props using the new recalibrated model
    te_recal_exp_props, te_recal_obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(
        y_pred_test, y_std_test, y_true_test, recal_model=recal_model
    )
    # Show the updated average calibration plot AFTER
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    uct.viz.plot_calibration(
        y_pred_test,
        y_std_test,
        y_true_test,
        n_subset=n_subset,
        exp_props=te_recal_exp_props,
        obs_props=te_recal_obs_props,
        ax=ax2
    )
    ax2.set_title("Calibration Curve - After Recalibration")
    plt.gcf().set_size_inches(4.0, 4.0)
    plt.tight_layout()

    if savefig:
        fig_save_path = Path(save_dir) / "Calib_curve_after_recalibration"
        uct.viz.save_figure(
            str(fig_save_path), ext_list=["png", "svg"], white_background=True
        )
    plt.show()
    plt.close()

    uctmetrics, _ = calculate_uct_metrics(
                y_pred=y_pred_test,
                y_std=y_std_test,
                y_true=y_true_test,
                Nbins=100,
                task_name=task_name,
                figpath=after_path_m,
            )
    uqmetrics, _ = calculate_uqtools_metrics(
        y_std_test,
        y_err_test,
        Nbins=100,
        include_bootstrap=True,
        task_name=task_name,
        figpath=after_path_m,
    )
    _after = {**uctmetrics, **uqmetrics}
    # plots_before = {**uctplots, **uqplots}
    plt.close()

    return recal_model



def calc_aleatoric_mean_var_notnan(vars_all, targets_all):
    """
    Calculate the mean and variance of vars_all considering only the valid (non-NaN) corresponding targets.

    Parameters:
    vars_all : torch.Tensor
        Aleatoric variances from the model, shape [batch_size, num_tasks, num_models]
    targets_all : torch.Tensor
        True target values, shape [batch_size, num_tasks]

    Returns:
    tuple of torch.Tensor
        Mean and variance of vars_all considering only valid entries, scalar values.
    """
    # Ensure targets_all has an additional dimension for broadcasting compatibility
    # if targets_all.dim() < vars_all.dim():
    #     targets_all = targets_all.unsqueeze(-1)  # Shape [datapoints, tasks, 1]

    # Create a mask of valid (non-NaN) targets
    valid_mask = ~torch.isnan(targets_all)  # Shape [datapoints, tasks, 1] [6525, 20, 1]

    # Apply the mask to vars_all
    # valid_vars = torch.where(valid_mask, vars_all, torch.tensor(float('nan'), device=vars_all.device))

    # Flatten the valid_vars to a 1D array for mean and variance calculation
    # if valid_mask.shape[]
    valid_vars_flat = vars_all[valid_mask]

    # Calculate mean and variance on the flattened valid data
    vars_mean = torch.mean(valid_vars_flat)
    vars_var = torch.var(valid_vars_flat)

    return vars_mean, vars_var


#
# def log_mol_table(smiles, inputs, targets, outputs, targets_names):
#     data = []
#     for smi, inp, tar, out in zip(
#         smiles, inputs.to("cpu"), targets.to("cpu"), outputs.to("cpu")
#     ):
#         row = {
#             "smiles": smi,
#             "molecule": wandb.Molecule.from_smiles(smi),
#             "molecule_2D": wandb.Image(smi_to_pil_image(smi)),
#             "ECFP": inp,
#             "fp_length": len(inp),
#         }
#
#         # Iterate over each pair of output and target
#         for targetName, target, output in zip(targets_names, tar, out):
#             row[f"{targetName}_label"] = target.item()
#             row[f"{targetName}_predicted"] = output.item()
#
#         data.append(row)
#
#     dataframe = pd.DataFrame.from_records(data)
#     table = wandb.Table(dataframe=dataframe)
#     wandb.log({"mols_table": table}, commit=False)
#
#
# def _make_uct_plots(
#     y_preds: np.ndarray,
#     y_std: np.ndarray,
#     y_true: np.ndarray,
#     task_name: str = None,
#     n_subset: int = 100,
#     ylims: Tuple[float, float] | None = (-3.0, 3.0),
#     num_stds_confidence_bound: float = 2.0,  # 1.96 for 95% CI for normal distribution
#     plot_save_str: str = "uct_plot",
#     savefig: bool = True,
#     save_dir: Path = FIGS_DIR,
# ) -> plt.Figure:
#     """
#     Generate a set of UCT plots including prediction intervals, calibration, adversarial
#     group calibration, sharpness, and residuals vs. predictive standard deviation.
#
#     Parameters
#     ----------
#     y_preds : np.ndarray
#         Predicted values.
#     y_std : np.ndarray
#         Standard deviations of predictions.
#     y_true : np.ndarray
#         True target values.
#     task_name : str, optional
#         Name of the task, used in plot titles.
#     n_subset : int, optional
#         Number of points to subset for plotting.
#     ylims : tuple, optional
#         Y-limits for the plots.
#     num_stds_confidence_bound : float, optional
#         Number of standard deviations for confidence interval.
#     plot_save_str : str, optional
#         String to be used in the plot's save path.
#     savefig : bool, optional
#         Whether to save the figure.
#     save_dir : Path, optional
#         Directory to save the figure.
#
#     Returns
#     -------
#     plt.Figure
#         The matplotlib figure object containing all UCT plots.
#     """
#     if not all(isinstance(arr, np.ndarray) for arr in [y_preds, y_std, y_true]):
#         raise ValueError("y_preds, y_std, and y_true must be numpy arrays.")
#
#     task_name = task_name if task_name is not None else "PCM"
#
#     fig, axs = plt.subplots(2, 4, figsize=(30, 10))
#     axs = axs.flatten()  # Flatten to index easily
#     # Prediction Intervals plot
#     uct_viz.plot_intervals(
#         y_preds,
#         y_std,
#         y_true,
#         n_subset=n_subset,
#         ylims=ylims,
#         num_stds_confidence_bound=num_stds_confidence_bound,
#         ax=axs[0],
#     )
#     axs[0].set_title(f"Prediction Intervals - {task_name}")
#     rmse = np.sqrt(np.mean((y_preds - y_true) ** 2))
#     axs[0].text(
#         0.05,
#         0.95,
#         f"RMSE: {rmse:.2f}",
#         transform=axs[0].transAxes,
#         verticalalignment="top",
#     )
#
#     # Calibration plot
#     uct_viz.plot_calibration(y_preds, y_std, y_true, n_subset=n_subset, ax=axs[1])
#     axs[1].set_title(f"Average Calibration - {task_name}")
#
#     # Adversarial group calibration plot
#     uct_viz.plot_adversarial_group_calibration(
#         y_preds, y_std, y_true, n_subset=n_subset, ax=axs[2]
#     )
#     axs[2].set_title(f"Adversarial Group Calibration - {task_name}")
#
#     # Sharpness plot
#     uct_viz.plot_sharpness(y_std, n_subset=n_subset, ax=axs[3])
#     axs[3].set_title(f"Sharpness - {task_name}")
#
#     # Residuals vs. Predictive Std plot
#     uct_viz.plot_residuals_vs_stds(y_preds, y_std, y_true, n_subset=n_subset, ax=axs[4])
#     axs[4].set_title(f"Residuals vs. Predictive Std - {task_name}")
#
#     # New plot: Ordered Prediction Intervals
#     uct_viz.plot_intervals_ordered(
#         y_preds,
#         y_std,
#         y_true,
#         n_subset=n_subset,
#         ylims=ylims,
#         num_stds_confidence_bound=num_stds_confidence_bound,
#         ax=axs[5],
#     )
#     axs[5].set_title(f"Ordered Prediction Intervals - {task_name}")
#
#     # New plot: True vs Predictions Plot
#     plot_true_vs_preds(y_preds, y_true, task_name=task_name, ax=axs[6])
#
#     plt.tight_layout(pad=2.0)
#
#     # Save figure if required
#     if savefig:
#         full_save_path = Path(save_dir) / plot_save_str
#         uct_viz.save_figure(
#             str(full_save_path), ext_list=["png", "svg"], white_background=True
#         )
#
#     return fig
#
#
# def _2make_uct_plots(
#     y_preds,
#     y_std,
#     y_true,
#     task_name=None,
#     n_subset=100,
#     ylims=(-3, 3),
#     num_stds_confidence_bound=2.0,  # Use 1.96 for 95% confidence interval for normal distribution
#     plot_save_str="uct_plot",  # "row",
#     savefig=True,
#     save_dir=FIGS_DIR,
# ):
#     """
#     Make set of plots.
#     Adapted from https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/main/examples/viz_readme_figures.py
#     """
#     if (
#         not isinstance(y_preds, np.ndarray)
#         or not isinstance(y_std, np.ndarray)
#         or not isinstance(y_true, np.ndarray)
#     ):
#         raise ValueError("y_preds, y_std, and y_true must be numpy arrays.")
#     task_name = str(task_name) if task_name else "PCM"
#
#     fig, axs = plt.subplots(1, 5, figsize=(25, 5))  # (28, 8)
#
#     axs[0] = uct.plot_intervals(
#         y_preds,
#         y_std,
#         y_true,
#         n_subset=n_subset,
#         ylims=ylims,
#         num_stds_confidence_bound=num_stds_confidence_bound,
#         ax=axs[0],
#     )
#     axs[0].set_title("Prediction Intervals - {}".format(task_name))
#     # calculate RMSE and add it to the plot left upper corner
#     rmse = np.sqrt(np.mean((y_preds - y_true) ** 2))
#     axs[0].text(0.05, 0.95, "RMSE: {:.2f}".format(rmse), transform=axs[0].transAxes)
#
#     # Make calibration plot
#     axs[1] = uct.plot_calibration(y_preds, y_std, y_true, n_subset=n_subset, ax=axs[1])
#     axs[1].set_title("Average Calibration - {}".format(task_name))
#
#     # Make adversarial group calibration plot
#     axs[2] = uct.plot_adversarial_group_calibration(
#         y_preds, y_std, y_true, n_subset=n_subset, ax=axs[2]
#     )
#     axs[2].set_title("Adversarial Group Calibration - {}".format(task_name))
#
#     # Make sharpness plot
#     axs[3] = uct.plot_sharpness(y_std, n_subset=n_subset, ax=axs[3])
#     axs[3].set_title("Sharpness - {}".format(task_name))
#
#     # Make residual vs stds plot
#     axs[4] = uct.plot_residuals_vs_stds(
#         y_preds, y_std, y_true, n_subset=n_subset, ax=axs[4]
#     )
#     axs[4].set_title("Residuals vs. Predictive Std - {}".format(task_name))
#
#     # Adjust subplots spacing
#     fig.subplots_adjust(wspace=0.5)
#     # Adjust the spacing between subplots
#     plt.tight_layout()
#
#     # Save figure
#     if savefig:
#         full_save_path = Path(save_dir) / f"{plot_save_str}"
#         uct.viz.save_figure(
#             str(full_save_path), ext_list=["png", "svg"], white_background=True
#         )
#
#     return fig
