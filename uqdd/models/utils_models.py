import math
import os
import random
import logging
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import wandb
from torch import nn
from torch.utils.data import DataLoader

from uqdd import CONFIG_DIR, MODELS_DIR, TODAY, DEVICE
from uqdd.utils import get_config, create_logger

string_types = (type(b""), type(""))


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    Parameters:
    -----------
    - seed (int): The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure that cudnn is deterministic
    torch.backends.cudnn.deterministic = True

    # Disabling the benchmark mode so that deterministic algorithms are used
    torch.backends.cudnn.benchmark = False


# Adapted from ChemProp - Uncertainty
def compute_pnorm(model: nn.Module) -> float:
    """Computes the norm of the parameters of a model."""
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:
    """Computes the norm of the gradients of a model."""
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))


def get_desc_len_from_dataset(dataset):  # , desc_prot=True):
    """
    Get the length of the descriptors for the protein and chemical  inputs
    Parameters
    ----------
    dataset
        One of the datasets from the get_datasets function train or val or test.

    Returns
    -------
    desc_prot_len: int
        Length of the protein descriptors
    desc_chem_len: int
        Length of the chemical descriptors
    """
    # desc_prot then descriptor_chemical
    inputs = dataset[0][0]
    if isinstance(inputs, tuple):
        desc_prot = inputs[0]
        desc_chem = inputs[1]
        return desc_prot.shape[0], desc_chem.shape[0]
    elif isinstance(inputs, torch.Tensor):
        return 0, inputs.shape[0]
    else:
        raise ValueError("Unknown input type to get_desc_len function.")


def get_desc_len(*descriptors: str):
    desc_config = get_config(config_name="desc_dim", config_dir=CONFIG_DIR)
    lengths = []
    for d in descriptors:
        lengths.append(desc_config.get(d, 0))
    return tuple(lengths)


def get_model_config(model_name="baseline", **kwargs):
    """
    Retrieve the configuration dictionary for model training.

    Parameters:
    - config (dict or None): A dictionary containing configuration parameters. If None, the default configuration will be loaded.
    - **kwargs: Additional keyword arguments that override the values in the config dictionary.

    Returns:
    - dict: Merged dictionary containing the configuration parameters.

    Notes:
    - If `config` is None, the function will load the default configuration from a JSON file.
    - The default configuration values will be overridden by `config` and `kwargs`, if provided.
    - If both `config` and `kwargs` contain the same key, the value from `kwargs` will take precedence.

    Examples:
    # Example 1: Read from default config file
    config = get_config()

    # Example 2: Read from a custom config file
    config = get_config(config="path/to/custom/config.json")

    # Example 3: Provide config as a dictionary
    custom_config = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "num_epochs": 500
    }
    config = get_config(config=custom_config)

    # Example 4: Provide config as a dictionary and additional keyword arguments
    config = get_config(config=custom_config, num_epochs=1000, batch_size=32)
    """
    assert model_name in [
        "baseline",
        "ensemble",
        "mcdropout",
        "evidential",
        "gp",
    ], f"Invalid model name: {model_name}"
    return get_config(config_name=model_name, config_dir=CONFIG_DIR, **kwargs)


def get_sweep_config(model_name="baseline", **kwargs):
    """
    Retrieve the sweep configuration for hyperparameter tuning.

    Parameters:
    -----------
    - config (dict, str, or None): A dictionary containing sweep configuration parameters,
    a path to a YAML or JSON config file, or None to use the default configuration.
    - **kwargs: Additional keyword arguments that override the values in the 'parameters' dictionary.

    Returns:
    --------
    - dict: Merged dictionary containing the sweep configuration parameters.

    Notes:
    ------
    - If `config` is None, the function will return the default sweep configuration.
    - If `config` is a path to a YAML or JSON file, the function will load the configuration from the file.
    - The default configuration values will be overridden by `config` and `kwargs`, if provided.
    - If both `config` and `kwargs` contain the same key in the 'parameters' dictionary,
        the value from `kwargs` will take precedence.
    """
    assert model_name in [
        "baseline",
        "ensemble",
        "mcdropout",
        "evidential",
        "gp",
    ], f"Invalid model name: {model_name}"
    kwargs = {k: {"value": v} for k, v in kwargs.items()}
    config = get_config(
        config_name=f"{model_name}-sweep",
        config_dir=CONFIG_DIR,
    )
    config["parameters"].update(kwargs)
    return config


def build_datasets(
    data_name="papyrus",
    n_targets: int = -1,
    activity_type: str = "xc50",
    split_type: str = "random",
    desc_prot: Union[str, None] = None,
    desc_chem: Union[str, None] = None,
    median_scaling: bool = False,
    task_type: str = "regression",
    # label_scaling_func: Callable[[torch.Tensor], torch.Tensor] | str = None,
    ext: str = "pkl",
    logger: logging.Logger = None,
):
    logger = create_logger(name="build_datasets") if not logger else logger
    logger.debug(f"Building datasets for {data_name}")
    # if isinstance(label_scaling_func, str):
    #     label_scaling_func = get_label_scaling_func(scaling_type=label_scaling_func)
    if data_name == "papyrus":
        from uqdd.data.data_papyrus import get_datasets

        datasets = get_datasets(
            n_targets=n_targets,
            activity_type=activity_type,
            split_type=split_type,
            desc_prot=desc_prot,
            desc_chem=desc_chem,
            median_scaling=median_scaling,
            task_type=task_type,
            ext=ext,
            logger=logger,
        )
    elif data_name == "tdc":
        from uqdd.data.data_tdc import get_datasets

        datasets = get_datasets()
    elif data_name == "other":
        from uqdd.data.data_other import get_datasets

        datasets = get_datasets()
    else:
        raise ValueError(
            f"Unknown data name: {data_name}"
            f"Please choose from 'papyrus', 'tdc', or 'other'"
        )

    return datasets


def build_loader(datasets, batch_size, shuffle=False):
    try:
        # num_cpu_cores = os.cpu_count()
        dataloaders = {}
        for k, v in datasets.items():
            dataloaders[k] = DataLoader(
                v,
                batch_size=batch_size,
                shuffle=shuffle,
                # num_workers=4,
                # pin_memory=True,
            )
        logging.info("Data loaders created")
    except Exception as e:
        raise RuntimeError(f"Error loading data {e}")

    return dataloaders


def _build_loader(datasets, batch_size, ecfp_size=1024):
    try:
        train_set = datasets[f"train_ecfp{ecfp_size}"]
        val_set = datasets[f"val_ecfp{ecfp_size}"]
        test_set = datasets[f"test_ecfp{ecfp_size}"]
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        logging.info("Data loaders created")
    except Exception as e:
        raise RuntimeError(f"Error loading data {e}")

    return train_loader, val_loader, test_loader


def build_optimizer(model, optimizer, lr, weight_decay):
    if optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError("Unknown optimizer: {}".format(optimizer))

    return optimizer


# ### Custom Loss Functions ###
# class MultiTaskLoss(nn.Module):
#     def __init__(self, loss_type="huber", reduction="mean"):
#         super(MultiTaskLoss, self).__init__()
#         self.loss_type = loss_type
#         self.loss_fn = build_loss(loss_type, reduction=reduction)
#
#     def forward(self, outputs, targets):
#         nan_mask = torch.isnan(targets)
#         # loss
#         loss = calc_loss_notnan(outputs, targets, nan_mask, self.loss_fn)
#         return loss
#
#
# def build_loss(loss, reduction="none", mt=False):
#     if mt:
#         return MultiTaskLoss(loss_type=loss, reduction=reduction)
#
#     if loss.lower() in ["mse", "l2"]:
#         loss_fn = nn.MSELoss(reduction=reduction)
#     elif loss.lower() in ["mae", "l1"]:
#         loss_fn = nn.L1Loss(reduction=reduction)
#     elif loss.lower() in ["huber", "smoothl1"]:
#         loss_fn = nn.SmoothL1Loss(reduction=reduction)
#     else:
#         raise ValueError("Unknown loss: {}".format(loss))
#     return loss_fn


def build_lr_scheduler(optimizer, lr_scheduler, patience=20, factor=0.2, **kwargs):
    if lr_scheduler is None:
        return None
    elif lr_scheduler.lower() == "plateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=factor, patience=patience, mode="min", **kwargs
        )
    elif lr_scheduler.lower() == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=patience, gamma=factor, **kwargs
        )
    elif lr_scheduler.lower() == "exp":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=factor, **kwargs
        )
    elif lr_scheduler.lower() == "cos":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=patience, **kwargs
        )
    else:
        raise ValueError("Unknown lr_scheduler: {}".format(lr_scheduler))

    return lr_scheduler


# def calc_nanaware_metrics(tensor, nan_mask, all_tasks_agg=False):
#     """
#     Aggregate a tensor by excluding NaN values based on a nan_mask.
#
#     Calculates the mean of the non-NaN values along the specified dimension.
#     Optionally, it can aggregate the mean across all tasks.
#
#     Parameters
#     ----------
#     tensor : torch.Tensor
#         The input tensor to be aggregated.
#     nan_mask : torch.Tensor
#         A boolean mask indicating the NaN values in the tensor.
#     all_tasks_agg : bool or str, optional
#         Determines whether to aggregate across all tasks. If False (default),
#         returns the mean for each task. If 'mean', returns the mean of all tasks.
#         If 'sum', returns the sum of all tasks.
#
#     Returns
#     -------
#     torch.Tensor
#         The aggregated tensor based on the specified aggregation method.
#
#     Notes
#     -----
#     - The nan_mask should have the same shape as tensor_for_agg.
#     - The nan_mask should be a boolean tensor with True indicating NaN values.
#
#     Examples
#     --------
#     >>> import torch
#     >>> tensor_for_agg = torch.tensor([[1, 2, 3], [4, float('nan'), 6]])
#     >>> nan_mask = torch.isnan(tensor_for_agg)
#     >>> aggregated_tensor = calc_nanaware_metrics(tensor_for_agg, nan_mask, all_tasks_agg=True)
#     >>> print(aggregated_tensor)
#     tensor(3.3333)
#
#     The above example demonstrates the usage of the `agg_notnan` function.
#     The input tensor contains NaN values, and the nan_mask is used to identify those NaN values.
#     By specifying `all_tasks_agg=True`, the function calculates the mean of the non-NaN values and then
#     returns the mean of all tasks. In this case, the output is `tensor(3.3333)`.
#     """
#     # Now we only include the non-Nan targets in the mean calc.
#     # tensor_means = torch.sum(tensor, dim=0) / torch.sum(~nan_mask, dim=0)
#
#     valid_values = torch.where(
#         ~nan_mask, tensor, torch.tensor(0.0, device=tensor.device)
#     )
#     sum_values = torch.sum(valid_values, dim=0)
#     valid_counts = torch.sum(~nan_mask, dim=0)
#
#     tensor_means = sum_values / valid_counts.clamp(min=1)  # Avoid division by zero
#
#     # If we want to aggregate across all tasks, we do so here.
#     # if not all_tasks_agg:
#     #     return tensor_means
#     if all_tasks_agg == "mean":
#         return torch.nanmean(tensor_means)
#     elif all_tasks_agg == "sum":
#         return torch.nansum(tensor_means)
#     return tensor_means
#
#
# def calc_regr_metrics(targets, outputs):
#     # Handle extra dimensions (ensembles) by averaging ensemble predictions
#     if outputs.dim() > targets.dim():
#         outputs = outputs.mean(dim=-1)
#
#     # Adjust dimensions if necessary (for MTL to STL comparison)
#     if targets.dim() < outputs.dim():
#         targets = targets.unsqueeze(-1)
#
#     targets = targets.cpu().numpy()
#     outputs = outputs.cpu().numpy()
#
#     # no reduction here because we want to calc per task metrics
#     rmse = mean_squared_error(targets, outputs, squared=False, multioutput="raw_values")
#     r2 = r2_score(targets, outputs, multioutput="raw_values")
#     evs = explained_variance_score(targets, outputs, multioutput="raw_values")
#
#     return rmse, r2, evs
#
#
# #
# # def calc_loss_notnan(outputs, targets, nan_mask, loss_fn):
# #     valid_targets = torch.where(
# #         ~nan_mask, targets, torch.tensor(0.0, device=targets.device)
# #     )
# #     valid_outputs = torch.where(
# #         ~nan_mask, outputs, torch.tensor(0.0, device=outputs.device)
# #     )
# #
# #     loss_per_task = loss_fn(valid_outputs, valid_targets, reduction="none")
# #     loss = calc_nanaware_metrics(loss_per_task, nan_mask, all_tasks_agg="sum")
# #
# #     return loss
#
# # targets[nan_mask], outputs[nan_mask] = 0.0, 0.0
# #
# # loss_per_task = loss_fn(outputs, targets)
# #
# # # Now we only include the non-Nan targets in the mean calc.
# # loss = calc_nanaware_metrics(
# #     tensor=loss_per_task, nan_mask=nan_mask, all_tasks_agg="sum"
# # )
# # # task_losses = torch.sum(loss_per_task, dim=1) / torch.sum(~nan_mask, dim=1)
# # # loss = torch.sum(task_losses)
# # return loss
#
#
# def process_preds(predictions, targets, task_idx=None):
#     # Get the predictions mean and std
#     preds_mu = predictions.mean(dim=-1)  # (dim=2)
#     preds_std = predictions.std(dim=-1)  # (dim=2)
#
#     if task_idx is not None:
#         # For MTL, select predictions for the specific task
#         preds_mu = preds_mu[:, task_idx]
#         preds_std = preds_std[:, task_idx]
#         targets = targets[:, task_idx]
#     elif predictions.dim() > 2 and targets.dim() == 1:
#         preds_mu = preds_mu.flatten()
#         preds_std = preds_std.flatten()
#     # else:
#     #     # flatten # TODO I dont think this works for STL
#     #     preds_mu = torch.flatten(preds_mu.transpose(0, 1))
#     #     preds_std = torch.flatten(preds_std.transpose(0, 1))
#     #     targets = torch.flatten(targets.transpose(0, 1))
#
#     # nan mask filter
#     nan_mask = ~torch.isnan(targets)
#     preds_mu = preds_mu[nan_mask]
#     preds_std = preds_std[nan_mask]
#     targets = targets[nan_mask]
#
#     # convert to numpy and to cpu
#     y_true = targets.cpu().numpy()
#     y_pred = preds_mu.cpu().numpy()
#     y_std = preds_std.cpu().numpy()
#     y_err = abs(y_true - y_pred)
#
#     return y_true, y_pred, y_std, y_err


def save_model(
    config,
    model,
    model_name=f"{TODAY}-baseline_random_ankh-base_ecfp2048",
    data_specific_path=None,
    desc_prot_len=0,
    desc_chem_len=1024,
    onnx=True,
):
    # n_targets=-1,
    # data_name="papyrus",
    # activity_type="xc50",
    # split_type="random",
    # desc_prot=None,
    # desc_chem=None,
    try:
        if model is None:
            raise ValueError(f"No model to save - {model=}")

        model_dir = MODELS_DIR / "saved_models" / data_specific_path
        # model_dir = (
        #     MODELS_DIR
        #     / "saved_models"
        #     / data_name
        #     / activity_type
        #     / get_topx(n_targets)
        # )
        model_dir.mkdir(parents=True, exist_ok=True)
        # model_name = f"{TODAY}-{model_type}_{split_type}_{desc_prot}_{desc_chem}-{wandb.run.name}"

        pt_path = model_dir / f"-{model_name}.pt"
        torch.save(model.state_dict(), pt_path)
        wandb_model_path = pt_path

        if onnx:
            onnx_path = model_dir / f"{model_name}.onnx"
            # batch_size = config.get("batch_size", 64)
            # in case we have two inputs ?
            if desc_prot_len == 0:
                dummy_input = {
                    "inputs": torch.zeros(
                        (config.batch_size, desc_chem_len),
                        dtype=torch.float32,
                        device=DEVICE,
                        requires_grad=False,
                    )
                }
            else:
                dummy_input = {
                    "inputs": tuple(
                        (
                            torch.zeros(
                                (config.batch_size, desc_prot_len),
                                dtype=torch.float32,
                                device=DEVICE,
                                requires_grad=False,
                            ),
                            torch.zeros(
                                (config.batch_size, desc_chem_len),
                                dtype=torch.float32,
                                device=DEVICE,
                                requires_grad=False,
                            ),
                        )
                    )
                }
            torch.onnx.export(model, dummy_input, onnx_path)
            wandb_model_path = str(onnx_path)

        # Model logging
        wandb.save(wandb_model_path, base_path=model_dir)

    except Exception as e:
        print("Error saving models: " + str(e))


# def make_uct_plots(
#     y_preds,
#     y_std,
#     y_true,
#     task_name=None,
#     n_subset=100,
#     ylims=(-3, 3),
#     num_stds_confidence_bound=2.0,  # Use 1.96 for 95% confidence interval for normal distribution
#     plot_save_str="row",
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
#         uct.viz.save_figure(
#             plot_save_str, ext_list=["png", "svg"], white_background=True
#         )
#
#     return fig
#
#
# def plot_true_vs_preds(
#     y_preds,
#     y_true,
#     task_name,
#     save_path=None,
# ):
#     # Sort the values based on y_true
#     sorted_indices = np.argsort(y_true)
#     sorted_y_true = y_true[sorted_indices]
#     sorted_y_preds = y_preds[sorted_indices]
#
#     # Plot the graph
#     fig, ax = plt.subplots()
#     ax.plot(sorted_y_true, sorted_y_preds, "o", label="Predictions")
#     ax.set_xlabel("True Values")
#     ax.set_ylabel("Predicted Values")
#     ax.set_title("True vs. Predicted Values - {}".format(task_name))
#
#     # Calculate the best-fitting line
#     best_fit_coeffs = np.polyfit(sorted_y_true, sorted_y_preds, deg=1)
#     best_fit_line = np.poly1d(best_fit_coeffs)
#     ax.plot(
#         sorted_y_true, best_fit_line(sorted_y_true), color="red", label="Best Fit Line"
#     )
#
#     # Calculate the distances of each point to the best-fitted line
#     distances = np.abs(best_fit_line(sorted_y_true) - sorted_y_preds)
#     normalized_distances = distances / np.max(distances)
#     ax.plot(sorted_y_true, sorted_y_preds, color="gray", alpha=0.2)
#     # for i in range(len(sorted_y_true)):
#     #     ax.plot([sorted_y_true[i], sorted_y_true[i]], [sorted_y_preds[i], best_fit_line(sorted_y_true[i])],
#     #              color='gray', alpha=normalized_distances[i])
#     # Plot the grey lines between dots and best fit line
#     for i in range(len(sorted_y_true)):
#         ax.plot(
#             [sorted_y_true[i], sorted_y_true[i]],
#             [sorted_y_preds[i], best_fit_line(sorted_y_true[i])],
#             color="gray",
#             alpha=0.2,
#         )
#
#     # Calculate and display the RMSE
#     rmse = np.sqrt(np.mean((sorted_y_preds - best_fit_line(sorted_y_true)) ** 2))
#     # ax.text(0.05, 0.9, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes)
#     ax.text(
#         0.95, 0.05, f"RMSE: {rmse:.2f}", transform=ax.transAxes, ha="right", va="bottom"
#     )
#
#     # Show the legend and display/save the graph
#     ax.legend(loc="upper left")
#
#     if save_path is not None:
#         fig.savefig(save_path)
#         print(f"Saved true_vs_preds_plot to {save_path}")
#     else:
#         plt.show()
#
#     return fig
#
#
# def calculate_uct_metrics(
#     y_pred, y_std, y_true, task_name, activity, split, model_type="ensemble"
# ):
#     """
#     Calculate metrics for the predictions.
#
#     Parameters
#     ----------
#     y_pred : ndarray
#         (Mean of) Predicted values.
#     y_std : ndarray
#         Standard deviation of predicted values.
#     y_true : ndarray
#         True values.
#     task_name : str
#         Name of the task.
#     activity : str
#         Activity name.
#     split : str
#         Split name.
#     model_type : str, optional
#         Type of the model. The default is "ensemble".
#
#     Returns
#     -------
#     metrics : dict
#         Dictionary containing calculated metrics.
#     img     : wandb.Image
#         Image of the UCT plot - ready for logging
#     """
#     metrics = get_all_metrics(
#         y_pred=y_pred,
#         y_std=y_std,
#         y_true=y_true,
#         num_bins=100,
#         resolution=99,
#         scaled=True,
#         verbose=False,
#     )
#
#     figures_path = os.path.join(FIGS_DIR, model_type, activity, split)
#     os.makedirs(figures_path, exist_ok=True)
#
#     metrics_filepath = os.path.join(figures_path, f"{task_name}_metrics.pkl")
#     with open(metrics_filepath, "wb") as file:
#         pickle.dump(metrics, file)
#
#     fig = make_uct_plots(
#         y_pred,
#         y_std,
#         y_true,
#         task_name=task_name,
#         n_subset=min(500, len(y_true)),
#         ylims=None,
#         num_stds_confidence_bound=1.96,
#         plot_save_str=os.path.join(figures_path, f"{task_name}_uct"),
#         savefig=True,
#     )
#
#     img = wandb.Image(fig)
#
#     return metrics, img
#
#
# class UCTMetricsTable:
#     def __init__(self, model_type=None, config=None, multitask=False):
#         """
#         Initialize the UCT metrics table.
#
#         Returns:
#         --------
#         None
#         """
#         cols = []
#         # self.task_name = task_name
#         self.config = config
#         self.model_type = model_type
#         self.mt = multitask
#         if model_type is not None:
#             cols = ["Model type"]
#         if multitask:
#             cols.extend(["Target"])
#         cols.extend(
#             [
#                 "Multi-Task",
#                 "Activity",
#                 "Split",
#                 "RMSE",
#                 "R2",
#                 "MAE",
#                 "MADAE",
#                 "MARPD",
#                 "Correlation",
#                 "RMS Calibration",
#                 "MA Calibration",
#                 "Miscalibration Area",
#                 "Sharpness",
#                 "NLL",
#                 "CRPS",
#                 "Check",
#                 "Interval",
#                 "UCT plots",
#             ]
#         )
#         self.table = wandb.Table(columns=cols)
#
#     def __call__(self, y_pred, y_std, y_true, task_name=None):
#         """
#         Calculate metrics and add them to the table.
#
#         Parameters
#         ----------
#         y_pred : ndarray
#             (Mean of) Predicted values.
#         y_std : ndarray
#             Standard deviation of predicted values.
#         y_true : ndarray
#             True values.
#         task_name : str, optional
#             Name of the task. The default is None.
#
#         Returns
#         -------
#         metrics : dict
#             Dictionary containing calculated metrics.
#         """
#         metrics, img = self.calculate_metrics(
#             y_pred=y_pred, y_std=y_std, y_true=y_true, task_name=task_name
#         )
#         self.add_data(task_name=task_name, config=self.config, metrics=metrics, img=img)
#
#         return metrics
#
#     def calculate_metrics(self, y_pred, y_std, y_true, task_name=None):
#         metrics, img = calculate_uct_metrics(
#             y_pred=y_pred,
#             y_std=y_std,
#             y_true=y_true,
#             task_name=task_name,
#             activity=self.config.activity,
#             split=self.config.split,
#             model_type=self.model_type,
#         )
#         return metrics, img
#
#     def add_data(self, task_name, config, metrics, img):
#         """
#         Add data to the UCT metrics table.
#
#         Parameters
#         ----------
#         task_name : str
#             Name of the task.
#         config : wandb.config
#             Configuration object.
#         metrics : dict
#             Dictionary containing calculated metrics.
#         img : wandb.Image
#             Image of the UCT plot.
#
#         Returns
#         -------
#         None
#         """
#         vals = [self.model_type] if self.model_type is not None else []
#         vals.extend(
#             [
#                 task_name,
#                 config.activity,
#                 config.split,
#                 metrics["accuracy"]["rmse"],
#                 metrics["accuracy"]["r2"],
#                 metrics["accuracy"]["mae"],
#                 metrics["accuracy"]["mdae"],
#                 metrics["accuracy"]["marpd"],
#                 metrics["accuracy"]["corr"],
#                 metrics["avg_calibration"]["rms_cal"],
#                 metrics["avg_calibration"]["ma_cal"],
#                 metrics["avg_calibration"]["miscal_area"],
#                 metrics["sharpness"]["sharp"],
#                 metrics["scoring_rule"]["nll"],
#                 metrics["scoring_rule"]["crps"],
#                 metrics["scoring_rule"]["check"],
#                 metrics["scoring_rule"]["interval"],
#                 img,
#             ]
#         )
#
#         self.table.add_data(*vals)
#         plt.close()
#
#     def wandb_log(self):
#         """
#         Export the UCT metrics table to wandb.
#         """
#         wandb.log({f"UCT Metrics Table {self.model_type}": self.table})
#
#
# def uct_metrics_logger(uct_metrics_table, task_name, config, metrics, img):
#     """
#     Log UCT metrics to the UCT metrics table.
#
#     Parameters
#     ----------
#     uct_metrics_table : UCTMetricsTable
#         UCT metrics table object for logging.
#     task_name : str
#         Name of the task.
#     config : wandb.config
#         Configuration object.
#     metrics : dict
#         Dictionary containing calculated metrics.
#     img : wandb.Image
#         Image of the UCT plot.
#
#     Returns
#     -------
#     None
#     """
#     uct_metrics_table.add_data(
#         task_name,
#         config,
#         metrics,
#         img,
#     )
#     plt.close()
#
#
# def log_mol_table(smiles, inputs, targets, outputs, targets_names):
#     # targets_cols = targets.columns()
#     # table_cols = ['smiles', 'mol', 'mol_2D', 'ECFP', 'fp_length']
#     # for t in targets_cols:
#     #     table_cols.append(f'{t}_label')
#     #     table_cols.append(f'{t}_predicted')
#     # table = wandb.Table(columns=table_cols)
#     # with wandb.init(dir=wandb_dir, mode=wandb_mode):
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
# def postmodel_metrics():
#     raise NotImplementedError
