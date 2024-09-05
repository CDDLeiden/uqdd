import argparse
from tqdm import tqdm
import wandb
import torch
from uqdd import DEVICE
from uqdd.models.baseline import BaselineDNN
from uqdd.utils import create_logger, parse_list

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


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def mc_predict(model, test_loader, num_mc_samples=100, device=DEVICE):
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


def run_mcdropout(config=None):
    if config is None:
        config = get_model_config(
            "mcdropout", split_type="random", activity_type="xc50"
        )  # * Defaulting to random split_type and xc50 activity_type *
    num_mc_samples = config.get("num_mc_samples", 100)
    # best_model, dataloaders, config, logger, _ = train_model_e2e(
    best_model, config, _, _ = train_model_e2e(
        config,
        model=BaselineDNN,
        model_type="mcdropout",
        logger=LOGGER,
    )
    dataloaders = get_dataloader(config, device=DEVICE, logger=LOGGER)
    # aleatoric = config.get("aleatoric", False)

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


def run_mcdropout_wrapper(**kwargs):
    global LOGGER
    LOGGER = create_logger(name="mcdropout", file_level="debug", stream_level="info")
    config = get_model_config(model_type="mcdropout", **kwargs)
    return run_mcdropout(config)


def run_mcdropout_hyperparm(**kwargs):
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


if __name__ == "__main__":
    run_mcdropout_wrapper(
        data_name="papyrus",
        activity_type="xc50",
        n_targets=-1,
        descriptor_protein="ankh-large",
        descriptor_chemical="ecfp2048",
        median_scaling=False,
        split_type="random",
        ext="pkl",
        task_type="regression",
        wandb_project_name=f"mcdp-test",
        epochs=5,
        num_mc_samples=5,
    )
# #
# Here we need to selectively set batchnorm layers to eval mode
# for m in model.modules():
#     if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
#         m.eval()
# network.train()
# # put bachnorm in eval mode
# for m in network.modules():
#     if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
#         m.eval()
# with torch.no_grad():
# def main():
#     parser = argparse.ArgumentParser(description="Run MC Dropout Model")
#     parser.add_argument(
#         "--num_mc_samples",
#         type=int,
#         default=100,
#         help="Number of MC dropout samples",
#     )
#     parser.add_argument(
#         "--data_name",
#         type=str,
#         default="papyrus",
#         choices=["papyrus", "tdc", "other"],
#         help="Data name argument",
#     )
#     parser.add_argument(
#         "--activity_type",
#         type=str,
#         default="xc50",
#         choices=["xc50", "kx"],
#         help="Activity argument",
#     )
#     parser.add_argument(
#         "--n_targets",
#         type=int,
#         default=-1,
#         help="Number of targets argument (default=-1 for all targets)",
#     )
#     parser.add_argument(
#         "--descriptor_protein",
#         type=str,
#         default=None,
#         choices=[
#             None,
#             "ankh-base",
#             "ankh-large",
#             "unirep",
#             "protbert",
#             "protbert_bfd",
#             "esm1_t34",
#             "esm1_t12",
#             "esm1_t6",
#             "esm1b",
#             "esm_msa1",
#             "esm_msa1b",
#             "esm1v",
#         ],
#         help="Protein descriptor argument",
#     )
#     parser.add_argument(
#         "--descriptor_chemical",
#         type=str,
#         default="ecfp2048",
#         choices=[
#             "ecfp1024",
#             "ecfp2048",
#             "mold2",
#             "mordred",
#             "cddd",
#             "fingerprint",  # "moldesc"
#         ],
#         help="Chemical descriptor argument",
#     )
#     parser.add_argument(
#         "--median_scaling",
#         action="store_true",
#         help="Use median scaling",
#     )
#     parser.add_argument(
#         "--split_type",
#         type=str,
#         default="random",
#         choices=["random", "scaffold", "time", "scaffold_cluster"],
#         help="Split argument",
#     )
#     parser.add_argument(
#         "--ext",
#         type=str,
#         default="pkl",
#         choices=["pkl", "parquet", "csv", "feather"],
#         help="File extension argument",
#     )
#     parser.add_argument(
#         "--task_type",
#         type=str,
#         default="regression",
#         choices=["regression", "classification"],
#         help="Task type argument",
#     )
#     parser.add_argument(
#         "--wandb-project-name",
#         type=str,
#         default="ensemble-test",
#         help="Wandb project name argument",
#     )
#     parser.add_argument(
#         "--sweep-count",
#         type=int,
#         default=None,
#         help="Sweep count argument",
#     )
#     # take chem layers as list input
#     parser.add_argument(
#         "--chem_layers",
#         type=parse_list,
#         default=None,
#         help="Chem layers sizes",
#     )
#     parser.add_argument(
#         "--prot_layers", type=parse_list, default=None, help="Prot layers sizes"
#     )
#     parser.add_argument(
#         "--regressor_layers",
#         type=parse_list,
#         default=None,
#         help="Regressor layers sizes",
#     )
#     parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
#     parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
#     parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
#     parser.add_argument(
#         "--early_stop", type=int, default=None, help="Early stopping patience"
#     )
#     parser.add_argument("--loss", type=str, default=None, help="Loss function")
#     parser.add_argument(
#         "--loss_reduction", type=str, default=None, help="Loss reduction method"
#     )
#     parser.add_argument("--optimizer", type=str, default=None, help="Optimizer")
#     parser.add_argument("--lr", type=float, default=None, help="Learning rate")
#     parser.add_argument(
#         "--weight_decay", type=float, default=None, help="Weight decay rate"
#     )
#     parser.add_argument(
#         "--lr_scheduler", type=str, default=None, help="LR scheduler type"
#     )
#     parser.add_argument(
#         "--lr_scheduler_patience", type=int, default=None, help="LR scheduler patience"
#     )
#     parser.add_argument(
#         "--lr_scheduler_factor", type=float, default=None, help="LR scheduler factor"
#     )
#
#     args = parser.parse_args()
#     # Construct kwargs, excluding arguments that were not provided
#     kwargs = {k: v for k, v in vars(args).items() if v is not None}
#
#     sweep_count = args.sweep_count
#     if sweep_count is not None and sweep_count > 0:
#         run_mcdropout_hyperparm(
#             **kwargs,
#         )
#     else:
#         run_mcdropout_wrapper(
#             **kwargs,
#         )
#
#

#
# def run_mcdropout(
#     datasets=None,
#     config=os.path.join(CONFIG_DIR, "baseline", "baseline_xc50_random_best.json"),
#     activity="xc50",
#     split="random",
#     wandb_project_name="multitask-learning-mcdropout",
#     num_samples=100,
#     seed=42,
#     **kwargs,
# ):
#     # load the config
#     config = get_model_config(
#         config=config, activity=activity, split=split, num_samples=num_samples, **kwargs
#     )
#     # Load the dataset
#     if datasets is None:
#         datasets = get_datasets(activity=activity, split=split)
#
#     # Get tasks names:
#     tasks = get_tasks(activity=activity, split=split)
#
#     # Initialize wandb
#     with wandb.init(
#         dir=LOG_DIR,
#         mode=wandb_mode,
#         project=wandb_project_name,
#         config=config,
#         name=f"{today}_mcdropout_{activity}_{split}",
#     ):
#         config = wandb.config
#
#         # Initialize the table to store the metrics
#         uct_metrics_logger = UCTMetricsTable(model_type="mcdropout", config=config)
#
#         # Define the data loaders
#         train_loader, val_loader, test_loader = build_loader(
#             datasets, config.batch_size, config.input_dim
#         )
#
#         # Train the baseline model
#         set_seed(seed)
#         # Train the model
#         # TODO : Add the option to load a pretrained model and train it further or just evaluate it
#         best_model, loss_fn = train_model(
#             train_loader, val_loader, config=config, seed=seed
#         )
#
#         # Perform MC Dropout during predictions
#         preds, targets = predict(
#             best_model.to(device),
#             test_loader,
#             num_samples=config.num_samples,
#             return_targets=True,
#         )
#
#         # Process the predictions
#         y_pred, y_std, y_true = process_preds(preds, targets, None)
#
#         # Calculate and log the metrics
#         metrics = uct_metrics_logger(
#             y_pred=y_pred, y_std=y_std, y_true=y_true, task_name="All 20 Targets"
#         )
#
#         for task_idx in range(len(tasks)):
#             task_y_pred, task_y_std, task_y_true = process_preds(
#                 preds, targets, task_idx=task_idx
#             )
#
#             task_name = tasks[task_idx]
#             metrics = uct_metrics_logger(
#                 y_pred=task_y_pred,  # y_pred[:, task_idx],
#                 y_std=task_y_std,  # y_std[:, task_idx],
#                 y_true=task_y_true,  # targets[:, task_idx],
#                 task_name=task_name,
#             )
#
#         uct_metrics_logger.wandb_log()
#
#
# if __name__ == "__main__":
#     run_mcdropout(wandb_project_name="mtl-mcdropout-test")

# def _mc_predict(model, test_loader, aleatoric=False, num_mc_samples=100, device=DEVICE):
#     model.train()  # Enable dropout
#     outputs_all = []
#     targets_all = []
#     aleatoric_all = []
#     # Here we need to selectively set batchnorm layers to eval mode
#     for m in model.modules():
#         if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
#             m.eval()
#     # network.train()
#     # # put bachnorm in eval mode
#     # for m in network.modules():
#     #     if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
#     #         m.eval()
#     with torch.no_grad():
#         for inputs, targets in tqdm(
#             test_loader, total=len(test_loader), desc="MC prediction"
#         ):
#             inputs = tuple(x.to(device) for x in inputs)
#             output_samples, alea_samples = [], []
#             for _ in range(num_mc_samples):  # Multiple forward passes
#                 if aleatoric:
#                     outputs, vars_ = model(inputs)
#                     # vars_ = torch.exp(logvars)
#                     output_samples.append(outputs)
#                     alea_samples.append(vars_)
#                 else:
#                     outputs = model(inputs)
#                     output_samples.append(outputs)
#
#             outputs = torch.stack(output_samples, dim=2)
#             if aleatoric:
#                 vars_ = torch.stack(alea_samples, dim=2)
#                 aleatoric_all.append(vars_)
#             outputs_all.append(outputs)
#             targets_all.append(targets)
#
#     model.eval()  # Disable dropout
#     outputs_all = torch.cat(outputs_all, dim=0).cpu()
#     targets_all = torch.cat(targets_all, dim=0).cpu()
#
#     if aleatoric:
#         aleatoric_all = torch.cat(aleatoric_all, dim=0).cpu()
#         return outputs_all, targets_all, aleatoric_all
#
#     return outputs_all, targets_all, None

#
#
#
#
# import os
# import sys
#
# sys.path.append(
#     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# )
# from datetime import date
# import torch
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from uqdd.models.models_utils import set_seed, get_model_config, get_datasets, get_tasks
# from uqdd.models.models_utils import (
#     build_loader,
#     build_optimizer,
#     MultiTaskLoss,
#     save_models,
# )
# from uqdd.models.models_utils import UCTMetricsTable, process_preds
# from uqdd.models.baseline import train_model
#
# from functools import partial
# import numpy as np
# import torch.nn as nn
# import wandb
# from torch.optim.lr_scheduler import ReduceLROnPlateau
#
# today = date.today()
# today = today.strftime("%Y%m%d")
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device: " + str(device))
# print(torch.version.cuda) if device == "cuda" else None
#
# LOG_DIR = os.environ.get("LOG_DIR")
# DATA_DIR = os.environ.get("DATA_DIR")
# DATASET_DIR = os.path.join(DATA_DIR, "dataset")
# CONFIG_DIR = os.environ.get("CONFIG_DIR")
# FIGS_DIR = os.environ.get("FIGS_DIR")
#
# # wandb_dir = '../logs/'
# wandb_mode = "online"  # 'offline')))))


# def mc_predict(
#     model,
#     test_loader,
#     aleatoric=False,
#     num_samples=100,
#     device=DEVICE
# ):
#     model.train()  # Enable dropout
#     outputs_all = []
#     targets_all = []
#     vars_all = []
#
#     with torch.no_grad():
#         for inputs, targets in tqdm(test_loader, total=len(test_loader), desc="MC prediction"):
#             inputs = tuple(x.to(device) for x in inputs)
#             output_samples, vars_samples = [], []
#             for _ in range(num_samples): # Multiple forward passes
#                 if aleatoric:
#                     outputs, logvars = model(inputs)
#                     vars_ = torch.exp(logvars)
#                     output_samples.append(outputs)
#                     vars_samples.append(vars_)
#                 else:
#                     outputs = model(inputs)
#                     output_samples.append(outputs)
#
#             outputs = torch.stack(output_samples, dim=2)
#             if aleatoric:
#                 vars_ = torch.stack(vars_samples, dim=2)
#                 vars_all.append(vars_)
#             outputs_all.append(outputs)
#             targets_all.append(targets)
#
#     model.eval()  # Disable dropout
#     outputs_all = torch.cat(outputs_all, dim=0).cpu()
#     targets_all = torch.cat(targets_all, dim=0).cpu()
#
#     if aleatoric:
#         vars_all = torch.cat(vars_all, dim=0).cpu()
#         return outputs_all, targets_all, vars_all
#
#     return outputs_all, targets_all, None
# if return_targets:
#     targets_all = torch.cat(targets_all, dim=0)
#     return outputs_all, targets_all
# return outputs_all


# def mc_uncertainty_estimate(outputs):
#     outputs = outputs.cpu().detach()
#     y_mean = outputs.mean(dim=2).numpy()
#     y_std = outputs.std(dim=2).numpy()
#     # y_var = outputs.var(dim=0)
#     # y_std = torch.sqrt(y_var)
#
#     return y_mean, y_std # , y_var


# def plot_predictions(y_true, y_pred, y_std):
#     plt.figure(figsize=(12, 6))
#     plt.errorbar(y_true, y_pred, yerr=y_std, fmt="o")
#     plt.xlabel("True values")
#     plt.ylabel("Predicted values")
#     plt.grid()
#     plt.show()
#
#
# def plot_uncertainty_distribution(y_std):
#     plt.figure(figsize=(12, 6))
#     plt.hist(y_std, bins=50)
#     plt.xlabel("Uncertainty")
#     plt.ylabel("Frequency")
#     plt.grid()
#     plt.show()

#
#         for inputs, targets in tqdm(
#             test_loader, total=len(test_loader), desc="MC prediction"
#         ):
#
#             inputs = tuple(x.to(device) for x in inputs)
#         output_samples, alea_samples = [], []
#         for _ in range(num_mc_samples):  # Multiple forward passes
#             if aleatoric:
#                 outputs, vars_ = model(inputs)
#                 # vars_ = torch.exp(logvars)
#                 output_samples.append(outputs)
#                 alea_samples.append(vars_)
#             else:
#                 outputs = model(inputs)
#                 output_samples.append(outputs)
#
#         outputs = torch.stack(output_samples, dim=2)
#         if aleatoric:
#             vars_ = torch.stack(alea_samples, dim=2)
#             aleatoric_all.append(vars_)
#         outputs_all.append(outputs)
#         targets_all.append(targets)
#
# model.eval()  # Disable dropout
# outputs_all = torch.cat(outputs_all, dim=0).cpu()
# targets_all = torch.cat(targets_all, dim=0).cpu()
#
# if aleatoric:
#     aleatoric_all = torch.cat(aleatoric_all, dim=0).cpu()
#     return outputs_all, targets_all, aleatoric_all
#
# return outputs_all, targets_all, None
