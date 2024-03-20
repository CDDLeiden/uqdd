# get today's date as yyyy/mm/dd format
import argparse
from datetime import datetime
from functools import partial
import wandb
import torch
import torch.nn as nn
from uqdd import TODAY, DEVICE, WANDB_MODE, WANDB_DIR
from uqdd.data.utils_data import get_tasks
from uqdd.models.baseline import BaselineDNN
from uqdd.utils import create_logger

from uqdd.models.utils_train import run_model, premodel_init, predict
from uqdd.models.utils_metrics import MetricsTable, process_preds, make_df_preds
from typing import Union

from uqdd.models.utils_models import (
    get_model_config,
    set_seed,
    save_model,
)


class EnsembleDNN(nn.Module):
    def __init__(
        self, config=None, model_class=BaselineDNN, ensemble_size=100, **kwargs
    ):
        super(EnsembleDNN, self).__init__()
        self.ensemble_size = ensemble_size
        self.logger = create_logger(name="EnsembleDNN")
        if config is None:
            config = get_model_config(model_name="ensemble", **kwargs)
        # set_seed(seed)
        self.models = nn.ModuleList(
            [model_class(config, **kwargs) for _ in range(ensemble_size)]
        )

    def forward(self, inputs):
        outputs = torch.stack([model(inputs) for model in self.models], dim=2)
        return outputs


def run_ensemble(
    config=None,
    ensemble_size=100,
    data_name="papyrus",
    activity_type="xc50",
    n_targets=-1,
    descriptor_protein=None,
    descriptor_chemical=None,
    median_scaling=False,
    split_type="random",
    ext="pkl",
    task_type="regression",
    wandb_project_name=f"{TODAY}-ensemble",
    logger=None,
    **kwargs,
):
    (
        dataloaders,
        config,
        logger,
        desc_prot_len,
        desc_chem_len,
        start_time,
        data_specific_path,
    ) = premodel_init(
        config,
        "ensemble",
        data_name,
        activity_type,
        n_targets,
        descriptor_protein,
        descriptor_chemical,
        split_type,
        median_scaling,
        task_type,
        ext,
        logger,
        **kwargs,
    )

    m_tag = "median_scaling" if median_scaling else "no_median_scaling"
    mt = n_targets > 1
    mt_tag = "MT" if mt else "ST"
    wandb_tags = [
        "ensemble",
        data_name,
        activity_type,
        descriptor_protein,
        descriptor_chemical,
        split_type,
        task_type,
        m_tag,
        mt_tag,
    ]
    with wandb.init(
        dir=WANDB_DIR,
        mode=WANDB_MODE,
        project=wandb_project_name,
        config=config,
        tags=wandb_tags,
    ):
        config = wandb.config

        # Define the ensemble models
        ensemble_model = EnsembleDNN(
            config=config,
            model_class=BaselineDNN,
            ensemble_size=ensemble_size,
            chem_input_dim=desc_chem_len,
            prot_input_dim=desc_prot_len,
            task_type=task_type,
            n_targets=n_targets,
            logger=logger,
        ).to(DEVICE)

        # Train the ensemble model
        best_model, test_loss = run_model(
            config,
            ensemble_model,
            dataloaders,
            n_targets=n_targets,
            device=DEVICE,
            logger=logger,
        )
        model_name = (
            f"{TODAY}-ensemble_{split_type}_{descriptor_protein}_{descriptor_chemical}"
        )
        # Save the best model
        save_model(
            config,
            best_model,
            model_name,
            data_specific_path,
            desc_prot_len,
            desc_chem_len,
            onnx=True,
        )
        # Predictions on Test set
        # Initialize the table to store the metrics
        config.activity = activity_type
        config.split = split_type
        uct_metrics_logger = MetricsTable(
            model_type="ensemble",
            config=config,
            desc_prot=descriptor_protein,
            desc_chem=descriptor_chemical,
            multitask=mt,
            task_type=task_type,
            data_specific_path=data_specific_path,
            model_name=model_name,
            logger=logger,
        )
        ensemble_preds, targets = predict(
            best_model, dataloaders["test"], return_targets=True
        )

        if mt:
            tasks = get_tasks(
                data_name=data_name, activity=activity_type, n_targets=n_targets
            )

            y_true, y_pred, y_std, y_err = process_preds(ensemble_preds, targets, None)
            df = make_df_preds(
                y_true,
                y_pred,
                y_std,
                y_err,
                True,
                data_specific_path,
                model_name + "_MT_AllTargets",
            )
            logger.debug(
                f"Ensemble - predictions saved to Dataframe with shape {df.shape}"
            )
            metrics, plots = uct_metrics_logger(
                y_pred=y_pred,
                y_std=y_std,
                y_true=y_true,
                y_err=y_err,
                task_name=f"All {n_targets} Targets",
            )

            for task_idx in range(len(tasks)):
                task_y_true, task_y_pred, task_y_std, task_y_err = process_preds(
                    ensemble_preds, targets, task_idx=task_idx
                )
                # Calculate and log the metrics
                task_name = tasks[task_idx]
                taskmetrics, taskplots = uct_metrics_logger(
                    y_pred=task_y_pred,
                    y_std=task_y_std,
                    y_true=task_y_true,
                    y_err=task_y_err,
                    task_name=task_name,
                )
                metrics[taskmetrics] = taskmetrics
                plots[taskplots] = taskplots

        else:  # ST
            task_name = f"PCM {task_type}"
            # Process the predictions
            y_true, y_pred, y_std, y_err = process_preds(ensemble_preds, targets, None)
            df = make_df_preds(
                y_true, y_pred, y_std, y_err, True, data_specific_path, model_name
            )
            logger.debug(
                f"Ensemble - predictions saved to Dataframe with shape {df.shape}"
            )

            # Calculate and log the metrics
            metrics, plots = uct_metrics_logger(
                y_pred=y_pred,
                y_std=y_std,
                y_true=y_true,
                y_err=y_err,
                task_name=task_name,
            )

        uct_metrics_logger.wandb_log()

    logger.info(f"Ensemble - end time: {datetime.now()}")
    logger.info(f"Ensemble - duration: {datetime.now() - start_time}")
    return test_loss, ensemble_preds, metrics, plots


def main():
    parser = argparse.ArgumentParser(description="Run Ensemble Model")

    parser.add_argument(
        "--data_name",
        type=str,
        default="papyrus",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--activity_type",
        type=str,
        default="xc50",
        help="Type of activity",
    )
    parser.add_argument(
        "--n_targets",
        type=int,
        default=-1,
        help="Number of targets",
    )
    parser.add_argument(
        "--descriptor_protein",
        type=str,
        default=None,
        help="Type of protein descriptor",
    )
    parser.add_argument(
        "--descriptor_chemical",
        type=str,
        default=None,
        help="Type of chemical descriptor",
    )
    parser.add_argument(
        "--median_scaling",
        action="store_true",
        help="Use median scaling",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="random",
        help="Type of split",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="pkl",
        help="Extension of the dataset",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="regression",
        help="Type of task",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=100,
        help="Size of the ensemble",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=f"{TODAY}-ensemble",
        help="Name of the wandb project",
    )

    args = parser.parse_args()
    run_ensemble(**vars(args))


if __name__ == "__main__":
    main()

    # TEST
    # run_ensemble(
    #     data_name="papyrus",
    #     activity_type="xc50",
    #     n_targets=-1,
    #     descriptor_protein="ankh-base",
    #     descriptor_chemical="ecfp2048",
    #     median_scaling=False,
    #     split_type="random",
    #     ext="pkl",
    #     task_type="regression",
    #     wandb_project_name=f"{TODAY}-ensemble-test",
    #     ensemble_size=5,
    # )

# def build_ensemble(config=wandb.config):
#     ensemble_models = []
#     try:
#         seed = config.seed
#     except AttributeError:
#         seed = 42
#     # deterministic cuda algorithms
#     torch.backends.cudnn.deterministic = True
#
#     for _ in range(config.ensemble_size):
#         set_seed(seed)
#         model = BaselineDNN(
#             config.input_dim,
#             config.hidden_dim_1,
#             config.hidden_dim_2,
#             config.hidden_dim_3,
#             config.output_dim,
#             config.dropout,
#         )
#         ensemble_models.append(model)
#         seed += 1
#
#     return ensemble_models
#
#
# def train_model(
#     model_idx,
#     ensemble_models,
#     train_loader,
#     val_loader,
#     test_loader,
#     config,
#     **kwargs,
#     # config=None,
# ):
#     """
#     Train a single model from the ensemble.
#
#     Parameters
#     ----------
#     model_idx : int
#         Index of the model in the ensemble.
#     ensemble_models : list
#         List of models in the ensemble.
#     train_loader : torch.utils.data.DataLoader
#         Training data loader.
#     val_loader : torch.utils.data.DataLoader
#         Validation data loader.
#     test_loader : torch.utils.data.DataLoader
#         Test data loader.
#     config : wandb.config
#         Configuration object containing hyperparameters and settings.
#     **kwargs : dict, optional
#         Additional keyword arguments.
#
#     Returns
#     -------
#     best_model : torch.nn.Module
#         Best trained model based on validation loss.
#     """
#     # config = wandb.config if config is None else config
#
#     # Get the model for the given model_idx
#     model = ensemble_models[model_idx]
#     # Move the model to the device
#     model.to(device)
#
#     # Define the loss function
#     loss_fn = MultiTaskLoss(
#         loss_type=config.loss,
#         reduction="none",
#     )
#     # Define the optimizer with weight decay and learning rate scheduler
#     optimizer = build_optimizer(
#         model, config.optimizer, config.learning_rate, config.weight_decay
#     )
#     lr_scheduler = ReduceLROnPlateau(
#         optimizer,
#         mode="min",
#         factor=config.lr_factor,
#         patience=config.lr_patience,
#         verbose=True,
#     )
#
#     # Train the model
#     best_val_loss = float("inf")
#     early_stop_counter = 0
#     best_model = None
#     for epoch in tqdm(range(config.num_epochs), desc="Epochs"):
#         try:
#             epoch, train_loss, val_loss, val_rmse, val_r2, val_evs = run_epoch(
#                 model,
#                 train_loader,
#                 val_loader,
#                 loss_fn,
#                 optimizer,
#                 lr_scheduler,
#                 epoch=epoch,
#             )
#             wandb.log(
#                 data={
#                     f"epoch": epoch,
#                     f"model{model_idx}/train_loss": train_loss,
#                     f"model{model_idx}/val_loss": val_loss,
#                     f"model{model_idx}/val_rmse": val_rmse,
#                     f"model{model_idx}/val_r2": val_r2,
#                     f"model{model_idx}/val_evs": val_evs,
#                 },
#                 # step=epoch
#             )
#             # Early stopping
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 early_stop_counter = 0
#                 # Save the best model - dropped to avoid memory issues
#                 # Update the best model and its performance
#                 best_model = model
#
#             else:
#                 early_stop_counter += 1
#                 if early_stop_counter > config.early_stop:
#                     break
#
#         except Exception as e:
#             raise Exception(
#                 f"The following exception occurred inside the epoch loop {e}"
#             )
#
#         # Save the best model
#         # save_model(config, best_model, model_idx)
#
#     predictions = predict(best_model, test_loader, return_targets=False)
#
#     return best_model, predictions

#
# def run_ensemble(
#     datasets=None,
#     config: Union[str, dict] = "uqdd/config/ensemble/ensemble.json",
#     activity: str = "xc50",
#     split: str = "random",
#     wandb_project_name: str = "multitask-learning-ensemble",
#     ensemble_size: int = 100,
#     seed: int = 42,
#     **kwargs,
# ):
#     # Load the config
#     config = get_model_config(
#         config=config,
#         activity=activity,
#         split=split,
#         ensemble_size=ensemble_size,
#         **kwargs,
#     )
#     # Load the dataset
#     if datasets is None:
#         datasets = get_datasets(activity=activity, split=split)
#     # Get tasks names:
#     tasks = get_tasks(activity=activity, split=split)
#
#     # Initialize wandb for the ensemble models
#     with wandb.init(
#         dir=LOG_DIR,
#         mode=wandb_mode,
#         project=wandb_project_name,
#         config=config,
#         name=f"{today}_ensemble_{activity}_{split}",
#     ):
#         config = wandb.config
#
#         # Initialize the table to store the metrics
#         uct_metrics_logger = MetricsTable(model_type="ensemble", config=config)
#
#         # Define the data loaders
#         train_loader, val_loader, test_loader = build_loader(
#             datasets, config.batch_size, config.input_dim
#         )
#         # Define the ensemble models
#         config.seed = seed  # TODO FIX THIS
#         ensemble_models = build_ensemble(config=config)
#         # Initialize lists to store the results
#         best_models = []
#         predictions = []
#         # results = []
#         _, targets = predict(
#             ensemble_models[0].to(device), test_loader, return_targets=True
#         )
#         # Train the ensemble models
#         for model_idx in tqdm(range(len(ensemble_models)), desc="Ensemble models"):
#             # Train the model
#             best_model, preds = train_model(
#                 model_idx,
#                 ensemble_models,
#                 train_loader,
#                 val_loader,
#                 test_loader,
#                 config,
#             )
#             # Store the results of the model
#             predictions.append(preds)
#             best_models.append(best_model)
#
#         # Save Best Ensemble Models
#         best_models = nn.ModuleList(best_models)
#         save_model(config, best_models, f"{wandb.run.name}_ensemble_model", onnx=False)
#
#         # Ensemble the predictions
#         ensemble_preds = torch.stack(predictions, dim=2)
#         # Process ensemble predictions
#         y_pred, y_std, y_true = process_preds(ensemble_preds, targets, None)
#
#         # Calculate and log the metrics
#         # task_name =
#         metrics = uct_metrics_logger(
#             y_pred=y_pred, y_std=y_std, y_true=y_true, task_name="All 20 Targets"
#         )
#         for task_idx in range(len(tasks)):
#             task_y_pred, task_y_std, task_y_true = process_preds(
#                 ensemble_preds, targets, task_idx=task_idx
#             )
#             # Calculate and log the metrics
#             task_name = tasks[task_idx]
#             metrics = uct_metrics_logger(
#                 y_pred=task_y_pred,
#                 y_std=task_y_std,
#                 y_true=task_y_true,
#                 task_name=task_name,
#             )
#
#         uct_metrics_logger.wandb_log()
#
# #
# # if __name__ == "__main__":
# #     # datasets = get_datasets('xc50', 'random')
# #     test_config = {
# #         "activity": "xc50",
# #         "batch_size": 64,
# #         "dropout": 0.1,
# #         "early_stop": 100,
# #         "hidden_dim_1": 2048,
# #         "hidden_dim_2": 256,
# #         "hidden_dim_3": 256,
# #         "input_dim": 2048,
# #         "learning_rate": 0.01,
# #         "loss": "huber",
# #         "lr_scheduler": "ReduceLROnPlateau",  # TODO not necessary anymore
# #         "lr_factor": 0.5,
# #         "lr_patience": 20,
# #         "num_epochs": 3,
# #         "num_tasks": 20,
# #         "optimizer": "SGD",
# #         "output_dim": 20,
# #         "weight_decay": 0.001,
# #         "seed": 42,
# #         "split": "random",
# #         "ensemble_size": 3,
# #     }
# #
# #     # test_loss, test_predictions = \
# #     run_ensemble(
# #         config=test_config,  # os.path.join(CONFIG_DIR, 'ensemble/ensemble.json'),
# #         activity="xc50",
# #         split="random",
# #         ensemble_size=5,
# #         # ensemble_method='fusion',
# #         wandb_project_name="mtl-ensemble-test",
# #         seed=42,
# #     )
# #
# # print(test_loss)
# # print(test_predictions)
#
#
# # chem_input_dim=None,
# # prot_input_dim=None,
# # task_type="regression",
# # n_targets=-1,
# # seed = 42
# # set_seed(seed)
# #
# # logger = create_logger(name="ensemble", file_level="debug", stream_level="info")
# #
# # start_time = datetime.now()
# # logger.info(f"Ensemble - start time: {start_time}")
# #
# # # get datasets
# # datasets = build_datasets(
# #     data_name=data_name,
# #     n_targets=n_targets,
# #     activity_type=activity_type,
# #     split_type=split_type,
# #     desc_prot=descriptor_protein,
# #     desc_chem=descriptor_chemical,
# #     label_scaling_func=label_scaling_func,
# #     ext=ext,
# #     logger=logger,
# # )
# # desc_prot_len, desc_chem_len = get_desc_len_from_dataset(datasets["train"])
# # logger.info(f"Chemical descriptor {descriptor_chemical} of length: {desc_chem_len}")
# # logger.info(f"Protein descriptor {descriptor_protein} of length: {desc_prot_len}")
