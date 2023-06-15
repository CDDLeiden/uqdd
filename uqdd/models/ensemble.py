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
import sys;

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import wandb
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from uqdd.models.models_utils import set_seed, get_config, get_datasets, get_tasks
from uqdd.models.models_utils import build_loader, build_optimizer, MultiTaskLoss, save_models
from uqdd.models.models_utils import calc_regr_metrics, make_true_vs_preds_plot, make_uct_plots

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


def run_ensemble(
        datasets=None,
        config='uqdd/config/ensemble/ensemble.json',
        activity='xc50',
        split='random',
        wandb_project_name='multitask-learning-ensemble',
        ensemble_size=100,
        # ensemble_method='fusion',
        seed=42,
        **kwargs
        # optimizer,
        # loss_fn,
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
            # group=f'ensemble_model',
    ):
        # metrics_table = wandb.Table(columns=["Model", "Epoch", "Train Loss", "Val Loss", "Val RMSE", "Val R2", "Val EVS"])

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
            # with wandb.init(
            #         dir=LOG_DIR,
            #         mode=wandb_mode,
            #         project=wandb_project_name,
            #         config=config,
            #         name=f"ensemble_{activity}_{split}",
            #         group=f'ensemble_model{model_idx}'
            # ):

            model = ensemble_models[model_idx]
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

            # wandb.run.group #2.group = f'model{model_idx}'
            # wandb.run.run_group = f'model{model_idx}'
            # wandb.run.name = f"ensemble_{activity}_{split}"
            # wandb.run.name = f"model{model_idx}"
            for epoch in tqdm(range(config.num_epochs), desc='Epochs'):
                try:
                    epoch, train_loss, val_loss, val_rmse, val_r2, val_evs = run_epoch(
                        model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler, epoch=epoch
                    )
                    # metrics_table.add_data(f"Model {model_idx}", epoch, train_loss, val_loss, val_rmse, val_r2, val_evs)

                    # Log the metrics
                    # wandb.log(
                    #     data={
                    #         f'epoch/model{model_idx}': epoch,
                    #         f'train/loss/model{model_idx}': train_loss,
                    #         f'val/loss/model{model_idx}': val_loss,
                    #         f'val/rmse/model{model_idx}': val_rmse,
                    #         f'val/r2/model{model_idx}': val_r2,
                    #         f'val/evs/model{model_idx}': val_evs,
                    #     },
                    #     step=epoch
                    # )
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
                    # wandb.log(
                    #     data={
                    #         'modelx': model_idx,
                    #         'epochx': epoch,
                    #         'trainx/loss': train_loss,
                    #         'valx/loss': val_loss,
                    #         'valx/rmse': val_rmse,
                    #         'valx/r2': val_r2,
                    #         'valx/evs': val_evs,
                    #     },
                    #     step=epoch,
                    # )
                    # wandb.log(
                    #     data={
                    #         f'model_{model_idx}': {
                    #             'epoch': epoch,
                    #             'train/loss': train_loss,
                    #             'val/loss': val_loss,
                    #             'val/rmse': val_rmse,
                    #             'val/r2': val_r2,
                    #         }},
                    #     # step=epoch
                    # )
                    # Create the plots for each value of interest
                    # if epoch == config.num_epochs - 1:
                    #     wandb.log(
                    #         data={
                    #             f'train/loss': wandb.plot.line_series(
                    #                 x='epoch',
                    #                 y=f'train/loss',
                    #                 title='Train Loss',
                    #                 xlabel='Epoch',
                    #                 ylabel='Loss',
                    #                 series=f'model_index',
                    #                 series_name=f'model{model_idx}'
                    #             ),
                    #             f'val/loss': wandb.plot.line(
                    #                 x='epoch',
                    #                 y=f'val/loss',
                    #                 title='Validation Loss',
                    #                 xlabel='Epoch',
                    #                 ylabel='Loss',
                    #                 series=f'model_index',
                    #                 series_name=f'model{model_idx}'
                    #             ),
                    #             f'val/rmse': wandb.plot.line(
                    #                 x='epoch',
                    #                 y=f'val/rmse',
                    #                 title='Validation RMSE',
                    #                 xlabel='Epoch',
                    #                 ylabel='RMSE',
                    #                 series=f'model_index',
                    #                 series_name=f'model{model_idx}'
                    #             ),
                    #             f'val/r2': wandb.plot.line(
                    #                 x='epoch',
                    #                 y=f'val/r2',
                    #                 title='Validation R^2 Score',
                    #                 xlabel='Epoch',
                    #                 ylabel='R^2 Score',
                    #                 series=f'model_index',
                    #                 series_name=f'model{model_idx}'
                    #             ),
                    #         }
                    #     )
                    # wandb.log(
                    #     data={
                    #         f'epoch/model{model_idx}': epoch,
                    #         f'train/loss/model{model_idx}': train_loss,
                    #         f'val/loss/model{model_idx}': val_loss,
                    #         f'val/rmse/model{model_idx}': val_rmse,
                    #         f'val/r2/model{model_idx}': val_r2,
                    #     },
                    #     step=epoch,
                    # )
                    # wandb.log(
                    #     data={
                    #         f'model{model_idx}/epoch': epoch,
                    #         f'model{model_idx}/train/loss': train_loss,
                    #         f'model{model_idx}/val/loss': val_loss,
                    #         f'model{model_idx}/val/rmse': val_rmse,
                    #         f'model{model_idx}/val/r2': val_r2,
                    #     }
                    # )
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

            # Test the model
            if model_idx == 0:
                model_y_preds, targets = predict(best_model, test_loader, return_targets=True)
            else:
                model_y_preds = predict(best_model, test_loader, return_targets=False)

            predictions.append(model_y_preds)  # (12172, 20)
            best_models.append(best_model)

        # plot from wandb table models as series, epochs as x, and train_loss as y
        # Create the plots using line_series
        # wandb.log(
        #     data={
        #         "train/loss": wandb.plot.line_series(
        #             xs=epochs,
        #             ys=train_losses,
        #             keys=models,
        #             title="Train Loss",
        #             xname="Epoch"
        #         ),
        #         "val/loss": wandb.plot.line_series(
        #             xs=epochs,
        #             ys=val_losses,
        #             keys=models,
        #             title="Validation Loss",
        #             xname="Epoch"
        #         ),
        #         "val/rmse": wandb.plot.line_series(
        #             xs=epochs,
        #             ys=val_rmses,
        #             keys=models,
        #             title="Validation RMSE",
        #             xname="Epoch"
        #         ),
        #         "val/r2": wandb.plot.line_series(
        #             xs=epochs,
        #             ys=val_r2s,
        #             keys=models,
        #             title="Validation R2 Score",
        #             xname="Epoch"
        #         ),
        #         "val/evs": wandb.plot.line_series(
        #             xs=epochs,
        #             ys=val_evs,
        #             keys=models,
        #             title="Validation EVS",
        #             xname="Epoch"
        #         )
        #     }
        # )

        # Ensemble the predictions
        ensemble_predictions = torch.stack(predictions, dim=2)  # torch.Size([datapoints, tasks, ensemble_size])
        # Ensemble the predictions
        ensemble_predictions_mu = ensemble_predictions.mean(dim=2)  # torch.Size([datapoints, tasks])
        ensemble_predictions_std = ensemble_predictions.std(dim=2)  # torch.Size([datapoints, tasks])
        # over all tasks metrics
        # Reshape the tensor to stack the tasks on top of each other
        alltasks_preds_mu = torch.flatten(ensemble_predictions_mu.transpose(0, 1))
            # ensemble_predictions_mu.reshape(-1, ensemble_predictions_mu.size(
            # 2))  # torch.Size([datapoints*tasks, features])
        alltasks_preds_std = torch.flatten(ensemble_predictions_std.transpose(0, 1))
        alltasks_targets = torch.flatten(targets.transpose(0, 1))
            # targets.repeat(1, ensemble_predictions_mu.size(1)).reshape(-1, targets.size(
            # 1))  # torch.Size([datapoints*tasks, features])
        # Calculate the metrics
        alltasks_nanmask = ~torch.isnan(alltasks_targets)
        alltasks_y_true = alltasks_targets[alltasks_nanmask]
        alltasks_y_preds = alltasks_preds_mu[alltasks_nanmask]
        alltasks_y_std = alltasks_preds_std[alltasks_nanmask]
        n_subset = 500 if len(alltasks_y_true) > 500 else len(alltasks_y_true)
        task_name = "All 20 Targets"

        alltasks_y_preds = alltasks_y_preds.cpu().numpy()
        alltasks_y_std = alltasks_y_std.cpu().numpy()
        alltasks_y_true = alltasks_y_true.cpu().numpy()

        # Calculate the metrics
        alltasks_metrics = get_all_metrics(
            y_pred=alltasks_y_preds,  # : np.ndarray,
            y_std=alltasks_y_std,  # : np.ndarray,
            y_true=alltasks_y_true,  # : np.ndarray,
            num_bins=100,
            resolution=99,
            scaled=True,
            verbose=False,
        )

        # Plot the metrics
        figures_path = os.path.join(FIGS_DIR, 'ensemble', activity, split)
        os.makedirs(figures_path, exist_ok=True)

        metrics_filepath = os.path.join(figures_path, f'{task_name}_metrics.pkl')
        with open(metrics_filepath, 'wb') as file:
            pickle.dump(alltasks_metrics, file)

        fig = make_uct_plots(
            alltasks_y_preds,
            alltasks_y_std,
            alltasks_y_true,
            task_name=task_name,
            n_subset=n_subset,  # 100,
            ylims=None,  # (-3, 3),
            num_stds_confidence_bound=1.96,
            plot_save_str=os.path.join(figures_path, f'{task_name}_uct'),
            savefig=True,
        )
        img = wandb.Image(fig)

        uct_metrics_table.add_data(
            task_name,
            config.activity,
            config.split,
            alltasks_metrics["accuracy"]["rmse"],
            alltasks_metrics["accuracy"]["r2"],
            alltasks_metrics["accuracy"]["mae"],
            alltasks_metrics["accuracy"]["mdae"],
            alltasks_metrics["accuracy"]["marpd"],
            alltasks_metrics["accuracy"]["corr"],
            alltasks_metrics["avg_calibration"]["rms_cal"],
            alltasks_metrics["avg_calibration"]["ma_cal"],
            alltasks_metrics["avg_calibration"]["miscal_area"],
            alltasks_metrics["sharpness"]["sharp"],
            alltasks_metrics["scoring_rule"]["nll"],
            alltasks_metrics["scoring_rule"]["crps"],
            alltasks_metrics["scoring_rule"]["check"],
            alltasks_metrics["scoring_rule"]["interval"],
            img
        )

        for task_idx in range(len(tasks)):
            task_y_true = targets[:, task_idx]  # (datapoints,)
            # nan mask from the targets
            nan_mask = ~torch.isnan(task_y_true)
            task_y_true = task_y_true[nan_mask]  # (datapoints,)

            task_y_preds = ensemble_predictions_mu[:, task_idx][nan_mask]  # (datapoints,)
            task_y_std = ensemble_predictions_std[:, task_idx][nan_mask]  # (datapoints,)
            task_name = tasks[task_idx]
            n_subset = 500 if len(task_y_true) > 500 else len(task_y_true)

            task_y_preds = task_y_preds.cpu().numpy()
            task_y_std = task_y_std.cpu().numpy()
            task_y_true = task_y_true.cpu().numpy()

            # Calculate the metrics
            task_metrics = get_all_metrics(
                y_pred=task_y_preds,  # : np.ndarray,
                y_std=task_y_std,  # : np.ndarray,
                y_true=task_y_true,  # : np.ndarray,
                num_bins=100,
                resolution=99,
                scaled=True,
                verbose=False,
            )

            metrics_filepath = os.path.join(figures_path, f'{task_name}_metrics.pkl')
            with open(metrics_filepath, 'wb') as file:
                pickle.dump(task_metrics, file)

            fig = make_uct_plots(
                task_y_preds,
                task_y_std,
                task_y_true,
                task_name=task_name,
                n_subset=n_subset,  # 100,
                ylims=None,  # (-3, 3),
                num_stds_confidence_bound=1.96,
                plot_save_str=os.path.join(figures_path, f'{task_name}_uct'),
                savefig=True,
            )
            img = wandb.Image(fig)

            uct_metrics_table.add_data(
                task_name,
                config.activity,
                config.split,
                task_metrics["accuracy"]["rmse"],
                task_metrics["accuracy"]["r2"],
                task_metrics["accuracy"]["mae"],
                task_metrics["accuracy"]["mdae"],
                task_metrics["accuracy"]["marpd"],
                task_metrics["accuracy"]["corr"],
                task_metrics["avg_calibration"]["rms_cal"],
                task_metrics["avg_calibration"]["ma_cal"],
                task_metrics["avg_calibration"]["miscal_area"],
                task_metrics["sharpness"]["sharp"],
                task_metrics["scoring_rule"]["nll"],
                task_metrics["scoring_rule"]["crps"],
                task_metrics["scoring_rule"]["check"],
                task_metrics["scoring_rule"]["interval"],
                img
            )

        wandb.log(
            data={
                f'uct_metrics': uct_metrics_table,
            }
        )


############## UCT PLOTS #####################

# return test_loss, test_predictions

# for i in tqdm(range(config.ensemble_size), desc='Ensemble models'):
#     # Get the model
#     model = ensemble_models[i]
#     model.to(device)
#
#     # Define the loss function
#     loss_fn = build_loss(config.loss, reduction='none')
#     # Define the optimizer with weight decay and learning rate scheduler
#     optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
#
# # Define the learning rate scheduler
# scheduler = ReduceLROnPlateau(
#     optimizer,
#     mode='min',
#     factor=config.lr_factor,
#     patience=config.lr_patience,
#     verbose=True
# )
#
# # Train the model
# best_val_loss = float('inf')
# early_stop_counter = 0
# for epoch in tqdm(range(config.num_epochs + 1)):
#
#     if epoch == 0:
#         # epoch_0_eval(model, train_loader, val_loader, loss_fn,)
#         continue
#
#     # Training
#     train_loss = train(model, train_loader, optimizer, loss_fn)
#     # Validation
#     val_loss, val_rmse, val_r2, val_evs =  evaluate(model, val_loader, loss_fn)
#     # Log the metrics
#     wandb.log(
#         data={
#             'epoch': epoch,
#             'train_loss': train_loss,
#             'val_loss': val_loss,
#             'val_rmse': val_rmse,
#             'val_r2': val_r2,
#             'val_evs': val_evs
#         }
#     )
#
#     # Update the learning rate
#
#
#
# # Train the model
# train_loss = train(
#     model,
#     train_loader,
#
#     return_model=True
# )


# def train_ensemble(
#         ensemble_models,
#         train_loader,
#         optimizer,
#         lr_scheduler,
#         loss_fn,
# ):
#     for i in tqdm(range(len(ensemble_models)), desc='Ensemble models'):
#         ensemble_models[i].to(device)
#
#         for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = ensemble_models[i](inputs)
#             loss = loss_fn(outputs, targets)
#             loss.backward()
#             optimizer.step()


# def train_ensemble(
#         train_loader, val_loader, test_loader,
#         input_size, hidden_size1, hidden_size2, hidden_size3,
#         output_size, num_epochs=3000):
#     ensemble = []
#     learning_rate = 0.005
#     learning_rate_decay = 0.4
#     early_stop = 200
#     for _ in range(100):
#         # Set random seed for reproducibility
#         # seed = random.randint(0, 10000)
#         # torch.manual_seed(seed)
#         # random.seed(seed)
#
#         # Create and train model
#         model = DNN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
#         model = train_model(
#             model,
#             train_loader,
#             val_loader,
#             num_epochs=num_epochs,
#             lr=learning_rate,
#             lr_decay=learning_rate_decay,
#             momentum=0.9,
#             nesterov=True,
#             early_stop=early_stop
#         )
#
#         # Test model and calculate uncertainties
#         val_rmse = test_model(model, val_loader)
#         test_rmse = test_model(model, test_loader)
#
#         if val_rmse < 1.2:
#             ensemble.append((model, val_rmse, test_rmse))
#
#     return ensemble

# def train(
#         model,
#         dataloader,
#         optimizer,
#         loss_fn,
# ):
#     model.train()
#     train_loss = 0
#     for batch_idx, (inputs, targets) in enumerate(dataloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         # Create a mask for Nan targets
#         nan_mask = torch.isnan(targets)
#         # Loss calculation without nan in the mean
#         loss = calc_loss_notnan(outputs, targets, nan_mask, loss_fn)
#         # loss = loss_fn(outputs, target)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#
#     train_loss /= len(dataloader)
#     return train_loss

#
# def evaluate(
#         model,
#         dataloader,
#         loss_fn,
# ):
#     model.eval()
#     total_loss = 0.0
#     targets_all = []
#     outputs_all = []
#
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(dataloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             # Create a mask for Nan targets
#             nan_mask = torch.isnan(targets)
#             # Loss calculation without nan in the mean
#             loss = calc_loss_notnan(outputs, targets, nan_mask, loss_fn)
#             total_loss += loss.item()
#             targets_all.append(targets)
#             outputs_all.append(outputs)
#         total_loss /= len(dataloader)
#         targets_all = torch.cat(targets_all, dim=0)
#         outputs_all = torch.cat(outputs_all, dim=0)
#
#         # Calculate metrics for the ensemble
#         ensemble_rmse, ensemble_r2, ensemble_evs = calc_regr_metrics(targets_all, outputs_all)
#
#         # Calculate metrics for each individual model in the ensemble
#         model_metrics = []
#         for model_output in outputs_all:
#             model_rmse, model_r2, model_evs = calc_regr_metrics(targets_all, model_output)
#             model_metrics.append((model_rmse, model_r2, model_evs))
#
#         # Calculate uncertainties or variances
#         uncertainties = torch.var(outputs_all, dim=0)
#     # return total_loss, rmse, r2, evs
#     return total_loss, ensemble_rmse, ensemble_r2, ensemble_evs, model_metrics, uncertainties

#
# def ensemble_pipeline(config=wandb.config, wandb_project_name="test-project"):
#     with wandb.init(
#             dir=LOG_DIR,
#             mode=wandb_mode,
#             project=wandb_project_name,
#             config=config
#     ):
#         config = wandb.config
#
#         # Load the dataset
#         train_loader, val_loader, test_loader = build_loader(config)
#
#         # Load the model
#         model = BaselineDNN(
#             input_dim=config.input_dim,
#             hidden_dim_1=config.hidden_dim_1,
#             hidden_dim_2=config.hidden_dim_2,
#             hidden_dim_3=config.hidden_dim_3,
#             num_tasks=config.num_tasks,
#             dropout=config.dropout
#         )
#         model = model.to(device)
#
#         # Define the loss function
#         loss_fn = build_loss(config.loss, reduction='none')
#
#         # Define the optimizer with weight decay and learning rate scheduler
#         optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
#
#         # Define Learning rate scheduler
#         lr_scheduler = ReduceLROnPlateau(
#             optimizer,
#             mode='min',
#             factor=config.lr_factor,
#             patience=config.lr_patience,
#             verbose=True
#         )
#
#         # Train the model
#         best_val_loss = float('inf')
#         early_stop_counter = 0
#         for epoch in tqdm(range(config.num_epochs)):
#             # Training
#             train_loss = train(model, train_loader, optimizer, loss_fn)
#             # Validation
#             val_loss, ensemble_rmse, ensemble_r2, ensemble_evs, model_metrics, uncertainties = evaluate(model,
#                                                                                                         test_loader,
#                                                                                                         loss_fn)
#             # val_loss, val_rmse, val_r2, val_evs = evaluate(model, val_loader, loss_fn)
#             # Log the metrics
#             wandb.log(
#                 data={
#                     'epoch': epoch,
#                     'train_loss': train_loss,
#                     'val_loss': val_loss,
#                     'ensemble_rmse': ensemble_rmse,
#                     'ensemble_r2': ensemble_r2,
#                     'ensemble_evs': ensemble_evs,
#                     'model_metrics': model_metrics,
#                     'uncertainties': uncertainties
#                 }
#             )
#
#             # Update the learning rate
#             lr_scheduler.step(val_loss)
#
#             # Early stopping
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 early_stop_counter = 0
#                 # Save the best model - dropped to avoid memory issues
#                 # Update the best model and its performance
#                 best_model = model
#                 # best_model = copy.deepcopy(model)
#
#                 # Save the best model
#                 # save_models(config, model)
#
#                 # optional: save model at the end to view in wandb
#                 # inputs, _ = next(iter(train_loader)) # Takes so much time # TODO: put it before epochs loop
#                 # inputs = inputs.to(device)
#                 # x = torch.zeros((config.batch_size, config.input_dim), dtype=torch.float32, device=device,
#                 #                      requires_grad=False)
#                 # torch.onnx.export(model, x, f'models/{today}_best_model.onnx')
#                 # wandb.save(f"models/{today}_best_model.onnx")
#
#             else:
#                 early_stop_counter += 1
#                 # print(config)
#                 if early_stop_counter > config.early_stop:
#                     break
#
#         # Save the best model
#         save_models(config, best_model)
#         # saved_model_path = f'models/saved_models/{config.activity}/'
#         # if not os.path.exists(saved_model_path):
#         # torch.save(best_model.state_dict(), f'models/{today}_best_model.pt')
#         # wandb.save(f"models/saved_models/{today}_best_model.pt")
#
#         # Load the best model
#         # model.load_state_dict(torch.load(f'models/{today}_best_model.pt'))
#         # Test
#         test_loss, test_rmse, test_r2, test_evs = evaluate(best_model, test_loader, loss_fn)  # , last_batch_log=True
#         # Log the final test metrics
#         wandb.log({
#             'test_loss': test_loss,
#             'test_rmse': test_rmse,
#             'test_r2': test_r2,
#             'test_evs': test_evs
#         })
#
#         return test_loss, test_rmse, test_r2, test_evs


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

# class EnsembleDNN(nn.Module):
#     def __init__(self, base_model, num_of_ensemble):
#         super(EnsembleDNN, self).__init__()
#         # TODO: use different random seeds for initialization of each model
#
#         self.models = nn.ModuleList([base_model for _ in range(num_of_ensemble)])
#
#     def forward(self, x):
#         outputs = torch.stack([model(x) for model in self.models])
#         return outputs
#

# def get_config(activity='xc50', split='random'):
#     config = {
#         'activity': activity,
#         'batch_size': 128,
#         'dropout': 0.1,
#         'early_stop': 100,
#         'hidden_dim_1': 2048,
#         'hidden_dim_2': 256,
#         'hidden_dim_3': 256,
#         'input_dim': 2048,
#         'learning_rate': 0.01,
#         'loss': 'huber',
#         'lr_factor': 0.5,
#         'lr_patience': 20,
#         'num_epochs': 3000,  # 20,
#         'num_tasks': 20,
#         'optimizer': 'sgd',
#         'output_dim': 20,
#         'weight_decay': 0.001,
#         'seed': 42,
#         'split': split,
#     }
#
#     return config
#
#
# def get_sweep_config(activity='xc50', split='random'):
#     # # Initialize wandb
#     # wandb.init(project='multitask-learning')
#     # Sweep configuration
#     sweep_config = {
#         'method': 'random',
#         'metric': {
#             'name': 'val_rmse', # 'val_loss',
#             'goal': 'minimize'
#         },
#         'parameters': {
#             'input_dim': {
#                 'values': [1024, 2048]
#             },
#             'hidden_dim_1': {
#                 'values': [512, 1024, 2048]
#             },
#             'hidden_dim_2': {
#                 'values': [256, 512]
#             },
#             'hidden_dim_3': {
#                 'values': [128, 256]
#             },
#             'num_tasks': {
#                 'value': 20
#             },
#             'batch_size': {
#                 'values': [64, 128, 256]
#             },
#             'loss': {
#                 'values': ['huber', 'mse']
#             },
#             'learning_rate': {
#                 'values': [0.001, 0.01]
#             },
#             'ensemble_size': {
#                 'value': 100
#             },
#             'weight_decay': {
#                 'value': 0.001
#             },
#             'dropout': {
#                 'values': [0.1, 0.2]
#             },
#             'lr_factor': {
#                 'value': 0.5
#             },
#             'lr_patience': {
#                 'value': 20
#             },
#             'num_epochs': {
#                 'value': 3000
#             },
#             'early_stop': {
#                 'value': 100
#             },
#             'optimizer': {
#                 'values': ['adamw', 'sgd']
#             },
#             'output_dim': {
#                 'value': 20
#             },
#             'activity': {
#                 'value': "xc50"
#             },
#             'seed': {
#                 'value': 42
#             },
#         },
#     }
#     # 576 combinations
#     return sweep_config
#
