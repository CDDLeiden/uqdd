__author__ = "Bola Khalil"
__supervisor__ = "Kajetan Schweighofer"
__copyright__ = "Copyright 2022, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__license__ = "All rights reserved, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__version__ = "0.0.1"
__maintainer__ = "Bola Khalil"
__email__ = "bkhalil@its.jnj.com"
__status__ = "Development"

import copy
import wandb

from tqdm import tqdm
from functools import partial
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.models_utils import build_loader, build_optimizer, build_loss, save_models, calc_loss_notnan, calc_regr_metrics, log_mol_table


# get today's date as yyyy/mm/dd format
from datetime import date
today = date.today()
today = today.strftime("%Y%m%d")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
print(torch.version.cuda) if device == 'cuda' else None

wandb_dir = 'logs/'
wandb_mode = 'online'
# data_dir = 'data/' # 'data/papyrus_filtered_high_quality_xc50_01_standardized.csv'
# dataset_dir = 'data/dataset/'


def get_config():
    config = {
        'activity': "xc50",
        'batch_size': 32,
        'dropout': 0.2,
        'early_stop': 5,
        'hidden_dim_1': 512,
        'hidden_dim_2': 256,
        'hidden_dim_3': 64,
        'input_dim': 1024,
        'learning_rate': 0.001,
        'loss': 'huber',
        'lr_factor': 0.1,
        'lr_patience': 10,
        'num_epochs': 10,  # 20,
        'num_tasks': 20,
        'optimizer': 'AdamW',
        'output_dim': 20,
        'weight_decay': 0.01,  # 1e-5,
        'seed': 42,
        'split': 'random'
    }

    return config


def get_sweep_config():
    # # Initialize wandb
    # wandb.init(project='multitask-learning')
    # Sweep configuration
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_rmse', # 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'input_dim': {
                'values': [1024, 2048]
            },
            'hidden_dim_1': {
                'values': [512, 1024, 2048]
            },
            'hidden_dim_2': {
                'values': [256, 512]
            },
            'hidden_dim_3': {
                'values': [128, 256]
            },
            'num_tasks': {
                'value': 20
            },
            'batch_size': {
                'values': [64, 128, 256]
            },
            'loss': {
                'values': ['huber', 'mse']
            },
            'learning_rate': {
                'values': [0.001, 0.01]
            },
            'ensemble_size': {
                'value': 100
            },
            'weight_decay': {
                'value': 0.001
            },
            'dropout': {
                'values': [0.1, 0.2]
            },
            'lr_factor': {
                'value': 0.5
            },
            'lr_patience': {
                'value': 20
            },
            'num_epochs': {
                'value': 3000
            },
            'early_stop': {
                'value': 100
            },
            'optimizer': {
                'values': ['adamw', 'sgd']
            },
            'output_dim': {
                'value': 20
            },
            'activity': {
                'value': "xc50"
            },
            'seed': {
                'value': 42
            },
            'split': {
                'values': ['random', 'scaffold']
            },
        },
    }
    # 576 combinations
    return sweep_config


class BaselineDNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim_1,
                 hidden_dim_2=None,
                 hidden_dim_3=None,
                 num_tasks=1,
                 dropout=0.2,
                 ):
        if hidden_dim_2 is None:
            hidden_dim_2 = hidden_dim_1
        if hidden_dim_3 is None:
            hidden_dim_3 = hidden_dim_1

        super(BaselineDNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )
        self.task_specific = nn.Linear(hidden_dim_3, num_tasks)
        self.apply(self.init_wt)

    @staticmethod
    def init_wt(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = self.task_specific(features)
        return outputs


def train(
        model,
        dataloader,
        optimizer,
        loss_fn,
):
    model.train()
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        # ➡ Forward pass
        outputs = model(inputs)
        # Create a mask for Nan targets
        nan_mask = torch.isnan(targets)
        # Loss calculation without nan in the mean
        loss = calc_loss_notnan(outputs, targets, nan_mask, loss_fn)
        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # if i % 100 == 0:
        #     # Log the loss in your run history
        #     wandb.log({"batch loss": loss})

    total_loss /= len(dataloader)
    return total_loss


def evaluate(
        model,
        loader,
        loss_fn,
):
    model.eval()
    total_loss = 0.0
    targets_all = []
    outputs_all = []
    # nan_masks_all = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Create a mask for Nan targets
            nan_mask = torch.isnan(targets)
            # Loss calculation without nan in the mean
            loss = calc_loss_notnan(outputs, targets, nan_mask, loss_fn)
            total_loss += loss.item()

            targets_all.append(targets)
            outputs_all.append(outputs)
            # nan_masks_all.append(nan_mask)

        total_loss /= len(loader)
        targets_all = torch.cat(targets_all, dim=0)
        outputs_all = torch.cat(outputs_all, dim=0)
        # nan_masks_all = torch.cat(nan_masks_all, dim=0)

        # Calculate metrics
        rmse, r2, evs = calc_regr_metrics(targets_all, outputs_all)

    return total_loss, rmse, r2, evs


def model_pipeline(config=wandb.config, wandb_project_name="test-project"):  #
    with wandb.init(
            dir=wandb_dir,
            mode=wandb_mode,
            project=wandb_project_name,
            config=config
    ):
        config = wandb.config

        # Load the dataset
        train_loader, val_loader, test_loader = build_loader(config)

        # set a random seed for reproducibility
        try:
            seed = config.seed
        except AttributeError:
            seed = 42
        # seed = 42 if not config.seed else config.seed
        torch.manual_seed(seed)
        # deterministic cuda algorithms
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)

        # Load the model
        model = BaselineDNN(
            input_dim=config.input_dim,
            hidden_dim_1=config.hidden_dim_1,
            hidden_dim_2=config.hidden_dim_2,
            hidden_dim_3=config.hidden_dim_3,
            # hidden_dim=config.hidden_dim,
            num_tasks=config.num_tasks,
            dropout=config.dropout
        )
        model = model.to(device)

        # Define the loss function
        loss_fn = build_loss(config.loss, reduction='none')

        # Define the optimizer with weight decay and learning rate scheduler
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)

        # Define Learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.lr_factor,
            patience=config.lr_patience,
            verbose=True
        )

        # Train the model
        best_val_loss = float('inf')
        early_stop_counter = 0
        for epoch in tqdm(range(config.num_epochs+1)):
            if epoch == 0:
                val_loss, val_rmse, val_r2, val_evs = evaluate(model, val_loader, loss_fn)
                train_loss, _, _, _ = evaluate(model, train_loader, loss_fn)
                # Log the metrics
                wandb.log(
                    data={
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_rmse': val_rmse,
                        'val_r2': val_r2,
                        'val_evs': val_evs
                    }
                )
                continue
            # else:
            # Training
            train_loss = train(model, train_loader, optimizer, loss_fn)
            # Validation
            val_loss, val_rmse, val_r2, val_evs = evaluate(model, val_loader, loss_fn)
            # Log the metrics
            wandb.log(
                data={
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'val_evs': val_evs
                }
            )

            # Update the learning rate
            lr_scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # Save the best model - dropped to avoid memory issues
                # Update the best model and its performance
                best_model = model

            else:
                early_stop_counter += 1
                # print(config)
                if early_stop_counter > config.early_stop:
                    break

        # Save the best model
        save_models(config, best_model)

        # Test
        test_loss, test_rmse, test_r2, test_evs = evaluate(best_model, test_loader, loss_fn)  # , last_batch_log=True
        # Log the final test metrics
        wandb.log({
            'test_loss': test_loss,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_evs': test_evs
        })

        return test_loss, test_rmse, test_r2, test_evs


def run_baseline(wandb_project_name=f"{today}-baseline"):
    config = get_config()
    test_loss, test_rmse, test_r2, test_evs = model_pipeline(
        config,
        wandb_project_name=wandb_project_name
    )
    return test_loss, test_rmse, test_r2, test_evs


def run_baseline_hyperparam(wandb_project_name=f"{today}-baseline-hyperparam", sweep_count=1):
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(
        sweep_config,
        project=wandb_project_name,
    )
    wandb_train_func = partial(
        model_pipeline,
        config=sweep_config,
        wandb_project_name=wandb_project_name,
    )
    wandb.agent(sweep_id, function=wandb_train_func, count=sweep_count)


if __name__ == '__main__':
    test_loss = run_baseline()
