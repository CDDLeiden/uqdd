__author__ = "Bola Khalil"
__supervisor__ = "Kajetan Schweighofer"
__copyright__ = "Copyright 2022, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__license__ = "All rights reserved, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__version__ = "0.0.1"
__maintainer__ = "Bola Khalil"
__email__ = "bkhalil@its.jnj.com"
__status__ = "Development"

import os
# import sys; sys.path.append()
from datetime import date
import json, yaml
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from torch.utils.data import DataLoader

from uqdd.chemutils import smi_to_pil_image
from uqdd.models.papyrus import PapyrusDataset

DATA_DIR = os.environ.get('DATA_DIR')
LOGS_DIR = os.environ.get('LOGS_DIR')
CONFIG_DIR = os.environ.get('CONFIG_DIR')

# wandb_dir = LOGS_DIR  # 'logs/'
wandb_mode = 'online'
# data_dir = DATA_DIR  # 'data/'  # 'data/papyrus_filtered_high_quality_xc50_01_standardized.csv'
dataset_dir = os.path.join(DATA_DIR, 'dataset/')  # 'data/dataset/'

today = date.today()
today = today.strftime("%Y%m%d")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device: " + str(device))
# print(torch.version.cuda) if device == 'cuda' else None


def log_mol_table(smiles, inputs, targets, outputs, targets_names):
    # targets_cols = targets.columns()
    # table_cols = ['smiles', 'mol', 'mol_2D', 'ECFP', 'fp_length']
    # for t in targets_cols:
    #     table_cols.append(f'{t}_label')
    #     table_cols.append(f'{t}_predicted')
    # table = wandb.Table(columns=table_cols)
    # with wandb.init(dir=wandb_dir, mode=wandb_mode):
    data = []
    for smi, inp, tar, out in zip(smiles, inputs.to("cpu"), targets.to("cpu"), outputs.to("cpu")):
        row = {
            "smiles": smi,
            "molecule": wandb.Molecule.from_smiles(smi),
            "molecule_2D": wandb.Image(smi_to_pil_image(smi)),
            "ECFP": inp,
            "fp_length": len(inp),
        }

        # Iterate over each pair of output and target
        for targetName, target, output in zip(targets_names, tar, out):
            row[f'{targetName}_label'] = target.item()
            row[f'{targetName}_predicted'] = output.item()

        data.append(row)

    dataframe = pd.DataFrame.from_records(data)
    table = wandb.Table(dataframe=dataframe)
    wandb.log({"mols_table": table}, commit=False)


def get_datasets(activity, split):
    try:
        d_dir = os.path.join(dataset_dir, activity, split)

        train_path = os.path.join(d_dir, "train.pkl")
        val_path = os.path.join(d_dir, "val.pkl")
        test_path = os.path.join(d_dir, "test.pkl")

        train_set_1024 = PapyrusDataset(train_path, input_col="ecfp1024", device=device)
        val_set_1024 = PapyrusDataset(val_path, input_col="ecfp1024", device=device)
        test_set_1024 = PapyrusDataset(test_path, input_col="ecfp1024", device=device)

        train_set_2048 = PapyrusDataset(train_path, input_col="ecfp2048", device=device)
        val_set_2048 = PapyrusDataset(val_path, input_col="ecfp2048", device=device)
        test_set_2048 = PapyrusDataset(test_path, input_col="ecfp2048", device=device)
        print("Train set size: " + str(len(train_set_1024)))
        print("Val set size: " + str(len(val_set_1024)))
        print("Test set size: " + str(len(test_set_1024)))

        return train_set_1024, val_set_1024, test_set_1024, \
            train_set_2048, val_set_2048, test_set_2048

    except Exception as e:
        # print("Error loading data")
        raise Exception(f"Error building dataset with PapyrusDataset {e}")


def build_loader(datasets, batch_size, ecfp_size=1024):
    try:
        train_set, val_set, test_set = datasets[:3] if ecfp_size == 1024 else datasets[3:]

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # , pin_memory=True
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)  # , pin_memory=True
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)  # , pin_memory=True
        print("Data loaders created")
    except Exception as e:
        # print("Error loading data")
        raise Exception(f"Error loading data {e}")

    return train_loader, val_loader, test_loader


def _build_loader(config=wandb.config):
    """Deprecated function"""
    try:
        d_dir = os.path.join(dataset_dir, config.activity, config.split)

        train_path = os.path.join(d_dir, "train.pkl")
        val_path = os.path.join(d_dir, "val.pkl")
        test_path = os.path.join(d_dir, "test.pkl")
        print("Loading data from: " + d_dir)
        train_set = PapyrusDataset(train_path, input_col=f"ecfp{config.input_dim}", device=device)
        val_set = PapyrusDataset(val_path, input_col=f"ecfp{config.input_dim}", device=device)
        test_set = PapyrusDataset(test_path, input_col=f"ecfp{config.input_dim}", device=device)
        print("Train set size: " + str(len(train_set)))
        print("Val set size: " + str(len(val_set)))
        print("Test set size: " + str(len(test_set)))

        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)  # , pin_memory=True
        val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)  # , pin_memory=True
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)  # , pin_memory=True
        print("Data loaders created")

    except Exception as e:
        raise Exception("Error loading data: " + str(e))

    return train_loader, val_loader, test_loader


def build_optimizer(model, optimizer, lr, weight_decay):
    if optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Unknown optimizer: {}'.format(optimizer))

    return optimizer


def build_loss(loss, reduction='none'):
    if loss.lower() == 'mse':
        loss_fn = nn.MSELoss(reduction=reduction)
    elif loss.lower() in ['mae', 'l1']:
        loss_fn = nn.L1Loss(reduction=reduction)
    elif loss.lower() in ['huber', 'smoothl1']:
        loss_fn = nn.SmoothL1Loss(reduction=reduction)
    else:
        raise ValueError('Unknown loss: {}'.format(loss))
    return loss_fn


def save_models(config, model):
    try:
        model_dir = os.path.join('', 'models/saved_models', config.activity, config.split)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{today}-{wandb.run.name}-best-model")
        pt_path = model_path + ".pt"
        onnx_path = model_path + ".onnx"

        dummy_input = torch.zeros(
            (config.batch_size, config.input_dim),
            dtype=torch.float32,
            device=device,
            requires_grad=False
        )
        torch.onnx.export(model, dummy_input, onnx_path)
        torch.save(model.state_dict(), pt_path)
        # ONNX saving
        wandb.save(onnx_path)

    except Exception as e:
        print("Error saving models: " + str(e))


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
        >>> import torch
        >>> tensor_for_agg = torch.tensor([[1, 2, 3], [4, float('nan'), 6]])
        >>> nan_mask = torch.isnan(tensor_for_agg)
        >>> aggregated_tensor = calc_nanaware_metrics(tensor_for_agg, nan_mask, all_tasks_agg=True)
        >>> print(aggregated_tensor)
        tensor(3.3333)

        The above example demonstrates the usage of the `agg_notnan` function.
        The input tensor contains NaN values, and the nan_mask is used to identify those NaN values.
        By specifying `all_tasks_agg=True`, the function calculates the mean of the non-NaN values and then
        returns the mean of all tasks. In this case, the output is `tensor(3.3333)`.
        """
    # Now we only include the non-Nan targets in the mean calc.
    tensor_means = torch.sum(tensor, dim=0) / torch.sum(~nan_mask, dim=0)

    # If we want to aggregate across all tasks, we do so here.
    if not all_tasks_agg:
        return tensor_means
    # TODO - check if this is correct - SUM OR MEAN?
    elif all_tasks_agg == 'mean':
        return torch.nanmean(tensor_means)
    else:
        return torch.nansum(tensor_means)


def calc_regr_metrics(targets, outputs):
    targets = targets.cpu().numpy()
    outputs = outputs.cpu().numpy()

    # no reduction here because we want to calc per task metrics
    rmse = mean_squared_error(targets, outputs, squared=False)
    r2 = r2_score(targets, outputs)
    evs = explained_variance_score(targets, outputs)

    return rmse, r2, evs

    # multioutput='raw_values'
    # rmse = calc_nanaware_metrics(
    #     tensor=torch.from_numpy(rmse),
    #     nan_mask=torch.from_numpy(nan_mask),
    #     all_tasks_agg='mean'
    # )

    # r2 = calc_nanaware_metrics(
    #     tensor=torch.from_numpy(r2),
    #     nan_mask=torch.from_numpy(nan_mask),
    #     all_tasks_agg='mean'
    # )

    # evs = calc_nanaware_metrics(
    #     tensor=torch.from_numpy(evs),
    #     nan_mask=torch.from_numpy(nan_mask),
    #     all_tasks_agg='mean'
    # )


def calc_loss_notnan(outputs, targets, nan_mask, loss_fn):
    targets[nan_mask], outputs[nan_mask] = 0.0, 0.0

    loss_per_task = loss_fn(outputs, targets)

    # Now we only include the non-Nan targets in the mean calc.
    loss = calc_nanaware_metrics(tensor=loss_per_task, nan_mask=nan_mask, all_tasks_agg='sum')
    # task_losses = torch.sum(loss_per_task, dim=1) / torch.sum(~nan_mask, dim=1)
    # loss = torch.sum(task_losses)
    return loss


def get_config(
        config=os.path.join(CONFIG_DIR, 'baseline.json'),
        **kwargs
):
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
    # Load config from JSON file or YAML file
    default_config = {
        "activity": "xc50",
        "batch_size": 128,
        "dropout": 0.1,
        "early_stop": 100,
        "input_dim": 2048,
        "hidden_dim_1": 1024,
        "hidden_dim_2": 512,
        "hidden_dim_3": 256,
        "learning_rate": 0.01,
        "loss": "huber",
        "lr_factor": 0.5,
        "lr_patience": 20,
        "num_epochs": 3000,
        "num_tasks": 20,
        "optimizer": "sgd",
        "output_dim": 20,
        "weight_decay": 0.001,
        "seed": 42,
        "split": "scaffold",
        "ensemble_size": 10
    }

    if config is None:
        config = default_config

    elif os.path.isfile(config):
        if config.endswith('.json'):
            with open(config, 'r') as f:
                config = json.load(f)
        elif config.endswith('.yaml'):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported config file format. Please use JSON or YAML.")
    elif isinstance(config, dict):
        pass
    else:
        raise ValueError("Invalid config. Please provide a valid config file path or a dictionary.")

    if kwargs:
        config.update(kwargs)

    config = default_config.update(config)

    return config


def get_sweep_config(
        config=os.path.join(CONFIG_DIR, 'baseline_sweep.json'),
        **kwargs
):
    """
    Retrieve the sweep configuration for hyperparameter tuning.

    Parameters:
    -----------
    - config (dict, str, or None): A dictionary containing sweep configuration parameters, a path to a YAML or JSON config file, or None to use the default configuration.
    - **kwargs: Additional keyword arguments that override the values in the 'parameters' dictionary.

    Returns:
    --------
    - dict: Merged dictionary containing the sweep configuration parameters.

    Notes:
    ------
    - If `config` is None, the function will return the default sweep configuration.
    - If `config` is a path to a YAML or JSON file, the function will load the configuration from the file.
    - The default configuration values will be overridden by `config` and `kwargs`, if provided.
    - If both `config` and `kwargs` contain the same key in the 'parameters' dictionary, the value from `kwargs` will take precedence.
    """
    # Load config from JSON file or YAML file
    default_sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_rmse',  # 'val_loss',
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
            'split': {
                'value': "scaffold"
            },
        },
    }

    if config is None:
        config = default_sweep_config

    elif os.path.isfile(config):
        if config.endswith('.json'):
            with open(config, 'r') as f:
                config = json.load(f)
        elif config.endswith('.yaml'):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported config file format. Please use JSON or YAML.")
    elif isinstance(config, dict):
        pass
    else:
        raise ValueError("Invalid config. Please provide a valid config file path or a dictionary.")

    if kwargs:
        # Update the 'parameters' dictionary with kwargs
        for key, value in kwargs.items():
            if isinstance(value, list):
                config['parameters'][key] = {'values': value}
            else:
                config['parameters'][key] = {'value': value}

    config = default_sweep_config.update(config)

    return config


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

