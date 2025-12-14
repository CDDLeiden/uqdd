"""
Model architecture helpers and utilities.

This module provides functions for setting random seeds, computing norms,
building datasets and data loaders, initializing optimizers and learning rate
schedulers, saving and loading models, and generating model names and paths.
"""

import logging
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from uqdd import CONFIG_DIR, MODELS_DIR, TODAY, DEVICE
from uqdd.data.utils_data import get_topx
from uqdd.utils import get_config, create_logger

string_types = (type(b""), type(""))


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across libraries.

    Parameters
    ----------
    seed : int, optional
        Seed value to use for random number generation. Default is ``42``.

    Returns
    -------
    None
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
    """
    Compute the L2 norm of a model's parameters.

    Parameters
    ----------
    model : nn.Module
        Neural network model.

    Returns
    -------
    float
        L2 norm of the model's parameters.
    """
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:
    """
    Compute the L2 norm of a model's gradients.

    Parameters
    ----------
    model : nn.Module
        Neural network model.

    Returns
    -------
    float
        L2 norm of the model's gradients (over parameters with non-None gradients).
    """
    return math.sqrt(
        sum(
            [
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            ]
        )
    )


def get_desc_len_from_dataset(dataset: torch.utils.data.Dataset) -> Tuple[int, int]:
    """
    Retrieve the lengths of protein and chemical descriptors from a dataset sample.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset containing descriptor tensors.

    Returns
    -------
    (int, int)
        Lengths of protein descriptors and chemical descriptors, respectively.
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


def get_desc_len(
        *descriptors: Optional[str], logger: Optional[logging.Logger] = None
) -> Tuple[int, ...]:
    """
    Retrieve descriptor lengths from configuration.

    Parameters
    ----------
    descriptors : str or None
        Variable number of descriptor names to query.
    logger : logging.Logger or None, optional
        Logger for debug output.

    Returns
    -------
    tuple of int
        Lengths of the requested descriptors (0 if missing).
    """
    desc_config = get_config(config_name="desc_dim", config_dir=CONFIG_DIR)
    lengths = []
    for d in descriptors:
        l = desc_config.get(d, 0)
        if logger:
            logger.debug(f"{d} descriptor length: {l}")
        lengths.append(l)
    return tuple(lengths)


def get_model_config(model_type: str = "pnn", **kwargs) -> Dict:
    """
    Retrieve the configuration dictionary for model training.

    Parameters
    ----------
    model_type : str, optional
        Model type (e.g., "pnn", "ensemble", "mcdropout"). Default is ``"pnn"``.
    **kwargs
        Additional parameters to override default configuration values.

    Returns
    -------
    dict
        Model configuration dictionary.
    """
    assert model_type in [
        "pnn",
        "ensemble",
        "mcdropout",
        "evidential",
        "eoe",
        "emc",
        # "gp",
    ], f"Invalid model name: {model_type}"

    split_type = kwargs.get("split_type", "random")
    activity_type = kwargs.get("activity_type", "xc50")

    return get_config(
        config_name=model_type,
        config_dir=CONFIG_DIR,
        split_key=split_type,
        activity_key=activity_type,
        **kwargs,
    )


def get_sweep_config(model_name: str = "pnn", **kwargs) -> Dict:
    """
    Retrieve the sweep configuration for hyperparameter tuning.

    Parameters
    ----------
    model_name : str, optional
        Model name to retrieve sweep configurations for. Default is ``"pnn"``.
    **kwargs
        Additional parameters to override default sweep configurations.

    Returns
    -------
    dict
        Sweep configuration dictionary.

    Notes
    -----
    - If ``config`` is None, returns the default sweep configuration.
    - Default values are overridden by kwargs.
    """
    assert model_name in [
        "pnn",
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
        data_name: str = "papyrus",
        n_targets: int = -1,
        activity_type: str = "xc50",
        split_type: str = "random",
        desc_prot: Optional[str] = None,
        desc_chem: Optional[str] = None,
        median_scaling: bool = False,
        task_type: str = "regression",
        ext: str = "pkl",
        logger: Optional[logging.Logger] = None,
        device: Union[str, torch.device] = DEVICE,
) -> Dict[str, torch.utils.data.Dataset]:
    """
    Build datasets for training and evaluation.

    Parameters
    ----------
    data_name : str, optional
        Dataset to load (e.g., "papyrus"). Default is ``"papyrus"``.
    n_targets : int, optional
        Number of targets (``-1`` for all). Default is ``-1``.
    activity_type : str, optional
        Type of bioactivity measurement. Default is ``"xc50"``.
    split_type : str, optional
        Dataset split type (e.g., "random", "scaffold"). Default is ``"random"``.
    desc_prot : str or None, optional
        Protein descriptor name.
    desc_chem : str or None, optional
        Chemical descriptor name.
    median_scaling : bool, optional
        Whether to apply median scaling to labels. Default is ``False``.
    task_type : {"regression", "classification"}, optional
        Task type. Default is ``"regression"``.
    ext : str, optional
        File extension for dataset files. Default is ``"pkl"``.
    logger : logging.Logger or None, optional
        Logger for tracking dataset building.
    device : str or torch.device, optional
        Device on which data will be processed. Default is ``DEVICE``.

    Returns
    -------
    dict of str -> torch.utils.data.Dataset
        Training, validation, and test datasets.
    """
    logger = create_logger(name="build_datasets") if not logger else logger
    logger.debug(f"Building datasets for {data_name}")
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
            device=device,
        )

    else:
        raise ValueError(
            f"Unknown data name: {data_name}"
            f"Please choose from 'papyrus', 'tdc', or 'other'"
        )

    return datasets


def build_loader(
        datasets: Dict[str, torch.utils.data.Dataset],
        batch_size: int,
        shuffle: bool = False,
        wt_resampler: bool = False,
) -> Dict[str, DataLoader]:
    """
    Construct data loaders for training, validation, and testing.

    Parameters
    ----------
    datasets : dict of str -> torch.utils.data.Dataset
        Dataset splits keyed by name (e.g., "train", "val", "test").
    batch_size : int
        Batch size for loading data.
    shuffle : bool, optional
        Whether to shuffle data. Default is ``False``.
    wt_resampler : bool, optional
        Whether to use a weighted random sampler for the training set. Default is ``False``.

    Returns
    -------
    dict of str -> DataLoader
        Data loaders for each dataset split.
    """
    try:
        dataloaders = {}
        for k, v in datasets.items():
            if k == "train" and wt_resampler:
                sampler = get_sampler(v, bins=1000)
            else:
                sampler = None
            dataloaders[k] = DataLoader(
                v,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
            )
        logging.debug("Data loaders created")
    except Exception as e:
        raise RuntimeError(f"Error loading data {e}")

    return dataloaders


def get_sampler(
        dataset: torch.utils.data.Dataset, bins: int = 50
) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for handling imbalanced datasets.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset for which to create the sampler.
    bins : int, optional
        Number of bins for discretizing the label distribution. Default is ``50``.

    Returns
    -------
    WeightedRandomSampler
        Weighted random sampler for balanced sampling.
    """
    labels = dataset.labels
    min_val, max_val = labels.min(), labels.max()
    bin_edges = torch.linspace(min_val, max_val, bins + 1, device=labels.device)
    bin_indices = torch.bucketize(labels, bin_edges)
    bin_counts = torch.bincount(bin_indices.squeeze(), minlength=bins + 1).float()

    # Compute weights for each sample - inverse of the frequency of the bin
    weights = 1.0 / (bin_counts[bin_indices] + 1e-2)
    # Normalize the weights
    weights = weights / weights.sum() * len(weights)
    # Create the sampler with the calculated weights
    sampler = WeightedRandomSampler(
        weights=weights.squeeze(),
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator(device=labels.device),
    )
    return sampler


def build_optimizer(
        model: nn.Module, optimizer: str, lr: float, weight_decay: float
) -> optim.Optimizer:
    """
    Initialize an optimizer for training a model.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters will be optimized.
    optimizer : str
        Name of optimizer (e.g., "adam", "sgd").
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay (L2 penalty).

    Returns
    -------
    torch.optim.Optimizer
        Initialized optimizer instance.
    """
    if optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    elif optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    return optimizer


def build_lr_scheduler(
        optimizer: optim.Optimizer,
        lr_scheduler: Optional[str],
        patience: int = 20,
        factor: float = 0.2,
        **kwargs,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Construct a learning rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer for which to adjust the learning rate.
    lr_scheduler : str or None, optional
        Type of learning rate scheduler (e.g., "plateau", "step", "exp", "cos").
    patience : int, optional
        Number of epochs with no improvement before reducing LR. Default is ``20``.
    factor : float, optional
        Factor by which to reduce LR. Default is ``0.2``.
    **kwargs
        Additional parameters for the scheduler.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler or None
        Learning rate scheduler instance, or ``None``.
    """
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
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")

    return lr_scheduler


def save_model(
        config: Dict[str, Any],
        model: nn.Module,
        model_name: str = f"{TODAY}-pnn_random_ankh-base_ecfp2048",
        data_specific_path: Optional[str] = None,
        desc_prot_len: int = 0,
        desc_chem_len: int = 1024,
        onnx: bool = True,
        tracker: str = "wandb",
) -> None:
    """
    Save a trained model in both PyTorch and ONNX formats.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model training parameters.
    model : nn.Module
        Trained model to save.
    model_name : str, optional
        Saved model file name. Default is a name generated from date and settings.
    data_specific_path : str or None, optional
        Path for saving model-specific configurations.
    desc_prot_len : int, optional
        Length of protein descriptors. Default is ``0``.
    desc_chem_len : int, optional
        Length of chemical descriptors. Default is ``1024``.
    onnx : bool, optional
        Whether to save the model in ONNX format. Default is ``True``.
    tracker : str, optional
        Model tracking system to use (e.g., "wandb"). Default is ``"wandb"``.

    Returns
    -------
    None
    """
    try:
        if model is None:
            raise ValueError(f"No model to save - {model=}")

        model_dir = MODELS_DIR / "saved_models" / data_specific_path

        model_dir.mkdir(parents=True, exist_ok=True)

        pt_path = model_dir / f"{model_name}.pt"
        i = 1
        while pt_path.exists():
            model_name += f"_{i}"
            pt_path = model_dir / f"{model_name}.pt"
            i += 1

        torch.save(model.state_dict(), pt_path)
        wandb_model_path = pt_path

        if onnx:
            onnx_path = model_dir / f"{model_name}.onnx"
            batch_size = config.get("batch_size", 64)
            if desc_prot_len == 0:
                dummy_input = {
                    "inputs": torch.zeros(
                        (batch_size, desc_chem_len),
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
                                (batch_size, desc_prot_len),
                                dtype=torch.float32,
                                device=DEVICE,
                                requires_grad=False,
                            ),
                            torch.zeros(
                                (batch_size, desc_chem_len),
                                dtype=torch.float32,
                                device=DEVICE,
                                requires_grad=False,
                            ),
                        )
                    )
                }
            torch.onnx.export(model, dummy_input, onnx_path)
            wandb_model_path = str(onnx_path)

        if tracker.lower() == "wandb":
            wandb.save(wandb_model_path, base_path=model_dir)

    except Exception as e:
        print("Error saving models: " + str(e))


def add_prefix_to_state_dict_keys(
        state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    """
    Add a prefix to all keys in a model's state dictionary.

    Parameters
    ----------
    state_dict : dict of str -> torch.Tensor
        Original state dictionary.
    prefix : str
        Prefix to prepend to each key.

    Returns
    -------
    dict of str -> torch.Tensor
        Updated state dictionary with modified keys.
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key if key.startswith(prefix) else f"{prefix}{key}"
        new_state_dict[new_key] = value
    return new_state_dict


def load_model(
        model_class: nn.Module | Any,
        model_path: str,
        prefix_to_state_keys: Optional[str] = None,
        **model_kwargs,
) -> nn.Module:
    """
    Load a PyTorch model from a saved state dictionary.

    Parameters
    ----------
    model_class : nn.Module or Any
        Model class to instantiate.
    model_path : str
        Path to the saved state dictionary.
    prefix_to_state_keys : str or None, optional
        Prefix to prepend to state dictionary keys if needed.
    **model_kwargs
        Additional arguments for initializing the model.

    Returns
    -------
    nn.Module
        Model instance with loaded weights.
    """
    model = model_class(**model_kwargs)

    state_dict = torch.load(model_path)

    if prefix_to_state_keys:
        state_dict = add_prefix_to_state_dict_keys(state_dict, prefix_to_state_keys)

    model.load_state_dict(state_dict)

    return model


def get_ckpt_path(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate and return the path for model checkpoint storage.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model details.

    Returns
    -------
    dict
        Updated configuration dictionary with checkpoint path information.
    """
    dir = MODELS_DIR / "ckpt"
    dir.mkdir(parents=True, exist_ok=True)
    if config.get("ckpt_name", None):
        ckpt_name = config.get("ckpt_name")
    else:
        ckpt_name = get_model_name(config)
    i = 0
    model_path = Path(dir / f"{ckpt_name}.pth")
    while model_path.exists():
        ckpt_name_m = ckpt_name + f"_{i}"
        model_path = dir / f"{ckpt_name_m}.pth"
        i += 1
    model_path.touch()

    config["ckpt_name"] = model_path.stem
    config["ckpt_path"] = model_path

    return config


def ckpt(model: nn.Module, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save a model checkpoint to disk.

    Parameters
    ----------
    model : nn.Module
        Model whose state should be saved.
    config : dict
        Configuration dictionary containing checkpoint details.

    Returns
    -------
    dict
        Updated configuration dictionary with checkpoint path.
    """
    ckpt_path = config.get("ckpt_path", None)
    if not ckpt_path:
        config = get_ckpt_path(config)
        ckpt_path = config["ckpt_path"]
    torch.save(model.state_dict(), ckpt_path)
    return config


def load_ckpt(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """
    Load a model checkpoint from disk.

    Parameters
    ----------
    model : nn.Module
        Model to which weights should be loaded.
    config : dict
        Configuration dictionary containing checkpoint path.

    Returns
    -------
    nn.Module
        Model with restored weights.
    """
    ckpt_path = config.get("ckpt_path", None)
    ckpt_name = config.get("ckpt_name", None)
    if ckpt_path:
        print(f"Loading model {ckpt_name} from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
    return model


def get_model_name(config: Dict[str, Any], run: Optional[wandb.run] = None) -> str:
    """
    Generate a model name based on configuration settings.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model details.
    run : wandb.run or None, optional
        W&B run object for naming the model.

    Returns
    -------
    str
        Generated model name.
    """
    if config.get("model_name", None):
        model_name = config.get("model_name")
    else:
        data_name = config.get("data_name", "papyrus")
        activity_type = config.get("activity_type", "xc50")
        descriptor_protein = config.get("descriptor_protein", None)
        descriptor_chemical = config.get("descriptor_chemical", None)
        split_type = config.get("split_type", "random")
        model_type = config.get("model_type", "pnn")
        multitask = config.get("MT", False)
        seed = config.get("seed", 42)

        model_name = f"{TODAY}-{data_name}_{activity_type}_{model_type}_{split_type}_{descriptor_protein}_{descriptor_chemical}_{seed}"
        model_name += "_MT" if multitask else ""

    if run and not model_name.endswith(run.name):
        model_name += f"_{run.name}"

    return model_name


def get_data_specific_path(
        config: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> Path:
    """
    Construct a data-specific directory path for model storage.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing dataset details.
    logger : logging.Logger or None, optional
        Logger for debug output.

    Returns
    -------
    Path
        Directory path for storing dataset-specific results.
    """
    if config.get("data_specific_path", None):
        return config.get("data_specific_path")

    data_name = config.get("data_name", "papyrus")
    activity_type = config.get("activity_type", "xc50")
    n_targets = config.get("n_targets", -1)

    data_specific_path = Path(data_name) / activity_type / get_topx(n_targets)
    if logger:
        logger.debug(f"Data specific path: {data_specific_path}")
    return data_specific_path


def calculate_means(*tensors: torch.Tensor) -> List[torch.Tensor]:
    """
    Compute the mean along the last dimension for multiple tensors.

    Parameters
    ----------
    tensors : torch.Tensor
        One or more tensors to compute the mean over.

    Returns
    -------
    list of torch.Tensor
        Mean-reduced tensors.
    """
    return [torch.mean(tensor, dim=2) for tensor in tensors]


def stack_vars(*tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Stack multiple tensors along the last dimension.

    Parameters
    ----------
    tensors : list of torch.Tensor
        One or more tensors to stack.

    Returns
    -------
    list of torch.Tensor
        Stacked tensors.
    """
    return [torch.stack(tensor, dim=2) for tensor in tensors]
