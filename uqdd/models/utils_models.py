import math
import random
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple, Union
from collections import OrderedDict

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
    Sets the random seed for reproducibility across libraries.

    Parameters:
    -----------
    seed : int, default=42
        The seed value to use for random number generation.

    Returns:
    --------
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
    Computes the norm of a model's parameters.

    Parameters:
    -----------
    model : nn.Module
        The neural network model whose parameter norm is to be computed.

    Returns:
    --------
    float
        The L2 norm of the model's parameters.
    """
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:
    """
    Computes the norm of a model's gradients.

    Parameters:
    -----------
    model : nn.Module
        The neural network model whose gradient norm is to be computed.

    Returns:
    --------
    float
        The L2 norm of the model's gradients.
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
    Retrieves the lengths of protein and chemical descriptors from a dataset.

    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        The dataset containing descriptor information.

    Returns:
    --------
    Tuple[int, int]
        A tuple containing the length of protein descriptors and chemical descriptors.
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
    Retrieves descriptor lengths from the configuration.

    Parameters:
    -----------
    descriptors : Optional[str]
        Variable number of descriptor names for which lengths are required.
    logger : Optional[logging.Logger], default=None
        Logger for debugging information.

    Returns:
    --------
    Tuple[int, ...]
        A tuple containing the lengths of the requested descriptors.
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
    Retrieves the configuration dictionary for model training.

    Parameters:
    -----------
    model_type : str, default="pnn"
        The type of model configuration to load (e.g., "pnn", "ensemble", "mcdropout").
    **kwargs : dict
        Additional parameters to override default configuration values.

    Returns:
    --------
    Dict
        The model configuration dictionary.
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
    # print(split_type)

    return get_config(
        config_name=model_type,
        config_dir=CONFIG_DIR,
        split_key=split_type,
        activity_key=activity_type,
        **kwargs,
    )


def get_sweep_config(model_name: str = "pnn", **kwargs) -> Dict:
    """
    Retrieves the sweep configuration for hyperparameter tuning.

    Parameters:
    -----------
    model_name : str, default="pnn"
        The name of the model to retrieve sweep configurations for.
    **kwargs : dict
        Additional parameters to override default sweep configurations.

    Returns:
    --------
    Dict
        The sweep configuration dictionary.

    Notes:
    ------
    - If `config` is None, the function will return the default sweep configuration.
    - If `config` is a path to a YAML or JSON file, the function will load the configuration from the file.
    - The default configuration values will be overridden by `config` and `kwargs`, if provided.
    - If both `config` and `kwargs` contain the same key in the 'parameters' dictionary,
        the value from `kwargs` will take precedence.
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
    Builds datasets for training and evaluation.

    Parameters:
    -----------
    data_name : str, default="papyrus"
        The name of the dataset to load ("papyrus", "tdc", or "other").
    n_targets : int, default=-1
        The number of target entries to load (-1 for all).
    activity_type : str, default="xc50"
        The type of bioactivity measurement.
    split_type : str, default="random"
        The type of dataset split (e.g., "random", "scaffold").
    desc_prot : Optional[str], default=None
        The type of protein descriptor to use.
    desc_chem : Optional[str], default=None
        The type of chemical descriptor to use.
    median_scaling : bool, default=False
        Whether to apply median scaling to labels.
    task_type : str, default="regression"
        The type of task ("regression" or "classification").
    ext : str, default="pkl"
        The file extension format for loading the dataset.
    logger : Optional[logging.Logger], default=None
        Logger for tracking dataset building.
    device : str, default=DEVICE
        The device on which data will be processed.

    Returns:
    --------
    Dict[str, torch.utils.data.Dataset]
        A dictionary containing training, validation, and test datasets.
    """
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
            device=device,
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


def build_loader(
    datasets: Dict[str, torch.utils.data.Dataset],
    batch_size: int,
    shuffle: bool = False,
    wt_resampler: bool = False,
) -> Dict[str, DataLoader]:
    """
    Constructs data loaders for training, validation, and testing.

    Parameters:
    -----------
    datasets : Dict[str, torch.utils.data.Dataset]
        A dictionary containing dataset splits (e.g., "train", "val", "test").
    batch_size : int
        The batch size for loading data.
    shuffle : bool, optional
        Whether to shuffle the data (default: False).
    wt_resampler : bool, optional
        Whether to use a weighted random sampler for the training set (default: False).

    Returns:
    --------
    Dict[str, DataLoader]
        A dictionary containing the data loaders for each dataset split.
    """
    try:
        # num_cpu_cores = os.cpu_count()
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
                # num_workers=4,
                # pin_memory=True,
            )
        logging.debug("Data loaders created")
    except Exception as e:
        raise RuntimeError(f"Error loading data {e}")

    return dataloaders


def get_sampler(
    dataset: torch.utils.data.Dataset, bins: int = 50
) -> WeightedRandomSampler:
    """
    Creates a weighted random sampler for handling imbalanced datasets.

    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        The dataset for which to create the sampler.
    bins : int, optional
        The number of bins for discretizing the label distribution (default: 50).

    Returns:
    --------
    WeightedRandomSampler
        A PyTorch weighted random sampler for balanced sampling.
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
    # labels[torch.where(weights>1000)]
    return sampler


def build_optimizer(
    model: nn.Module, optimizer: str, lr: float, weight_decay: float
) -> optim.Optimizer:
    """
    Initializes an optimizer for training a model.

    Parameters:
    -----------
    model : nn.Module
        The model whose parameters will be optimized.
    optimizer : str
        The name of the optimizer to use (e.g., "adam", "sgd").
    lr : float
        The learning rate for optimization.
    weight_decay : float
        The weight decay (L2 penalty) applied to the optimizer.

    Returns:
    --------
    optim.Optimizer
        The initialized optimizer instance.
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
        raise ValueError("Unknown optimizer: {}".format(optimizer))

    return optimizer


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    lr_scheduler: Optional[str],
    patience: int = 20,
    factor: float = 0.2,
    **kwargs,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Constructs a learning rate scheduler.

    Parameters:
    -----------
    optimizer : optim.Optimizer
        The optimizer for which to adjust the learning rate.
    lr_scheduler : str, optional
        The type of learning rate scheduler (e.g., "plateau", "step", "exp", "cos").
    patience : int, optional
        The number of epochs with no improvement before reducing LR (default: 20).
    factor : float, optional
        The factor by which to reduce LR (default: 0.2).
    **kwargs : dict
        Additional parameters for the scheduler.

    Returns:
    --------
    Optional[optim.lr_scheduler._LRScheduler]
        The learning rate scheduler instance, or None if no scheduler is used.
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
        raise ValueError("Unknown lr_scheduler: {}".format(lr_scheduler))

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
    Saves a trained model in both PyTorch and ONNX formats.

    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing model training parameters.
    model : nn.Module
        The trained model to save.
    model_name : str, optional
        The name of the saved model file (default: generated from date and settings).
    data_specific_path : str, optional
        Path for saving model-specific configurations (default: None).
    desc_prot_len : int, optional
        Length of protein descriptors (default: 0).
    desc_chem_len : int, optional
        Length of chemical descriptors (default: 1024).
    onnx : bool, optional
        Whether to save the model in ONNX format (default: True).
    tracker : str, optional
        Model tracking system to use ("wandb" or other) (default: "wandb").

    Returns:
    --------
    None
    """
    try:
        if model is None:
            raise ValueError(f"No model to save - {model=}")

        model_dir = MODELS_DIR / "saved_models" / data_specific_path

        model_dir.mkdir(parents=True, exist_ok=True)
        # model_name = f"{TODAY}-{model_type}_{split_type}_{desc_prot}_{desc_chem}-{wandb.run.name}"

        pt_path = model_dir / f"{model_name}.pt"
        # check if path exists
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
            # in case we have two inputs ?
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

        # Model logging if wandb
        if tracker.lower() == "wandb":
            wandb.save(wandb_model_path, base_path=model_dir)

    except Exception as e:
        print("Error saving models: " + str(e))


def add_prefix_to_state_dict_keys(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    """
    Adds a prefix to all keys in a model's state dictionary.

    Parameters:
    -----------
    state_dict : Dict[str, torch.Tensor]
        The original state dictionary.
    prefix : str
        The prefix to append to each key.

    Returns:
    --------
    Dict[str, torch.Tensor]
        The updated state dictionary with modified keys.
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key if key.startswith(prefix) else f"{prefix}{key}"
        # new_key = prefix + key
        new_state_dict[new_key] = value
    return new_state_dict


def load_model(
    model_class: nn.Module,
    model_path: str,
    prefix_to_state_keys: Optional[str] = None,
    **model_kwargs,
) -> nn.Module:
    """
    Loads a PyTorch model from a saved state dictionary.

    Parameters:
    -----------
    model_class : nn.Module
        The class of the model to instantiate.
    model_path : str
        Path to the saved state dictionary.
    prefix_to_state_keys : str, optional
        Prefix to prepend to state dictionary keys if needed.
    **model_kwargs : dict
        Additional arguments for initializing the model.

    Returns:
    --------
    nn.Module
        The model instance with loaded weights.
    """
    # Initialize the model
    model = model_class(**model_kwargs)

    # Load the state dictionary from the specified path
    state_dict = torch.load(model_path)

    if prefix_to_state_keys:
        state_dict = add_prefix_to_state_dict_keys(state_dict, prefix_to_state_keys)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    return model


def get_ckpt_path(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates and returns the path for model checkpoint storage.

    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing model details.

    Returns:
    --------
    Dict[str, Any]
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
    # we need to create the file here as placeholder
    model_path.touch()

    config["ckpt_name"] = model_path.stem
    config["ckpt_path"] = model_path

    return config


def ckpt(model: nn.Module, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Saves a model checkpoint to disk.

    Parameters:
    -----------
    model : nn.Module
        The model whose state should be saved.
    config : Dict[str, Any]
        Configuration dictionary containing checkpoint details.

    Returns:
    --------
    Dict[str, Any]
        Updated configuration dictionary with checkpoint path.
    """
    ckpt_path = config.get("ckpt_path", None)
    if not ckpt_path:  # First time saving
        config = get_ckpt_path(config)
        ckpt_path = config["ckpt_path"]
    torch.save(model.state_dict(), ckpt_path)
    return config


def load_ckpt(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """
    Loads a model checkpoint from disk.

    Parameters:
    -----------
    model : nn.Module
        The model to which weights should be loaded.
    config : Dict[str, Any]
        Configuration dictionary containing checkpoint path.

    Returns:
    --------
    nn.Module
        The model with restored weights.
    """
    ckpt_path = config.get("ckpt_path", None)
    ckpt_name = config.get("ckpt_name", None)
    if ckpt_path:
        print(f"Loading model {ckpt_name} from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
    return model


def get_model_name(config: Dict[str, Any], run: Optional[wandb.run] = None) -> str:
    """
    Generates a model name based on configuration settings.

    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing model details.
    run : wandb.run, optional
        The W&B run object for naming the model.

    Returns:
    --------
    str
        The generated model name.
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

    # check if run and the name doesnt end with run.name already
    if run and not model_name.endswith(run.name):
        model_name += f"_{run.name}"

    return model_name


def get_data_specific_path(
    config: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> Path:
    """
    Constructs a data-specific directory path for model storage.

    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing dataset details.
    logger : logging.Logger, optional
        Logger instance for debugging (default: None).

    Returns:
    --------
    Path
        The generated directory path for storing dataset-specific results.
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
    Computes the mean along the last dimension for multiple tensors.

    Parameters:
    -----------
    tensors : torch.Tensor
        One or more tensors to compute the mean over.

    Returns:
    --------
    List[torch.Tensor]
        A list containing the mean-reduced tensors.
    """
    return [torch.mean(tensor, dim=2) for tensor in tensors]
