import logging
from typing import Optional, Tuple, List

import wandb
import torch
import torch.nn as nn
from uqdd.utils import create_logger
from uqdd.models.utils_models import (
    get_model_config,
    get_sweep_config,
)
from uqdd.models.utils_train import train_model_e2e, predict


class PNN(nn.Module):
    """
    A Probabilistic neural network (PNN) for regression or classification tasks using
    chemical and (optionally) protein descriptors as input.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary containing model hyperparameters.
    logger : logging.Logger, optional
        Logger instance for debugging and info logging.
    aleavar_layer_included : bool, optional
        Whether to include an aleatoric uncertainty estimation layer (default: True).
    **kwargs : dict
        Additional keyword arguments for model configuration.

    Attributes
    ----------
    config : dict
        Stores model hyperparameters.
    task_type : str
        Type of task ('regression' or 'classification').
    output_dim : int
        Dimension of the model's output layer.
    aleatoric : bool
        Whether aleatoric uncertainty estimation is enabled.
    chem_feature_extractor : nn.Sequential
        MLP feature extractor for chemical descriptors.
    prot_feature_extractor : Optional[nn.Sequential]
        MLP feature extractor for protein descriptors (only for single-task learning).
    regressor_or_classifier : nn.Sequential
        MLP layers for regression or classification.
    aleavar_layer : Optional[nn.Sequential]
        Layer to estimate aleatoric uncertainty (if enabled).
    output_layer : nn.Linear
        Final layer mapping to output predictions.

    Methods
    -------
    forward(inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        Performs forward pass through the model.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
        aleavar_layer_included: bool = True,
        **kwargs,
    ) -> None:
        super(PNN, self).__init__()
        if config is None:
            config = get_model_config(model_type="pnn", **kwargs)
        self.config = config

        chem_input_dim = config.get("chem_input_dim", None)
        prot_input_dim = config.get("prot_input_dim", None)
        task_type = config.get("task_type", "regression")
        n_targets = config.get("n_targets", -1)
        self.MT = config.get("MT", n_targets > 1)
        self.MT = False if type(self.MT) != bool else self.MT

        self.aleatoric = config.get("aleatoric", False)
        self.aleavar_layer_included = aleavar_layer_included
        assert task_type in [
            "regression",
            "classification",
        ], "task_type must be either 'regression' or 'classification'"

        self.task_type = task_type
        # memory placeholders
        self.prot_feature_extractor = None
        self.chem_feature_extractor = None
        self.regressor_or_classifier = None
        self.aleavar_layer = None
        self.output_layer = None
        self.logger = (
            create_logger(name="pnn", file_level="debug", stream_level="info")
            if not logger
            else logger
        )

        n_targets = 1 if not self.MT else n_targets
        # active inactive per each target if classification
        self.output_dim = n_targets if task_type == "regression" else 2 * n_targets

        # Initialize feature extractors
        self.init_layers(config, chem_input_dim, prot_input_dim, self.output_dim)

        self.apply(self.init_wt)

    @staticmethod
    def init_wt(module: nn.Module) -> None:
        """
        Initializes the weights of the given module using Xavier normal initialization.

        Parameters
        ----------
        module : nn.Module
            The module whose weights need to be initialized.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("relu"))

    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Parameters
        ----------
        inputs : tuple of torch.Tensor
            Tuple containing protein and chemical descriptors as tensors.

        Returns
        -------
        output : torch.Tensor
            Model predictions (e.g., activity scores or class logits).
        var_ : torch.Tensor or None
            Aleatoric uncertainty estimates (if enabled), otherwise None.
        """
        prot_input, chem_input = inputs
        chem_features = self.chem_feature_extractor(chem_input)
        if not self.MT:
            prot_features = self.prot_feature_extractor(prot_input)
            combined_features = torch.cat((chem_features, prot_features), dim=1)
        else:
            combined_features = chem_features

        _output = self.regressor_or_classifier(combined_features)
        output = self.output_layer(_output)
        var_ = (
            self.aleavar_layer(_output)
            if self.aleatoric and self.aleavar_layer_included
            else None
        )

        return output, var_

    @staticmethod
    def create_mlp(
        input_dim: int, layer_dims: List[int], dropout: float
    ) -> nn.Sequential:
        """
        Creates a multi-layer perceptron (MLP) with ReLU activations and dropout.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        layer_dims : list of int
            List of layer dimensions.
        dropout : float
            Dropout rate.

        Returns
        -------
        nn.Sequential
            A PyTorch sequential model.
        """
        modules = []
        for i in range(len(layer_dims)):
            if i == 0:
                modules.append(nn.Linear(input_dim, layer_dims[i]))
            else:
                modules.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
                modules.append(nn.BatchNorm1d(layer_dims[i]))  # Add batch normalization
            modules.append(nn.ReLU())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
        return nn.Sequential(*modules)  # , layer_dims[-1]

    def init_layers(
        self,
        config: dict,
        chem_input_dim: Optional[int],
        prot_input_dim: Optional[int],
        output_dim: int,
    ) -> None:
        """
        Initializes the feature extractors and regressor/classifier layers.

        Parameters
        ----------
        config : dict
            Model configuration dictionary.
        chem_input_dim : Optional[int]
            Input dimension for chemical features.
        prot_input_dim : Optional[int]
            Input dimension for protein features (None for multitask learning).
        output_dim : int
            Output dimension for the model.
        """
        # Chemical feature extractor
        chem_layers = config["chem_layers"]
        self.chem_feature_extractor = self.create_mlp(
            chem_input_dim, chem_layers, config["dropout"]
        )
        self.logger.debug(
            f"Chemical feature extractor: {chem_input_dim} -> {chem_layers}"
        )

        if not self.MT:
            # Protein feature extractor (only for single-task learning)
            prot_layers = config["prot_layers"]
            self.prot_feature_extractor = self.create_mlp(
                prot_input_dim, prot_layers, config["dropout"]
            )
            self.logger.debug(
                f"Protein feature extractor: {prot_input_dim} -> {prot_layers}"
            )

            # Combined input dimension for STL
            chem_dim = config["chem_layers"][-1]
            prot_dim = config["prot_layers"][-1]
            combined_input_dim = chem_dim + prot_dim

        else:
            # Only chemical features for MTL
            combined_input_dim = config["chem_layers"][-1]

        self.logger.debug(f"Combined input dimension: {combined_input_dim}")
        regressor_layers = config["regressor_layers"]
        self.regressor_or_classifier = self.create_mlp(
            combined_input_dim, regressor_layers, config["dropout"]
        )

        self.logger.debug(f"Regressor layers: {regressor_layers}")
        self.logger.debug(f"Output dimension: {output_dim}")
        self.output_layer = nn.Linear(regressor_layers[-1], output_dim)
        if self.aleatoric and self.aleavar_layer_included:
            self.aleavar_layer = nn.Sequential(
                nn.Linear(regressor_layers[-1], output_dim),
                nn.Softplus(),  # TODO questionable
            )


def run_pnn(config: Optional[dict] = None) -> nn.Module:  # uq: bool = False
    """
    Runs training for the PNN model and returns the trained model.

    Parameters
    ----------
    config : Optional[dict], optional
        Configuration dictionary for model training, by default None.

    uq : bool
        Whether to use aleatoric uncertainty estimation, by default

    Returns
    -------
    nn.Module
        Trained PNN model.
    """
    best_model, config, _, _ = train_model_e2e(
        config,
        model=PNN,
        model_type="pnn",
        logger=LOGGER,
    )

    # if uq:
    #     dataloaders = get_dataloader(config, device=DEVICE, logger=LOGGER)
    #     preds, labels, alea_vars = predict(
    #         best_model, dataloaders["test"], device=DEVICE
    #     )
    #     # Then comes the predict metrics part
    #     metrics, plots, uct_logger = evaluate_predictions(
    #         config, preds, labels, alea_vars, "pnn", LOGGER
    #     )
    #     # RECALIBRATION # Get Calibration / Validation Set
    #     preds_val, labels_val, alea_vars_val = predict(
    #         best_model, dataloaders["val"], device=DEVICE
    #     )
    #     iso_recal_model = recalibrate_model(
    #         preds_val,
    #         labels_val,
    #         alea_vars_val,
    #         preds,
    #         labels,
    #         alea_vars,
    #         config=config,
    #         uct_logger=uct_logger,
    #     )
    #     uct_logger.wandb_log()
    #     wandb.finish()
    #     return best_model, iso_recal_model, metrics, plots

    return best_model


def run_pnn_wrapper(**kwargs) -> nn.Module:
    """
    Wrapper function for running the pnn model with logging.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for model configuration.

    Returns
    -------
    nn.Module
        Trained pnn model.
    """
    global LOGGER
    LOGGER = create_logger("pnn", file_level="debug", stream_level="info")

    config = get_model_config("pnn", **kwargs)
    return run_pnn(config=config)


def run_pnn_hyperparam(**kwargs) -> None:
    """
    Runs a hyperparameter sweep for the pnn model using Weights & Biases.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including:
        - `sweep_count` (int): Number of sweep runs.
        - `wandb_project_name` (str): Name of the Weights & Biases project.

    Returns
    -------
    None
    """
    global LOGGER
    LOGGER = create_logger(name="pnn-sweep", file_level="debug", stream_level="info")

    sweep_count = kwargs.pop("sweep_count")
    wandb_project_name = kwargs.pop("wandb_project_name")
    config = get_sweep_config("pnn", **kwargs, wandb_project_name=wandb_project_name)
    config["project"] = wandb_project_name

    sweep_id = wandb.sweep(
        config,
        project=wandb_project_name,
    )
    print(f"Running sweep with SWEEP_ID: {sweep_id}")

    wandb.agent(sweep_id, function=run_pnn, count=sweep_count)


# if __name__ == "__main__":
#     data_name = "papyrus"
#     n_targets = -1
#     task_type = "regression"
#     activity = "xc50"
#     split = "time"
#     desc_prot = "ankh-large"
#     desc_chem = "ecfp2048"
#     median_scaling = False
#     ext = "pkl"
#     wandb_project_name = "pnn-test"
#     sweep_count = 0  # 250
#     aleatoric = True
#     # epochs=1
#     #
#     run_pnn_wrapper(
#         data_name=data_name,
#         activity_type=activity,
#         n_targets=n_targets,
#         descriptor_protein=desc_prot,
#         descriptor_chemical=desc_chem,
#         # median_scaling=median_scaling,
#         split_type=split,
#         aleatoric=aleatoric,
#         ext=ext,
#         task_type=task_type,
#         wandb_project_name=wandb_project_name,
#         logger=None,
#         epochs=5,
#     )
#     #
#     sweep_count = 5
#     epochs = 5
#     run_pnn_hyperparam(
#         sweep_count=sweep_count,
#         epochs=epochs,
#         data_name=data_name,
#         activity_type=activity,
#         n_targets=n_targets,
#         descriptor_protein=desc_prot,
#         descriptor_chemical=desc_chem,
#         split_type=split,
#         ext=ext,
#         task_type=task_type,
#         wandb_project_name=wandb_project_name,
#     )
#     print("Done")
