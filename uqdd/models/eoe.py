from typing import Tuple, Optional, Dict, Any, List

import torch
import torch.nn as nn
import wandb

from uqdd import DEVICE, WANDB_DIR, WANDB_MODE
from uqdd.models.ensemble import process_results_arrs
from uqdd.models.evidential import (
    EvidentialDNN,
    ev_predict,
    ev_nll,
)
from uqdd.models.utils_models import (
    get_model_config,
    set_seed,
    calculate_means,
    stack_vars,
)
from uqdd.models.utils_train import (
    train_model_e2e,
    evaluate_predictions,
    recalibrate_model,
    assign_wandb_tags,
    get_dataloader,
    post_training_save_model,
)
from uqdd.utils import create_logger


class EoEDNN(nn.Module):
    """
    Evidential Deep Ensemble (EoE) Model.

    This model consists of multiple `EvidentialDNN` models, forming an ensemble to
    improve prediction uncertainty estimation.

    Parameters
    ----------
    config : Optional[Dict[str, Any]], optional
        Configuration dictionary containing model hyperparameters, by default None.
    model_list : Optional[List[nn.Module]], optional
        Pretrained list of `EvidentialDNN` models for ensemble construction, by default None.
    **kwargs
        Additional parameters for initializing individual `EvidentialDNN` models.
    """

    def __init__(
            self,
            config: Optional[Dict[str, Any]] = None,
            model_list: Optional[List[nn.Module]] = None,
            **kwargs,
    ):
        super(EoEDNN, self).__init__()
        self.model_list = model_list
        self.config = config
        self.device = DEVICE
        self.ensemble_size = config.get("ensemble_size", 10)
        self.model_type = config.get("model_type", "eoe")
        self.logger = config.get("logger", None)
        self.tracker = config.get("tracker", "tensor")

        if model_list is not None:
            models = model_list
        else:
            models = []
            seed = config.get("seed", 42)
            for _ in range(self.ensemble_size):
                model = EvidentialDNN(config=config, **kwargs)
                model.to(self.device)
                set_seed(seed)
                models.append(model)
                seed += 1
        self.models = nn.ModuleList(models)

    def forward(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the ensemble of evidential neural networks.

        Parameters
        ----------
        inputs : Tuple[torch.Tensor, torch.Tensor]
            Tuple containing (protein features, chemical features).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Returns the stacked ensemble predictions for:
            - Mean `mu`
            - Variance `v`
            - Alpha parameter `alpha`
            - Beta parameter `beta`
        """
        # outputs = []
        mus, vs, alphas, betas = [], [], [], []
        for model in self.models:
            output = model(inputs)
            mu, v, alpha, beta = output
            mus.append(mu)
            vs.append(v)
            alphas.append(alpha)
            betas.append(beta)

        mus, vs, alphas, betas = stack_vars(mus, vs, alphas, betas)
        mus, vs, alphas, betas = calculate_means(mus, vs, alphas, betas)

        return mus, vs, alphas, betas


def run_eoe(
        config: Optional[Dict[str, Any]] = None
) -> Tuple[nn.Module, Any, Dict[str, Any], Dict[str, Any]]:
    """
    Train an ensemble of Evidential Neural Networks (EoE) and perform uncertainty quantification.

    Parameters
    ----------
    config : Optional[Dict[str, Any]], optional
        Configuration dictionary containing training hyperparameters, by default None.

    Returns
    -------
    Tuple[EoEDNN, Any, Dict[str, Any], Dict[str, Any]]
        - Trained `EoEDNN` model.
        - Isotonic recalibration model.
        - Dictionary containing evaluation metrics.
        - Dictionary containing generated plots.
    """
    ensemble_size = config.get("ensemble_size", 10)
    logger = LOGGER
    best_models = []
    result_arrs = []
    test_arrs = []

    # start wandb run
    run = wandb.init(
        config=config,
        dir=WANDB_DIR,
        mode=WANDB_MODE,
        project=config.get("wandb_project_name", "eoe_test"),
        reinit=True,
    )
    assign_wandb_tags(run, config)
    for _ in range(ensemble_size):
        best_model, config_, results_arr, test_arr = train_model_e2e(
            config,
            model=EvidentialDNN,
            model_type="eoe",
            logger=logger,
            tracker="tensor",
            write_model=False,
        )
        best_models.append(best_model)
        config["seed"] += 1

        result_arrs.append(results_arr)
        test_arrs.append(test_arr)
    # print(f"{config_=}")
    res_arr = process_results_arrs(result_arrs, test_arrs, config_, logger, "eoe")
    logger.debug(f"{len(best_models)=}")
    eoe_model = EoEDNN(config=config, model_list=best_models).to(DEVICE)
    # eoe_model = nn.ModuleList(best_models)
    config_["model_name"] = post_training_save_model(
        eoe_model,
        config,
        model_type="eoe",
        tracker="wandb",
        run=run,
        logger=logger,
        write_model=True,
    )

    dataloaders = get_dataloader(config, device=DEVICE, logger=LOGGER)

    preds, labels, alea_vars, epist_var = ev_predict(
        eoe_model, dataloaders["test"], device=DEVICE
    )
    nll = ev_nll(eoe_model, dataloaders["test"], device=DEVICE)

    # alea_vars, epi_vars, preds = calculate_means(alea_vars, epi_vars, preds)

    metrics, plots, uct_logger = evaluate_predictions(
        config_,
        preds,
        labels,
        alea_vars,
        "eoe",
        logger,
        epist_var,
        wandb_push=False,
        verbose=True,
        nll=nll,
    )

    preds_val, labels_val, alea_vars_val, epi_vars_val = ev_predict(
        eoe_model, dataloaders["val"], device=DEVICE
    )
    nll = ev_nll(eoe_model, dataloaders["val"], device=DEVICE)
    # alea_vars_val, epi_vars_val, preds_val = calculate_means(
    #     alea_vars_val, epi_vars_val, preds_val
    # )

    iso_recal_model = recalibrate_model(
        preds_val,
        labels_val,
        alea_vars_val,
        preds,
        labels,
        alea_vars,
        config=config_,
        epi_val=epi_vars_val,
        epi_test=epist_var,
        uct_logger=uct_logger,
        nll=nll,  # TODO calculate nll inside recalibrate model before and after recalibration
    )

    uct_logger.wandb_log()
    wandb.finish()

    return eoe_model, iso_recal_model, metrics, plots


def run_eoe_wrapper(**kwargs):
    """
    Wrapper function to initialize and train an Evidential Deep Ensemble (EoE).

    Parameters
    ----------
    **kwargs
        Additional configuration parameters.

    Returns
    -------
    Tuple[EoEDNN, Any, Dict[str, Any], Dict[str, Any]]
        - Trained `EoEDNN` model.
        - Isotonic recalibration model.
        - Dictionary containing evaluation metrics.
        - Dictionary containing generated plots.
    """
    global LOGGER
    LOGGER = create_logger(name="eoe", file_level="debug", stream_level="info")
    config = get_model_config(model_type="eoe", **kwargs)
    return run_eoe(config=config)


if __name__ == "__main__":
    # vars
    eoe_model, iso_recal_model, metrics, plots = run_eoe_wrapper(
        data_name="papyrus",
        activity_type="kx",
        n_targets=-1,
        descriptor_protein="ankh-large",
        descriptor_chemical="ecfp2048",
        median_scaling=False,
        split_type="random",
        ext="pkl",
        task_type="regression",
        wandb_project_name="eoe-test",
        ensemble_size=5,
        epochs=5,
        seed=42,
    )
