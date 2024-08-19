import numpy as np
import wandb
import torch
import torch.nn as nn

from uqdd import DEVICE, WANDB_DIR, WANDB_MODE
from uqdd.models.evidential import EvidentialDNN, ev_predict
from uqdd.models.ensemble import process_results_arrs
from uqdd.utils import create_logger

from uqdd.models.utils_train import (
    train_model_e2e,
    evaluate_predictions,
    recalibrate_model,
    assign_wandb_tags,
    get_dataloader,
    post_training_save_model,
)

from uqdd.models.utils_models import get_model_config


def eoe_predict(eoe_model_list, dataloader, device=DEVICE):
    outputs_list = []
    targets_list = []
    alea_list = []
    epistemic_list = []
    for model in eoe_model_list:
        outputs, targets, alea, epistemic = ev_predict(model, dataloader, device)
        outputs_list.append(outputs)
        targets_list.append(targets)
        alea_list.append(alea)
        epistemic_list.append(epistemic)

    outputs = torch.stack(outputs_list, dim=2)
    targets = torch.stack(targets_list, dim=2)
    alea = torch.stack(alea_list, dim=2)
    epistemic = torch.stack(epistemic_list, dim=2)

    return outputs, targets, alea, epistemic


def run_eoe(config=None):
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
            model_type="evidential",
            logger=logger,
            tracker="tensor",
            write_model=False,
        )
        best_models.append(best_model)
        config["seed"] += 1

        result_arrs.append(results_arr)
        test_arrs.append(test_arr)

    res_arr = process_results_arrs(result_arrs, test_arrs, config_, logger, "eoe")

    logger.debug(f"{len(best_models)=}")
    eoe_model = nn.ModuleList(best_models)
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

    preds, labels, alea, epistemic = eoe_predict(
        eoe_model, dataloaders["test"], device=DEVICE
    )

    alea_mean = torch.mean(alea, dim=2)
    epistemic_mean = torch.mean(epistemic, dim=2)
    preds_mean = torch.mean(preds, dim=2)
    labels_mean = torch.mean(labels, dim=2)

    metrics, plots, uct_logger = evaluate_predictions(
        config_,
        preds_mean,
        labels_mean,
        alea_mean,
        "eoe",
        logger,
        epistemic_mean,
        wandb_push=False,
        verbose=True,
    )

    preds_val, labels_val, alea_val, epistemic_val = eoe_predict(
        eoe_model, dataloaders["val"], device=DEVICE
    )

    alea_mean_val = torch.mean(alea_val, dim=2)
    epistemic_mean_val = torch.mean(epistemic_val, dim=2)
    preds_mean_val = torch.mean(preds_val, dim=2)
    labels_mean_val = torch.mean(labels_val, dim=2)

    iso_recal_model = recalibrate_model(
        preds_mean_val,
        labels_mean_val,
        alea_mean_val,
        preds_mean,
        labels_mean,
        alea_mean,
        config=config_,
        epi_val=epistemic_mean_val,
        epi_test=epistemic_mean,
        uct_logger=uct_logger,
    )

    uct_logger.wandb_log()
    wandb.finish()

    return eoe_model, iso_recal_model, metrics, plots


def run_eoe_wrapper(**kwargs):
    global LOGGER
    LOGGER = create_logger(name="eoe", file_level="debug", stream_level="info")
    config = get_model_config(model_type="eoe", **kwargs)
    return run_eoe(config=config)


if __name__ == "__main__":
    # vars
    eoe_model, iso_recal_model, metrics, plots = run_eoe_wrapper(
        data_name="papyrus",
        activity_type="xc50",
        n_targets=-1,
        descriptor_protein="ankh-large",
        descriptor_chemical="ecfp2048",
        median_scaling=False,
        split_type="time",
        ext="pkl",
        task_type="regression",
        wandb_project_name="eoe-test",
        ensemble_size=5,
        epochs=5,
        seed=440,
    )


# class EoEDNN(nn.Module):
#     def __init__(self, config=None, model_list=None, **kwargs):
#         super(EoEDNN, self).__init__()
#         self.model_list = model_list
#         self.config = config
#         self.device = DEVICE
#         self.ensemble_size = config.get("ensemble_size", 10)
#         self.model_type = config.get("model_type", "eoe")
#         self.logger = config.get("logger", None)
#         self.tracker = config.get("tracker", "tensor")
#
#         if model_list is not None:
#             models = model_list
#         else:
#             models = []
#             seed = config.get("seed", 42)
#             for _ in range(self.ensemble_size):
#                 model = EvidentialDNN(config=config, **kwargs)
#                 model.to(self.device)
#                 set_seed(seed)
#                 models.append(model)
#                 seed += 1
#         self.models = nn.ModuleList(models)
#
#     def forward(self, inputs):
#         outputs = []
#         for model in self.models:
#             output = model(inputs)
#             outputs.append(output)
#         outputs = torch.stack(outputs, dim=2)
#         return outputs
