import numpy as np
import wandb
import torch
import torch.nn as nn

from uqdd import DEVICE, WANDB_DIR, WANDB_MODE
from uqdd.models.evidential import EvidentialDNN, ev_predict
from uqdd.models.ensemble import process_results_arrs
from uqdd.models.mcdropout import enable_dropout
from uqdd.utils import create_logger

from uqdd.models.utils_train import (
    train_model_e2e,
    evaluate_predictions,
    recalibrate_model,
    assign_wandb_tags,
    get_dataloader,
    post_training_save_model,
)

from uqdd.models.utils_models import get_model_config, set_seed, calculate_means
from tqdm import tqdm


def emc_predict(ev_model, dataloader, num_mc_samples=10, device=DEVICE):
    outputs_all, aleatoric_all, epistemic_all = [], [], []  # targets_all = []
    for _ in range(num_mc_samples):
        ev_model.eval()
        enable_dropout(ev_model)
        preds, labels, alea, epistemic = ev_predict(
            ev_model, dataloader, device=device, set_on_eval=False
        )
        outputs_all.append(preds)
        aleatoric_all.append(alea)
        epistemic_all.append(epistemic)

    outputs_all = torch.stack(outputs_all, dim=2)  # .mean(dim=2)
    aleatoric_all = torch.stack(aleatoric_all, dim=2)  # .mean(dim=2)
    epistemic_all = torch.stack(epistemic_all, dim=2)  # .mean(dim=2)

    return outputs_all.cpu(), labels.cpu(), aleatoric_all.cpu(), epistemic_all.cpu()


def run_emc(config=None):
    logger = LOGGER
    num_mc_samples = config.get("num_mc_samples", 10)

    # Train EvDNN
    best_model, config, _, _ = train_model_e2e(
        config,
        model=EvidentialDNN,
        model_type="emc",
        logger=LOGGER,
    )
    dataloaders = get_dataloader(config, device=DEVICE, logger=logger)

    preds, labels, alea_vars, epi_vars = emc_predict(
        best_model, dataloaders["test"], num_mc_samples=num_mc_samples, device=DEVICE
    )
    preds, alea_vars, epi_vars = calculate_means(preds, alea_vars, epi_vars)

    # Then comes the predict metrics part
    metrics, plots, uct_logger = evaluate_predictions(
        config,
        preds,
        labels,
        alea_vars,
        "emc",
        LOGGER,
        epi_vars,
        wandb_push=False,
    )

    # RECALIBRATION
    preds_val, labels_val, alea_vars_val, epi_vars_val = emc_predict(
        best_model, dataloaders["val"], num_mc_samples=num_mc_samples, device=DEVICE
    )
    preds_val, alea_vars_val, epi_vars_val = calculate_means(
        preds_val, alea_vars_val, epi_vars_val
    )
    iso_recal_model = recalibrate_model(
        preds_val,
        labels_val,
        alea_vars_val,
        preds,
        labels,
        alea_vars,
        config=config,
        epi_val=epi_vars_val,
        epi_test=epi_vars,
        uct_logger=uct_logger,
    )

    uct_logger.wandb_log()
    wandb.finish()

    return best_model, iso_recal_model, metrics, plots


def run_emc_wrapper(**kwargs):
    global LOGGER
    LOGGER = create_logger(name="emc", file_level="debug", stream_level="info")
    config = get_model_config(model_type="emc", **kwargs)
    return run_emc(config=config)


# if __name__ == "__main__":
#     run_emc_wrapper(
#         data_name="papyrus",
#         activity_type="xc50",
#         n_targets=-1,
#         descriptor_protein="ankh-large",
#         descriptor_chemical="ecfp2048",
#         median_scaling=False,
#         # split_type="scaffold_cluster",
#         split_type="random",
#         ext="pkl",
#         task_type="regression",
#         wandb_project_name="emc-test",
#         epochs=5,
#         num_mc_samples=5,
#     )
#     print("done")

# def emc_predict(model, dataloader, num_mc_samples=10, device=DEVICE):
#     model.train()  # Enable dropout
#     outputs_all = []
#     targets_all = []
#     aleatoric_all = []
#     ev_epistemic_all = []
#
#     # Here we need to selectively set batchnorm layers to eval mode for not affecting the running mean
#     for m in model.modules():
#         if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
#             m.eval()
#
#     with torch.no_grad():
#         for _ in range(num_mc_samples):
#             outputs, targets, alea, ev_epistemic = ev_predict(
#                 model, dataloader, device=device, set_on_eval=False
#             )
#             outputs_all.append(outputs)
#             targets_all.append(targets)
#             aleatoric_all.append(alea)
#             ev_epistemic_all.append(ev_epistemic)
#
#     # stack on dim 2
#     outputs_all = torch.stack(outputs_all, dim=2)
#     targets_all = torch.stack(targets_all, dim=2)
#     aleatoric_all = torch.stack(aleatoric_all, dim=2)
#     ev_epistemic_all = torch.stack(ev_epistemic_all, dim=2)
#
#     return (
#         outputs_all.cpu(),
#         targets_all.cpu(),
#         aleatoric_all.cpu(),
#         ev_epistemic_all.cpu(),
#     )
