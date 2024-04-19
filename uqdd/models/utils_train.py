from datetime import datetime
from pathlib import Path

import wandb
import torch
from torch import nn

from tqdm import tqdm
from uqdd import DEVICE, WANDB_DIR, WANDB_MODE, TODAY, FIGS_DIR, MODELS_DIR
from uqdd.data.utils_data import get_topx, get_tasks, export_pickle

from uqdd.models.loss import build_loss
from uqdd.models.utils_metrics import (
    calc_regr_metrics,
    MetricsTable,
    process_preds,
    create_df_preds,
    calc_aleatoric_mean_var_notnan,
    recalibrate,
)
from uqdd.models.utils_models import (
    build_loader,
    build_optimizer,
    build_lr_scheduler,
    save_model,
    set_seed,
    build_datasets,
    get_desc_len,
    compute_pnorm,
    compute_gnorm
)
from uqdd.utils import create_logger


def evidential_processing(outputs, alea_all):
    if len(outputs) == 4:  # Evidential model
        # mu, v, alpha, beta = (d.squeeze() for d in outputs)
        mu, v, alpha, beta = outputs
        vars_ = beta / (alpha - 1)  # aleatoric
        # var = torch.sqrt(beta / (v * (alpha - 1))) # epistemic
        alea_all.append(vars_)
        outputs = mu

        # return outputs, alea_all
        # TODO how to get the epistemic out of this function
    return outputs, alea_all


# def apply_model_aleatoric_option()
def train(
    model, dataloader, loss_fn, optimizer, aleatoric, device=DEVICE, epoch=0, max_norm=10.0  # pbar=None,
):
    # max_norm = 10.0
    model.train()
    total_loss = 0.0
    targets_all = []
    outputs_all = []
    vars_all = []

    # size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for inputs, targets in dataloader:
        inputs = tuple(x.to(device) for x in inputs)
        targets = targets.to(device)
        optimizer.zero_grad()
        if aleatoric:
            outputs, vars_ = model(inputs)
            # vars_ = torch.exp(logvars)
            loss = loss_fn(outputs, targets, vars_)
            vars_all.append(vars_)
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()
        total_loss += loss.item()  # * outputs.size(0)
        outputs, vars_all = evidential_processing(outputs, vars_all)
        targets_all.append(targets)
        outputs_all.append(outputs)

    total_loss /= num_batches  # len(dataloader.dataset)
    targets_all = torch.cat(targets_all, dim=0)
    outputs_all = torch.cat(outputs_all, dim=0)

    # Calculate metrics
    train_rmse, train_r2, train_evs = calc_regr_metrics(
        targets_all, outputs_all
    )
    pnorm = compute_pnorm(model)
    gnorm = compute_gnorm(model)

    wandb.log(
        data={
            # "epoch": epoch,
            "train/loss": total_loss,
            "train/rmse": train_rmse,
            "train/r2": train_r2,
            "train/evs": train_evs,
            "model/pnorm": pnorm,
            "model/gnorm": gnorm,
        },
        step=epoch,
    )
    if aleatoric:
        # vars_all = torch.exp(torch.stack(vars_all))
        # vars_mean = torch.mean(vars_all, dim=0)
        # vars_var = torch.var(vars_all, dim=0)

        vars_all = torch.cat(vars_all, dim=0)
        vars_mean, vars_var = calc_aleatoric_mean_var_notnan(vars_all, targets_all)
        wandb.log(
            data={
                "train/alea_mean": vars_mean.item(),
                "train/alea_var": vars_var.item(),
            },
            step=epoch,
        )
    return total_loss


def evaluate(
        model,
        dataloader,
        loss_fn,
        aleatoric=False,
        device=DEVICE,
        # pbar=None,
        metrics_per_task=False,
        subset="val",  # can be "train", "val" or "test"
        epoch=0,
):
    model.eval()
    total_loss = 0.0
    targets_all = []
    outputs_all = []
    vars_all = []

    num_batches = len(dataloader)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = tuple(x.to(device) for x in inputs)
            targets = targets.to(device)

            if aleatoric:
                outputs, vars_ = model(inputs)
                # vars_ = torch.exp(logvars)
                loss = loss_fn(outputs, targets, vars_)
                vars_all.append(vars_)
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            total_loss += loss.item()  # * outputs.size(0)
            outputs, vars_all = evidential_processing(outputs, vars_all)
            targets_all.append(targets)
            outputs_all.append(outputs)

        total_loss /= num_batches  # len(dataloader.dataset)
        targets_all = torch.cat(targets_all, dim=0)
        outputs_all = torch.cat(outputs_all, dim=0)

        # Calculate metrics
        rmse, r2, evs = calc_regr_metrics(targets_all, outputs_all)

        wandb.log(
            data={
                f"{subset}/loss": total_loss,
                f"{subset}/rmse": rmse,
                f"{subset}/r2": r2,
                f"{subset}/evs": evs,
            },
            step=epoch,
        )
        # Calculate metrics
        if metrics_per_task:
            tasks_rmse, tasks_r2, tasks_evs = calc_regr_metrics(
                targets_all, outputs_all, metrics_per_task
            )
            for task_idx in range(len(tasks_rmse)):
                wandb.log(
                    data={
                        f"{subset}/rmse/task_{task_idx}": tasks_rmse[task_idx],
                        f"{subset}/r2/task_{task_idx}": tasks_r2[task_idx],
                        f"{subset}/evs/task_{task_idx}": tasks_evs[task_idx],
                    },
                    step=epoch,
                )
        # Aleatoric Uncertainty
        if aleatoric:
            vars_all = torch.cat(vars_all, dim=0)
            vars_mean, vars_var = calc_aleatoric_mean_var_notnan(vars_all, targets_all)
            # vars_all = torch.cat(vars_all, dim=0)
            # # vars_all = torch.exp(torch.cat(logvars_all, dim=0))
            # vars_mean = torch.mean(vars_all)
            # vars_var = torch.var(vars_all)
            wandb.log(
                data={
                    f"{subset}/alea_mean": vars_mean.item(),
                    f"{subset}/alea_var": vars_var.item(),
                },
                step=epoch,
            )

    return total_loss


def predict(
        model,
        dataloader,
        aleatoric=False,
        device=DEVICE,
):
    model.eval()
    outputs_all = []
    targets_all = []
    vars_all = []

    with torch.no_grad():
        for inputs, targets in tqdm(
                dataloader, total=len(dataloader), desc="Predicting"
        ):
            inputs = tuple(x.to(device) for x in inputs)
            if aleatoric:
                outputs, vars_ = model(inputs)
                # vars_ = torch.exp(logvars)
                vars_all.append(vars_)
            else:
                outputs = model(inputs)
            outputs_all.append(outputs)
            targets_all.append(targets)

    outputs_all = torch.cat(outputs_all, dim=0).cpu()
    targets_all = torch.cat(targets_all, dim=0).cpu()

    if aleatoric:
        vars_all = torch.cat(vars_all, dim=0).cpu()
        return outputs_all, targets_all, vars_all

    return outputs_all, targets_all, None

#
# def _predict(
#         model,
#         dataloader,
#         aleatoric=False,
#         # evidential=False,
#         # num_mc_samples=1, # Default to 1 to mimic standard predict behavior, otherwise MC
#         device=DEVICE,
# ):
#     # Determine mode based on num_samples
#     # if num_mc_samples == 1:
#     model.eval()  # Standard evaluation mode for prediction
#     #     desc = "Predicting"
#     # else:
#     #     model.train()  # Enable dropout for MC prediction
#     #     desc = f"MC Dropout Prediction ({num_mc_samples})"
#
#     outputs_all = []
#     targets_all = []
#     aleatoric_all = []
#     epistemic_all = []
#
#     with torch.no_grad():
#         for inputs, targets in tqdm(
#                 dataloader, total=len(dataloader), desc="Predicting"
#         ):
#             inputs = tuple(x.to(device) for x in inputs)
#             targets = targets.to(device)
#             # output_samples, alea_samples, epistemic_samples = [], [], []
#             # for _ in range(num_mc_samples): # Multiple forward passes
#             if aleatoric:
#                 outputs, logvars = model(inputs)
#                 alea_vars = torch.exp(logvars)
#                 # vars_all.append(vars_)
#                 # output_samples.append(outputs)
#                 # alea_samples.append(alea_vars)
#
#             else:
#                 outputs = model(inputs)
#
#                 # if len(outputs) == 4: # EVidential Model
#                 #     mu, v, alpha, beta = (d.squeeze() for d in outputs)
#                 #     alea_vars = beta / (alpha - 1)  # aleatoric
#                 #     epist_var = torch.sqrt(beta / (v * (alpha - 1)))
#                 #     outputs = mu
#                 #     output_samples.append(outputs)
#                 #     alea_samples.append(alea_vars)
#                 #     epistemic_samples.append(epist_var)
#                 # else:
#                 #     output_samples.append(outputs)
#                 #
#                 #     alea_vars = torch.zeros_like(outputs)
#                 #     alea_samples.append(alea_vars)
#             #
#             # if num_mc_samples > 1: # MC Dropout - stack them dim 2
#             #     outputs = torch.stack(output_samples, dim=2)
#             #     alea_vars = torch.stack(alea_samples, dim=2)
#             #
#             # if len(outputs) == 4: # Evidential model
#             #     outputs = torch.cat(outputs, dim=0) # ??
#             #     alea_vars = torch.cat(alea_vars, dim=0)
#             #     epist_var = torch.cat(epistemic_samples, dim=0)
#             #
#             #     epistemic_all.append(epist_var)
#             outputs_all.append(outputs)
#             targets_all.append(targets)
#             if aleatoric:
#                 aleatoric_all.append(alea_vars)
#
#     # # Clean up: revert to evaluation mode if in MC dropout mode
#     # if num_mc_samples > 1:
#     #     model.eval()
#
#     outputs_all = torch.cat(outputs_all, dim=0).cpu()
#     targets_all = torch.cat(targets_all, dim=0).cpu()
#
#     if aleatoric:
#         # vars_all = torch.exp(torch.stack(logvars_all)).cpu()
#         vars_all = torch.cat(aleatoric_all, dim=0).cpu()
#         return outputs_all, targets_all, vars_all
#     return outputs_all, targets_all, None

    # if return_targets and not aleatoric:
    #     return outputs_all, targets_all, None
    # elif return_targets and aleatoric:
    #     return outputs_all, targets_all, logvars_all
    #
    # return outputs_all


def evaluate_predictions(
        config,
        preds,
        labels,
        alea_vars=None,  # TODO to decide whether to include this in the metrics part
        model_type="ensemble",
        logger=None,
        epi_vars=None,
):
    data_name = config.get("data_name", "papyrus")
    activity_type = config.get("activity_type", "xc50")
    n_targets = config.get("n_targets", -1)
    multitask = config.get("MT", False)
    data_specific_path = config.get(
        "data_specific_path", Path(data_name) / activity_type / get_topx(n_targets)
    )
    model_name = config.get("model_name", model_type)
    model_name += "_MT" if multitask else ""

    uct_metrics_logger = MetricsTable(
        model_type=model_type,
        config=config,
        add_plots_to_table=True, # * we can turn on if we want to see them in wandb * #
        logger=logger,
    )

    # preds, labels = predict(model, dataloaders["test"], return_targets=True)
    y_true, y_pred, y_std, y_err, y_alea = process_preds(preds, labels, alea_vars, epi_vars, None)
    _ = create_df_preds(
        y_true, y_pred, y_std, y_err, y_alea, True, data_specific_path, model_name, logger
    )
    task_name = f"All {n_targets} Targets" if n_targets > 1 else "PCM"
    # * note here we use only the epistemic part * #
    metrics, plots = uct_metrics_logger(
        y_pred=y_pred,
        y_std=y_std,
        y_true=y_true,
        y_err=y_err,
        y_alea=y_alea,
        task_name=task_name,
    )

    if multitask:
        tasks = get_tasks(data_name, activity_type, n_targets)
        for task_idx in range(len(tasks)):
            task_name = tasks[task_idx]
            y_true, y_pred, y_std, y_err, y_alea = process_preds(preds, labels, alea_vars, epi_vars, task_idx)
            taskmetrics, taskplots = uct_metrics_logger(
                y_pred=y_pred,
                y_std=y_std,
                y_true=y_true,
                y_err=y_err,
                y_alea=y_alea,
                task_name=task_name,
            )
            metrics[task_name] = taskmetrics
            plots[task_name] = taskplots

    uct_metrics_logger.wandb_log()

    return metrics, plots, uct_metrics_logger


def initial_evaluation(
        model, train_loader, val_loader, loss_fn, aleatoric, device=DEVICE, epoch=0  # pbar=None,
):
    # val_loss, val_rmse, val_r2, val_evs = evaluate(
    #     model, val_loader, loss_fn, aleatoric, device, pbar, False, epoch
    # )
    # train_loss, train_rmse, train_r2, train_evs = evaluate(
    #     model, train_loader, loss_fn, aleatoric, device, pbar, False, epoch
    # )
    val_loss = evaluate(
        model, val_loader, loss_fn, aleatoric, device, False, "val", epoch
    )
    train_loss = evaluate(
        model, train_loader, loss_fn, aleatoric, device, False, "train", epoch
    )
    return train_loss, val_loss


def run_one_epoch(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        lr_scheduler,
        aleatoric=False,
        epoch=0,
        device=DEVICE,
        max_norm=10.0
):
    """
    Run a single epoch of training and evaluation.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained and evaluated.
    train_loader : torch.utils.data.DataLoader
        Data Loader for training data.
    val_loader : torch.utils.data.DataLoader
        Data Loader for validation data.
    loss_fn : torch.nn.Module
        Loss function used for training and evaluation.
    optimizer : torch.optim.Optimizer
        model optimizer parameter updates.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    epoch : int, optional
        Current epoch number. Default is 0.
    device : torch.device, optional
        Device to run the model on. Default is DEVICE var.

    Returns
    -------
    float
        Validation loss for the epoch.
    """
    # total_steps = len(train_loader) + len(val_loader)
    # with tqdm(total=total_steps, desc=f"Epoch {epoch+1}", unit="batch") as pbar:
    # pbar = None
    if epoch == 0:
        train_loss, val_loss = initial_evaluation(
            model, train_loader, val_loader, loss_fn, aleatoric, device, epoch
        )

    else:
        train_loss = train(
            model, train_loader, loss_fn, optimizer, aleatoric, device, epoch, max_norm=max_norm
        )
        val_loss = evaluate(
            model, val_loader, loss_fn, aleatoric, device, False, "val", epoch
        )

        if lr_scheduler is not None:
            # Update the learning rate
            lr_scheduler.step(val_loss)

    return epoch, train_loss, val_loss


def wandb_epoch_logger(
        epoch,
        train_loss,
        train_rmse,
        train_r2,
        train_evs,
        val_loss,
        val_rmse,
        val_r2,
        val_evs,
):
    wandb.log(
        data={
            "epoch": epoch,
            "train/loss": train_loss,
            "train/rmse": train_rmse,
            "train/r2": train_r2,
            "train/evs": train_evs,
            "val/loss": val_loss,
            "val/rmse": val_rmse,
            "val/r2": val_r2,
            "val/evs": val_evs,
        }
    )


def wandb_test_logger(
        test_loss,
        test_rmse,
        test_r2,
        test_evs,
        tasks_rmse=None,
        tasks_r2=None,
        tasks_evs=None,
):
    wandb.log(
        data={
            "test/loss": test_loss,
            "test/rmse": test_rmse,
            "test/r2": test_r2,
            "test/evs": test_evs,
        }
    )
    if tasks_rmse is not None:
        for task_idx in range(len(tasks_rmse)):
            wandb.log(
                data={
                    f"test/task_{task_idx}/rmse": tasks_rmse[task_idx],
                    f"test/task_{task_idx}/r2": tasks_r2[task_idx],
                    f"test/task_{task_idx}/evs": tasks_evs[task_idx],
                }
            )


def train_model(
        model,
        config,
        train_loader,
        val_loader,
        n_targets=-1,
        seed=42,
        device=DEVICE,
        logger=None,
        max_norm=10.0
):
    try:
        set_seed(seed)
        multitask = n_targets > 1

        model = model.to(device)
        best_model = model

        best_val_loss = float("inf")
        early_stop_counter = 0

        optimizer = build_optimizer(
            model, config.optimizer, config.lr, config.weight_decay
        )
        aleatoric = config.get("aleatoric", False)

        if aleatoric and config.loss != "gaussnll":
            logger.warning(f"Aleatoric Uncertainty is to be calculated "
                           f"but the loss function provided = {config.loss} doesnt allow this. "
                           f"Changing loss to gaussianNLL")
            config.loss = "gaussnll"

        # if multitask:
        #     reduction='none'

        loss_fn = build_loss(
            config.loss,
            reduction=config.get("loss_reduction", "mean"),
            lamb=config.get("lamb", 1e-2),
            mt=multitask,
            # logger=logger
        )
        lr_scheduler = build_lr_scheduler(
            optimizer,
            config.get("lr_scheduler", None),
            config.get("lr_scheduler_patience", None),
            config.get("lr_scheduler_factor", None),
        )

        # "none", "mean", "sum"
        for epoch in tqdm(range(config.epochs), desc="Epochs"):
            # for epoch in range(config.epochs + 1):
            try:
                (
                    epoch,
                    train_loss,
                    val_loss,
                ) = run_one_epoch(
                    model,
                    train_loader,
                    val_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    aleatoric=aleatoric,
                    epoch=epoch,
                    device=device,
                    max_norm=max_norm
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    best_model = model
                else:
                    early_stop_counter += 1
                    if early_stop_counter > config.early_stop:
                        break
            except Exception as e:
                raise RuntimeError(
                    f"The following exception occurred inside the epoch loop {e}"
                )

        return best_model, loss_fn
    except Exception as e:
        raise RuntimeError(f"The following exception occurred in train_model {e}")


def run_model(
        config,
        model,
        dataloaders,
        device=DEVICE,
        logger=create_logger("run_model"),
        max_norm=10.0
):
    seed = 42
    n_targets = config.get("n_targets", -1)
    aleatoric = config.get("aleatoric", False)
    mt = n_targets > 1
    # Train the model
    best_model, loss_fn = train_model(
        model,
        config,
        dataloaders["train"],
        dataloaders["val"],
        n_targets,
        seed,
        device,
        logger=logger,
        max_norm=max_norm
    )

    # Testing metrics on the best model
    test_loss = evaluate(
        best_model, dataloaders["test"], loss_fn, aleatoric, device, metrics_per_task=mt, subset="test", epoch=None
    )

    # TODO FOR TESTING
    # outputs_all, targets_all, vars_all = predict(best_model, dataloaders["test"], aleatoric=aleatoric, device=device)

    return best_model, test_loss


def train_model_e2e(
        config,
        model,
        model_type="baseline",
        model_kwargs=None,
        logger=None,
        max_norm=10.0
):
    start_time = datetime.now()
    seed = 42
    set_seed(seed)

    if model_kwargs is None:
        model_kwargs = {}

    if config is not None:
        wandb_project_name = config.get(
            "wandb_project_name", "test-project"
        )  # add it to config only in baseline not the sweep
        run = wandb.init(
            config=config, dir=WANDB_DIR, mode=WANDB_MODE, project=wandb_project_name
        )
    else:  # TODO check the effect here?
        run = wandb.init(config=config, dir=WANDB_DIR, mode=WANDB_MODE)

    config = wandb.config

    data_name = config.get("data_name", "papyrus")
    activity_type = config.get("activity_type", "xc50")
    n_targets = config.get("n_targets", -1)
    descriptor_protein = config.get("descriptor_protein", None)
    descriptor_chemical = config.get("descriptor_chemical", None)
    median_scaling = config.get("median_scaling", False)
    split_type = config.get("split_type", "random")
    ext = config.get("ext", "pkl")
    task_type = config.get("task_type", "regression")
    # aleatoric = config.get("aleatoric", False)
    assert split_type in [
        "random",
        "scaffold",
        "time",
    ], "Split type must be either random or scaffold or time"

    m_tag = "median_scaling" if median_scaling else "no_median_scaling"
    mt = n_targets > 1
    mt_tag = "MT" if mt else "ST"
    config["MT"] = mt
    if mt and descriptor_protein:
        logger.warning(
            "For multitask learning, only chemical descriptors will be used."
            "Setting descriptor_protein to None"
        )
        descriptor_protein = None
    if mt and split_type == "time":
        logger.warning(
            "For multitask learning, only random or scaffold split will be used."
            "Setting split_type to random"
        )
        split_type = "random"

    wandb_tags = [
        model_type,
        data_name,
        activity_type,
        descriptor_protein,
        descriptor_chemical,
        split_type,
        task_type,
        m_tag,
        mt_tag,
        f"max_norm={max_norm}",
        TODAY
    ]
    # filter out None values
    wandb_tags = [tag for tag in wandb_tags if tag]
    run.tags += tuple(wandb_tags)

    logger = (
        create_logger(name=model_type, file_level="debug", stream_level="info")
        if not logger
        else logger
    )
    logger.info(f"{model_type} - start time: {start_time}")

    data_specific_path = Path(data_name) / activity_type / get_topx(n_targets)
    logger.info(f"Data specific path: {data_specific_path}")
    config["data_specific_path"] = data_specific_path

    datasets = build_datasets(
        data_name,
        n_targets,
        activity_type,
        split_type,
        descriptor_protein,
        descriptor_chemical,
        median_scaling,
        task_type,
        ext,
        logger,
    )

    desc_prot_len, desc_chem_len = get_desc_len(descriptor_protein, descriptor_chemical)
    logger.info(f"Chemical descriptor {descriptor_chemical} of length: {desc_chem_len}")
    logger.info(f"Protein descriptor {descriptor_protein} of length: {desc_prot_len}")
    config["prot_input_dim"] = desc_prot_len
    config["chem_input_dim"] = desc_chem_len

    batch_size = config.get("batch_size", 128)
    dataloaders = build_loader(datasets, batch_size, shuffle=False)

    model_ = model(config=config, **model_kwargs, logger=logger).to(DEVICE)
    logger.info(f"Model: {model_}")

    best_model, _ = run_model(
        config,
        model_,
        dataloaders,
        device=DEVICE,
        logger=logger,
        max_norm=max_norm
    )

    model_name = f"{TODAY}-{model_type}_{split_type}_{descriptor_protein}_{descriptor_chemical}-{run.name}"
    config["model_name"] = model_name

    save_model(
        config,
        best_model,
        model_name,
        data_specific_path,
        desc_prot_len,
        desc_chem_len,
        onnx=True,
    )

    logger.info(f"Baseline - end time: {datetime.now()}")
    logger.info(f"Baseline - duration: {datetime.now() - start_time}")

    return best_model, dataloaders, config, logger


def recalibrate_model(preds_val, labels_val, preds_test, labels_test, config, epi_val=None, epi_test=None, uct_logger=None):
    model_name = config.get("model_name", "ensemble")
    data_specific_path = config.get("data_specific_path", None)

    figures_path = FIGS_DIR / data_specific_path / model_name

    # Validation Set
    y_true_val, y_pred_val, y_std_val, y_err_val, _ = process_preds(preds_val, labels_val, epi_vars=epi_val)
    y_true_test, y_pred_test, y_std_test, y_err_test, _ = process_preds(preds_test, labels_test, epi_vars=epi_test)

    recal_model = recalibrate(
        y_true_val, y_pred_val, y_std_val, y_err_val, y_true_test, y_pred_test, y_std_test, y_err_test,
        n_subset=None, savefig=True, save_dir=figures_path, uct_logger=uct_logger
    )
    # TODO add task_name
    model_dir = MODELS_DIR / "saved_models" / data_specific_path
    model_dir.mkdir(exist_ok=True)
    model_name = config.get("model_name", "ensemble") + "_recalibrate_model.pkl"
    # pickle save the model to model_dir
    export_pickle(recal_model, model_dir / model_name)

    # Test Set
    return recal_model


# def _run_model_e2e(
#     model,
#     model_type="baseline",
#     config=None,
#     data_kwargs=None,
#     wandb_project_name="baseline-test",
#     model_save_name=f"{TODAY}-baseline_random_ankh-base_ecfp2048",
#     logger=None,
#     **kwargs,
# ):
#     # data_name="papyrus",
#     # activity_type="xc50",
#     # n_targets=-1,
#     # descriptor_protein=None,
#     # descriptor_chemical=None,
#     # label_scaling_func=None,
#     # split_type="random",
#     # ext="pkl",
#     if data_kwargs is None:
#         data_kwargs = {
#             "data_name": "papyrus",
#             "activity_type": "xc50",
#             "n_targets": -1,
#             "descriptor_protein": None,
#             "descriptor_chemical": None,
#             "label_scaling_func": None,
#             "split_type": "random",
#             "ext": "pkl",
#         }
#     (
#         dataloaders,
#         config,
#         logger,
#         desc_prot_len,
#         desc_chem_len,
#         start_time,
#         data_specific_path,
#     ) = premodel_init(
#         config,
#         model_type,
#         **data_kwargs,
#         logger=logger,
#         **kwargs,
#     )
#     # data_name,
#     # activity_type,
#     # n_targets,
#     # descriptor_protein,
#     # descriptor_chemical,
#     # label_scaling_func,
#     # split_type,
#     # ext,
#
#     with wandb.init(
#         dir=WANDB_DIR, mode=WANDB_MODE, project=wandb_project_name, config=config
#     ):
#         config = wandb.config
#         best_model, test_loss = run_model(
#             config,
#             model,
#             dataloaders,
#             data_kwargs.get("n_targets", -1),
#             DEVICE,
#             logger,
#         )
#         logger.info(f"Test Loss: {test_loss}")
#
#         save_model(
#             config=config,
#             model=best_model,
#             model_name=model_save_name,
#             data_specific_path=data_specific_path,
#             desc_prot_len=desc_prot_len,
#             desc_chem_len=desc_chem_len,
#             onnx=True,
#         )
#
#     return best_model, test_loss
#     #
#     # seed = 42
#     # set_seed(seed)
#     #
#     # config = get_model_config(model_type, **kwargs) if not config else config
#     # logger = (
#     #     create_logger(name=model_type, file_level="debug", stream_level="info")
#     #     if not logger
#     #     else logger
#     # )
#     #
#     # start_time = datetime.now()
#     # logger.info(f"{model_type} - start time: {start_time}")
#     #
#     # # get datasets
#     # datasets = build_datasets(
#     #     data_name=data_name,
#     #     n_targets=n_targets,
#     #     activity_type=activity_type,
#     #     split_type=split_type,
#     #     desc_prot=descriptor_protein,
#     #     desc_chem=descriptor_chemical,
#     #     label_scaling_func=label_scaling_func,
#     #     ext=ext,
#     #     logger=logger,
#     # )
#     #
#     # desc_prot_len, desc_chem_len = get_desc_len(descriptor_protein), get_desc_len(
#     #     descriptor_chemical
#     # )
#     # logger.info(f"Chemical descriptor {descriptor_chemical} of length: {desc_chem_len}")
#     # logger.info(f"Protein descriptor {descriptor_protein} of length: {desc_prot_len}")
#     #
#     # # Load the dataset
#     # dataloaders = build_loader(datasets, batch_size, shuffle=True)
