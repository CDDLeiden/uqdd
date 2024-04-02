from datetime import datetime
from pathlib import Path

import wandb
import torch

from tqdm import tqdm
from uqdd import DEVICE, WANDB_DIR, WANDB_MODE, TODAY
from uqdd.data.utils_data import get_topx, get_tasks

from uqdd.models.loss import build_loss
from uqdd.models.utils_metrics import (
    calc_regr_metrics,
    MetricsTable,
    process_preds,
    create_df_preds,
)
from uqdd.models.utils_models import (
    build_loader,
    build_optimizer,
    build_lr_scheduler,
    save_model,
    # calc_regr_metrics,
    set_seed,
    get_model_config,
    build_datasets,
    get_desc_len,
)
from uqdd.utils import create_logger


def train(model, dataloader, loss_fn, optimizer, device=DEVICE, pbar=None, epoch=0):
    model.train()
    total_loss = 0.0
    targets_all = []
    outputs_all = []
    # for inputs, targets in tqdm(dataloader, total=len(dataloader), desc="Training Batches"):
    for inputs, targets in dataloader:
        inputs = tuple(x.to(device) for x in inputs)
        # inputs = (
        #     tuple(x.to(device) for x in inputs)
        #     if isinstance(inputs, tuple) or isinstance(inputs, list)
        #     else inputs.to(device)
        # )
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # outputs = (
        #     outputs.squeeze()
        #     if isinstance(outputs, torch.Tensor)
        #     else (outp.squeeze() for outp in outputs)
        # )
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * outputs.size(0)
        targets_all.append(targets)
        outputs_all.append(outputs)
        if pbar:
            # pbar.set_description(f"Epoch {epoch + 1} : Training")
            pbar.set_postfix(loss=loss.item(), refresh=True)
            pbar.update(1)

    total_loss /= len(dataloader.dataset)

    targets_all = torch.cat(targets_all, dim=0)
    outputs_all = torch.cat(outputs_all, dim=0)
    # Calculate metrics
    train_rmse, train_r2, train_evs = calc_regr_metrics(
        targets_all, outputs_all
    )  # squeeze

    # # Here we want to report total_loss to pbar
    # if pbar:
    #     pbar.set_postfix(train_loss=total_loss, refresh=True)
    return total_loss, train_rmse, train_r2, train_evs


def evaluate(
    model,
    dataloader,
    loss_fn,
    device=DEVICE,
    pbar=None,
    metrics_per_task=False,
    epoch=0,
):
    model.eval()
    total_loss = 0.0
    targets_all = []
    outputs_all = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = tuple(x.to(device) for x in inputs)
            # inputs = (
            #     tuple(x.to(device) for x in inputs)
            #     if isinstance(inputs, tuple) or isinstance(inputs, list)
            #     else inputs.to(device)
            # )
            targets = targets.to(device)
            outputs = model(inputs)

            # outputs = (
            #     outputs.squeeze()
            #     if isinstance(outputs, torch.Tensor)
            #     else (outp.squeeze() for outp in outputs)
            # )
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * outputs.size(0)
            targets_all.append(targets)
            outputs_all.append(outputs)

            if pbar:
                # desc = ": Initial Evaluation" if epoch == 0 else ": Evaluating"
                # pbar.set_description(f"Epoch {epoch + 1} {desc}")
                pbar.set_postfix(loss=loss.item(), refresh=True)
                pbar.update(1)

        total_loss /= len(dataloader.dataset)
        targets_all = torch.cat(targets_all, dim=0)
        outputs_all = torch.cat(outputs_all, dim=0)
        # Calculate metrics
        if metrics_per_task:
            tasks_rmse, tasks_r2, tasks_evs = calc_regr_metrics(
                targets_all, outputs_all, metrics_per_task
            )
        rmse, r2, evs = calc_regr_metrics(targets_all, outputs_all)

        # if metrics_per_task:
        #     for task_idx in range(targets_all.size(1)):
        #         task_targets = targets_all[:, task_idx]
        #         task_outputs = outputs_all[:, task_idx]
        #         task_rmse, task_r2, task_evs = calc_regr_metrics(
        #             task_targets, task_outputs
        #         )
        #         rmse[f"task_{task_idx}"] = task_rmse
        #         r2[f"task_{task_idx}"] = task_r2
        #         evs[f"task_{task_idx}"] = task_evs
    # Here we want to report total_loss to pbar
    # if pbar:
    #     pbar.set_postfix(val_loss=total_loss, refresh=True)
    if metrics_per_task:
        return total_loss, rmse, r2, evs, tasks_rmse, tasks_r2, tasks_evs
    return total_loss, rmse, r2, evs


def predict(
    model,
    dataloader,
    return_targets=False,
    device=DEVICE,
):
    model.eval()
    outputs_all = []
    targets_all = []
    with torch.no_grad():
        for inputs, targets in tqdm(
            dataloader, total=len(dataloader), desc="Predicting"
        ):
            inputs = tuple(x.to(device) for x in inputs)
            # inputs = (
            #     tuple(x.to(device) for x in inputs)
            #     if isinstance(inputs, tuple) or isinstance(inputs, list)
            #     else inputs.to(device)
            # )
            outputs = model(inputs)
            outputs = (
                outputs.squeeze()
                if isinstance(outputs, torch.Tensor)
                else (outp.squeeze() for outp in outputs)
            )
            outputs_all.append(outputs)
            if return_targets:
                targets_all.append(targets)
    outputs_all = torch.cat(outputs_all, dim=0).cpu()
    if return_targets:
        targets_all = torch.cat(targets_all, dim=0).cpu()
        return outputs_all, targets_all
    return outputs_all


def initial_evaluation(
    model, train_loader, val_loader, loss_fn, device=DEVICE, pbar=None, epoch=0
):
    val_loss, val_rmse, val_r2, val_evs = evaluate(
        model, val_loader, loss_fn, device, pbar, False, epoch
    )
    train_loss, train_rmse, train_r2, train_evs = evaluate(
        model, train_loader, loss_fn, device, pbar, False, epoch
    )
    return (
        train_loss,
        train_rmse,
        train_r2,
        train_evs,
        val_loss,
        val_rmse,
        val_r2,
        val_evs,
    )


def run_one_epoch(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    lr_scheduler,
    epoch=0,
    device=DEVICE,
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
    pbar = None
    if epoch == 0:
        # Perform evaluation before training starts (epoch 0)
        (
            train_loss,
            train_rmse,
            train_r2,
            train_evs,
            val_loss,
            val_rmse,
            val_r2,
            val_evs,
        ) = initial_evaluation(
            model, train_loader, val_loader, loss_fn, device, pbar, epoch
        )

    else:
        train_loss, train_rmse, train_r2, train_evs = train(
            model, train_loader, loss_fn, optimizer, device, pbar, epoch
        )
        val_loss, val_rmse, val_r2, val_evs = evaluate(
            model, val_loader, loss_fn, device, pbar, False, epoch
        )

        # Update the learning rate
        lr_scheduler.step(val_loss)

    return (
        epoch,
        train_loss,
        train_rmse,
        train_r2,
        train_evs,
        val_loss,
        val_rmse,
        val_r2,
        val_evs,
    )


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
        loss_fn = build_loss(
            config.loss, reduction=config.loss_reduction, mt=multitask, logger=logger
        )
        lr_scheduler = build_lr_scheduler(
            optimizer,
            config.lr_scheduler,
            config.lr_scheduler_patience,
            config.lr_scheduler_factor,
        )

        # "none", "mean", "sum"
        for epoch in tqdm(range(config.epochs), desc="Epochs"):
            # for epoch in range(config.epochs + 1):
            try:
                (
                    epoch,
                    train_loss,
                    train_rmse,
                    train_r2,
                    train_evs,
                    val_loss,
                    val_rmse,
                    val_r2,
                    val_evs,
                ) = run_one_epoch(
                    model,
                    train_loader,
                    val_loader,
                    loss_fn,
                    optimizer,
                    lr_scheduler,
                    epoch=epoch,
                    device=device,
                )
                wandb_epoch_logger(
                    epoch,
                    train_loss,
                    train_rmse,
                    train_r2,
                    train_evs,
                    val_loss,
                    val_rmse,
                    val_r2,
                    val_evs,
                    # epoch, train_loss, val_loss, val_rmse, val_r2, val_evs
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
):
    seed = 42
    n_targets = config.get("n_targets", -1)
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
    )

    # Testing metrics on the best model
    if mt:
        test_loss, test_rmse, test_r2, test_evs, tasks_rmse, tasks_r2, tasks_evs = (
            evaluate(
                best_model, dataloaders["test"], loss_fn, device, metrics_per_task=True
            )
        )
        wandb_test_logger(
            test_loss, test_rmse, test_r2, test_evs, tasks_rmse, tasks_r2, tasks_evs
        )
    else:
        test_loss, test_rmse, test_r2, test_evs = evaluate(
            best_model, dataloaders["test"], loss_fn, device
        )

        wandb_test_logger(test_loss, test_rmse, test_r2, test_evs)

    logger.info(f"Test loss: {test_loss}")
    logger.info(f"Test RMSE: {test_rmse}")
    logger.info(f"Test R2: {test_r2}")
    logger.info(f"Test EVS: {test_evs}")

    return best_model, test_loss


def train_model_e2e(
    config,
    model,
    model_type="baseline",
    model_kwargs=None,
    logger=None,
):

    start_time = datetime.now()
    seed = 42
    set_seed(seed)

    if model_kwargs is None:
        model_kwargs = {}

    if config is not None:
        wandb_project_name = config.get(
            "wandb_project_name"
        )  # add it to config only in baseline not the sweep
        run = wandb.init(
            config=config, dir=WANDB_DIR, mode=WANDB_MODE, project=wandb_project_name
        )
    else:
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


def predict_uc_metrics(
    config,
    preds,
    labels,
    model_type="ensemble",
    logger=None,
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
        logger=logger,
    )

    # preds, labels = predict(model, dataloaders["test"], return_targets=True)
    y_true, y_pred, y_std, y_err = process_preds(preds, labels, None)
    _ = create_df_preds(
        y_true, y_pred, y_std, y_err, True, data_specific_path, model_name, logger
    )
    task_name = f"All {n_targets} Targets" if n_targets > 1 else "PCM"

    metrics, plots = uct_metrics_logger(
        y_pred=y_pred,
        y_std=y_std,
        y_true=y_true,
        y_err=y_err,
        task_name=task_name,
    )

    if multitask:
        tasks = get_tasks(data_name, activity_type, n_targets)
        for task_idx in range(len(tasks)):
            task_name = tasks[task_idx]
            y_true, y_pred, y_std, y_err = process_preds(preds, labels, task_idx)

            taskmetrics, taskplots = uct_metrics_logger(
                y_pred=y_pred,
                y_std=y_std,
                y_true=y_true,
                y_err=y_err,
                task_name=task_name,
            )
            metrics[taskmetrics] = taskmetrics
            plots[taskplots] = taskplots

    uct_metrics_logger.wandb_log()

    return metrics, plots


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
