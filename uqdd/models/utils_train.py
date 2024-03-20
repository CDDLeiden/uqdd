from datetime import datetime
from pathlib import Path

import wandb
import torch

from tqdm import tqdm
from uqdd import DEVICE, WANDB_DIR, WANDB_MODE, TODAY
from uqdd.data.utils_data import get_topx

from uqdd.models.loss import build_loss
from uqdd.models.utils_metrics import calc_regr_metrics
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
    # for inputs, targets in tqdm(dataloader, total=len(dataloader), desc="Training Batches"):
    for inputs, targets in dataloader:
        inputs = (
            tuple(x.to(device) for x in inputs)
            if isinstance(inputs, tuple) or isinstance(inputs, list)
            else inputs.to(device)
        )
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = (
            outputs.squeeze()
            if isinstance(outputs, torch.Tensor)
            else (outp.squeeze() for outp in outputs)
        )
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * outputs.size(0)
        if pbar:
            # pbar.set_description(f"Epoch {epoch + 1} : Training")
            pbar.set_postfix(loss=loss.item(), refresh=True)
            pbar.update(1)
    total_loss /= len(dataloader.dataset)
    # # Here we want to report total_loss to pbar
    # if pbar:
    #     pbar.set_postfix(train_loss=total_loss, refresh=True)
    return total_loss


def evaluate(model, dataloader, loss_fn, device=DEVICE, pbar=None, epoch=0):
    model.eval()
    total_loss = 0.0
    targets_all = []
    outputs_all = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = (
                tuple(x.to(device) for x in inputs)
                if isinstance(inputs, tuple) or isinstance(inputs, list)
                else inputs.to(device)
            )
            targets = targets.to(device)
            outputs = model(inputs)

            outputs = (
                outputs.squeeze()
                if isinstance(outputs, torch.Tensor)
                else (outp.squeeze() for outp in outputs)
            )
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
        rmse, r2, evs = calc_regr_metrics(targets_all.squeeze(), outputs_all)
    # Here we want to report total_loss to pbar
    # if pbar:
    #     pbar.set_postfix(val_loss=total_loss, refresh=True)
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
            inputs = (
                tuple(x.to(device) for x in inputs)
                if isinstance(inputs, tuple) or isinstance(inputs, list)
                else inputs.to(device)
            )
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
        model, val_loader, loss_fn, device, pbar, epoch
    )
    train_loss, _, _, _ = evaluate(model, train_loader, loss_fn, device, pbar, epoch)
    return train_loss, val_loss, val_rmse, val_r2, val_evs


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
        train_loss, val_loss, val_rmse, val_r2, val_evs = initial_evaluation(
            model, train_loader, val_loader, loss_fn, device, pbar, epoch
        )

    else:
        train_loss = train(model, train_loader, loss_fn, optimizer, device, pbar, epoch)
        val_loss, val_rmse, val_r2, val_evs = evaluate(
            model, val_loader, loss_fn, device, pbar, epoch
        )

        # Update the learning rate
        lr_scheduler.step(val_loss)

    return epoch, train_loss, val_loss, val_rmse, val_r2, val_evs


def wandb_epoch_logger(epoch, train_loss, val_loss, val_rmse, val_r2, val_evs):
    wandb.log(
        data={
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/rmse": val_rmse,
            "val/r2": val_r2,
            "val/evs": val_evs,
        }
    )


def wandb_test_logger(test_loss, test_rmse, test_r2, test_evs):
    wandb.log(
        data={
            "test/loss": test_loss,
            "test/rmse": test_rmse,
            "test/r2": test_r2,
            "test/evs": test_evs,
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
        loss_fn = build_loss(config.loss, reduction=config.loss_reduction, mt=multitask)
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
                epoch, train_loss, val_loss, val_rmse, val_r2, val_evs = run_one_epoch(
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
                    epoch, train_loss, val_loss, val_rmse, val_r2, val_evs
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


def premodel_init(
    config,
    model_type="baseline",
    data_name="papyrus",
    activity_type="xc50",
    n_targets=-1,
    descriptor_protein=None,
    descriptor_chemical=None,
    split_type="random",
    median_scaling=False,
    task_type="regression",
    ext="pkl",
    logger=None,
    **kwargs,
):
    # label_scaling_func=None,
    start_time = datetime.now()

    seed = 42
    set_seed(seed)
    config = get_model_config(model_type, **kwargs) if not config else config
    logger = (
        create_logger(name=model_type, file_level="debug", stream_level="info")
        if not logger
        else logger
    )
    logger.info(f"{model_type} - start time: {start_time}")

    data_specific_path = Path(data_name) / activity_type / get_topx(n_targets)
    datasets = build_datasets(
        data_name,
        n_targets,
        activity_type,
        split_type,
        descriptor_protein,
        descriptor_chemical,
        median_scaling,
        task_type,
        # label_scaling_func,
        ext,
        logger,
    )

    desc_prot_len, desc_chem_len = get_desc_len(descriptor_protein, descriptor_chemical)
    logger.info(f"Chemical descriptor {descriptor_chemical} of length: {desc_chem_len}")
    logger.info(f"Protein descriptor {descriptor_protein} of length: {desc_prot_len}")

    batch_size = config.get("batch_size", 64)
    dataloaders = build_loader(datasets, batch_size, shuffle=False)

    return (
        dataloaders,
        config,
        logger,
        desc_prot_len,
        desc_chem_len,
        start_time,
        data_specific_path,
    )


def run_model(
    config,
    model,
    dataloaders,
    n_targets=-1,
    device=DEVICE,
    logger=create_logger("run_model"),
):
    seed = 42
    # Train the model
    best_model, loss_fn = train_model(
        model,
        config,
        dataloaders["train"],
        dataloaders["val"],
        n_targets,
        seed,
        device,
    )

    # Testing metrics on the best model
    test_loss, test_rmse, test_r2, test_evs = evaluate(
        best_model, dataloaders["test"], loss_fn, device
    )
    logger.info(f"Test loss: {test_loss}")

    wandb_test_logger(test_loss, test_rmse, test_r2, test_evs)

    return best_model, test_loss


def run_model_e2e(
    model,
    model_type="baseline",
    config=None,
    data_kwargs=None,
    wandb_project_name="baseline-test",
    model_save_name=f"{TODAY}-baseline_random_ankh-base_ecfp2048",
    logger=None,
    **kwargs,
):
    # data_name="papyrus",
    # activity_type="xc50",
    # n_targets=-1,
    # descriptor_protein=None,
    # descriptor_chemical=None,
    # label_scaling_func=None,
    # split_type="random",
    # ext="pkl",
    if data_kwargs is None:
        data_kwargs = {
            "data_name": "papyrus",
            "activity_type": "xc50",
            "n_targets": -1,
            "descriptor_protein": None,
            "descriptor_chemical": None,
            "label_scaling_func": None,
            "split_type": "random",
            "ext": "pkl",
        }
    (
        dataloaders,
        config,
        logger,
        desc_prot_len,
        desc_chem_len,
        start_time,
        data_specific_path,
    ) = premodel_init(
        config,
        model_type,
        **data_kwargs,
        logger=logger,
        **kwargs,
    )
    # data_name,
    # activity_type,
    # n_targets,
    # descriptor_protein,
    # descriptor_chemical,
    # label_scaling_func,
    # split_type,
    # ext,

    with wandb.init(
        dir=WANDB_DIR, mode=WANDB_MODE, project=wandb_project_name, config=config
    ):
        config = wandb.config
        best_model, test_loss = run_model(
            config,
            model,
            dataloaders,
            data_kwargs.get("n_targets", -1),
            DEVICE,
            logger,
        )
        logger.info(f"Test Loss: {test_loss}")

        save_model(
            config=config,
            model=best_model,
            model_name=model_save_name,
            data_specific_path=data_specific_path,
            desc_prot_len=desc_prot_len,
            desc_chem_len=desc_chem_len,
            onnx=True,
        )

    return best_model, test_loss
    #
    # seed = 42
    # set_seed(seed)
    #
    # config = get_model_config(model_type, **kwargs) if not config else config
    # logger = (
    #     create_logger(name=model_type, file_level="debug", stream_level="info")
    #     if not logger
    #     else logger
    # )
    #
    # start_time = datetime.now()
    # logger.info(f"{model_type} - start time: {start_time}")
    #
    # # get datasets
    # datasets = build_datasets(
    #     data_name=data_name,
    #     n_targets=n_targets,
    #     activity_type=activity_type,
    #     split_type=split_type,
    #     desc_prot=descriptor_protein,
    #     desc_chem=descriptor_chemical,
    #     label_scaling_func=label_scaling_func,
    #     ext=ext,
    #     logger=logger,
    # )
    #
    # desc_prot_len, desc_chem_len = get_desc_len(descriptor_protein), get_desc_len(
    #     descriptor_chemical
    # )
    # logger.info(f"Chemical descriptor {descriptor_chemical} of length: {desc_chem_len}")
    # logger.info(f"Protein descriptor {descriptor_protein} of length: {desc_prot_len}")
    #
    # # Load the dataset
    # dataloaders = build_loader(datasets, batch_size, shuffle=True)
