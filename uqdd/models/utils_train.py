import numpy as np
import wandb
import torch

from tqdm import tqdm
from uqdd import DEVICE, WANDB_DIR, WANDB_MODE, TODAY, FIGS_DIR
from uqdd.data.utils_data import get_tasks  # , save_pickle, get_topx,

from uqdd.models.loss import build_loss
from uqdd.models.utils_metrics import (
    calc_regr_metrics,
    MetricsTable,
    process_preds,
    create_df_preds,
    calc_alea_epi_mean_var_notnan,
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
    compute_gnorm,
    ckpt,
    load_ckpt,
    get_model_name,
    get_data_specific_path,
)
from uqdd.utils import create_logger


def evidential_processing(outputs):  # , alea_all, epi_all
    # if len(outputs) == 4:  # Evidential model
    # mu, v, alpha, beta = (d.squeeze() for d in outputs)
    mu, v, alpha, beta = outputs
    alea_vars = beta / (alpha - 1)  # aleatoric
    epi_vars = torch.sqrt(beta / (v * (alpha - 1)))  # epistemic
    return alea_vars, epi_vars
    # return None, None
    # alea_all.append(alea_vars)
    # epi_all.append(epi_vars)
    #
    # outputs = mu

    # return outputs, alea_all


def model_forward(model, inputs, targets, lossfname="EvidentialRegression"):
    if lossfname.lower() == "evidential_regression":
        outputs = model(inputs)
        alea_vars, epi_vars = evidential_processing(outputs)
        args = (outputs, targets)
        return outputs, alea_vars, epi_vars, args
    elif lossfname.lower() == "gaussnll":
        outputs, vars_ = model(inputs)
        args = (outputs, targets, vars_)
        return outputs, vars_, None, args
    else:
        outputs, vars_ = model(inputs)
        args = (outputs, targets)
        return outputs, vars_, None, args


# def apply_model_aleatoric_option()
def train(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device=DEVICE,
    epoch=0,
    max_norm=None,
    lossfname="EvidenceRegressionLoss",
    tracker="wandb",  # pbar=None,
    subset="train",
):
    # max_norm = 10.0
    model.train()
    total_loss = 0.0
    targets_all = []
    outputs_all = []
    vars_all = []
    epis_all = []
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for inputs, targets in dataloader:
        inputs = tuple(x.to(device) for x in inputs)
        targets = targets.to(device)
        optimizer.zero_grad()
        # outputs, vars_ = model(inputs)

        outputs, alea_vars, epi_vars, args = model_forward(
            model, inputs, targets, lossfname=lossfname
        )
        # if lossfname.lower() == "evidential_regression":
        #     outputs = model(inputs)
        #     vars_, epis_ = evidential_processing(outputs)
        #
        #     epis_all.append(epis_)
        #
        #     args = (outputs, targets)
        # elif lossfname.lower() == "gaussnll":
        #     outputs, vars_ = model(inputs)
        #     args = (outputs, targets, vars_)
        # else:
        #     outputs, vars_ = model(inputs)
        #     args = (outputs, targets)
        # if outputs.dim() > targets.dim():
        #     _, _, num_repeats = outputs.shape
        #     targets = targets.repeat(num_repeats, 1).t()
        # t = t.unsqueeze(2).expand(-1,-1,5)
        # args = (
        #     (outputs, targets, vars_) if lossfname == "gaussnll" else (outputs, targets)
        # )
        loss = loss_fn(*args)
        loss.backward()

        vars_all.append(alea_vars.detach())
        epis_all.append(epi_vars.detach()) if epi_vars is not None else None

        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        total_loss += loss.item()  # * outputs.size(0)
        # outputs, vars_all = evidential_processing(outputs, vars_all)
        targets_all.append(targets)
        outputs = (
            outputs[0] if lossfname.lower() == "evidential_regression" else outputs
        )  # Because it gets 4 outputs
        outputs_all.append(outputs)

    total_loss /= num_batches  # len(dataloader.dataset)
    targets_all = torch.cat(targets_all, dim=0)
    outputs_all = torch.cat(outputs_all, dim=0)

    # Calculate metrics
    train_rmse, train_r2, train_evs = calc_regr_metrics(targets_all, outputs_all)
    pnorm = compute_pnorm(model)
    gnorm = compute_gnorm(model)
    vars_all = torch.cat(vars_all, dim=0)
    vars_mean, vars_var = calc_alea_epi_mean_var_notnan(vars_all, targets_all)

    if tracker.lower() == "wandb":
        data = {
            f"{subset}/loss": total_loss,
            f"{subset}/rmse": train_rmse,
            f"{subset}/r2": train_r2,
            f"{subset}/evs": train_evs,
            f"{subset}/alea_mean": vars_mean.item(),
            f"{subset}/alea_var": vars_var.item(),
            "model/pnorm": pnorm,
            "model/gnorm": gnorm,
        }

        if lossfname.lower() == "evidential_regression":
            epis_all = torch.cat(epis_all, dim=0)
            epis_mean, epis_var = calc_alea_epi_mean_var_notnan(epis_all, targets_all)
            data[f"{subset}/epis_mean"] = epis_mean.item()
            data[f"{subset}/epis_var"] = epis_var.item()

        wandb.log(
            data=data,
            step=epoch,
        )

    elif tracker.lower() == "tensor":
        vals = [
            epoch,
            total_loss,
            train_rmse,
            train_r2,
            train_evs,
            vars_mean.item(),
            vars_var.item(),
            pnorm,
            gnorm,
        ]
        if lossfname.lower() == "evidential_regression":
            epis_all = torch.cat(epis_all, dim=0)
            epis_mean, epis_var = calc_alea_epi_mean_var_notnan(epis_all, targets_all)
            vals.extend([epis_mean.item(), epis_var.item()])

        tracked_vals = np.array(
            vals,
            dtype=np.float32,
        )  # pnorm, gnorm
        return tracked_vals

    return total_loss


def evaluate(
    model,
    dataloader,
    loss_fn,
    device=DEVICE,
    metrics_per_task=False,
    subset="val",  # can be "train", "val" or "test"
    epoch: int | None = 0,
    lossfname="EvidenceRegressionLoss",
    tracker="wandb",
):
    model.eval()
    total_loss = 0.0
    targets_all = []
    outputs_all = []
    vars_all = []
    epis_all = []
    num_batches = len(dataloader)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = tuple(x.to(device) for x in inputs)
            targets = targets.to(device)

            outputs, alea_vars, epi_vars, args = model_forward(
                model, inputs, targets, lossfname=lossfname
            )

            # outputs, vars_ = model(inputs)
            # args = (
            #     (outputs, targets, vars_)
            #     if lossfname.lower() == "gaussnll"
            #     else (outputs, targets)
            # )
            loss = loss_fn(*args)

            vars_all.append(alea_vars.detach())
            epis_all.append(epi_vars.detach()) if epi_vars is not None else None

            total_loss += loss.item()  # * outputs.size(0)

            outputs = (
                outputs[0] if lossfname.lower() == "evidential_regression" else outputs
            )  # Because it gets 4 outputs
            outputs_all.append(outputs)
            targets_all.append(targets)

        total_loss /= num_batches  # len(dataloader.dataset)
        targets_all = torch.cat(targets_all, dim=0)
        outputs_all = torch.cat(outputs_all, dim=0)

        # Calculate metrics
        rmse, r2, evs = calc_regr_metrics(targets_all, outputs_all)

        # Aleatoric Uncertainty
        vars_all = torch.cat(vars_all, dim=0)
        vars_mean, vars_var = calc_alea_epi_mean_var_notnan(vars_all, targets_all)

        if tracker.lower() == "wandb":
            data = {
                f"{subset}/loss": total_loss,
                f"{subset}/rmse": rmse,
                f"{subset}/r2": r2,
                f"{subset}/evs": evs,
                f"{subset}/alea_mean": vars_mean.item(),
                f"{subset}/alea_var": vars_var.item(),
            }

            if lossfname.lower() == "evidential_regression":
                epis_all = torch.cat(epis_all, dim=0)
                epis_mean, epis_var = calc_alea_epi_mean_var_notnan(
                    epis_all, targets_all
                )
                data[f"{subset}/epis_mean"] = epis_mean.item()
                data[f"{subset}/epis_var"] = epis_var.item()

            wandb.log(
                data=data,
                step=epoch,
            )

        elif tracker.lower() == "tensor":
            vals = [epoch, total_loss, rmse, r2, evs, vars_mean.item(), vars_var.item()]
            (
                vals.extend([compute_pnorm(model), compute_gnorm(model)])
                if epoch == 0 and subset == "train"
                else None
            )

            if lossfname.lower() == "evidential_regression":
                epis_all = torch.cat(epis_all, dim=0)
                epis_mean, epis_var = calc_alea_epi_mean_var_notnan(
                    epis_all, targets_all
                )
                vals.extend([epis_mean.item(), epis_var.item()])
            tracked_vals = np.array(
                vals,
                dtype=np.float32,
            )

        # Calculate metrics
        if metrics_per_task:
            tasks_rmse, tasks_r2, tasks_evs = calc_regr_metrics(
                targets_all, outputs_all, metrics_per_task
            )
            for task_idx in range(len(tasks_rmse)):
                if tracker.lower() == "wandb":
                    wandb.log(
                        data={
                            f"{subset}/rmse/task_{task_idx}": tasks_rmse[task_idx],
                            f"{subset}/r2/task_{task_idx}": tasks_r2[task_idx],
                            f"{subset}/evs/task_{task_idx}": tasks_evs[task_idx],
                        },
                        step=epoch,
                    )
                elif tracker.lower() == "tensor":
                    tracked_vals = np.append(
                        tracked_vals,
                        [tasks_rmse[task_idx], tasks_r2[task_idx], tasks_evs[task_idx]],
                    )
    if tracker.lower() == "tensor":
        tracked_vals = np.array(
            tracked_vals,
            dtype=np.float32,
        )
        return tracked_vals

    return total_loss


def predict(
    model,
    dataloader,
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
            targets = targets.to(device)

            outputs, vars_ = model(inputs)
            vars_all.append(vars_)
            outputs_all.append(outputs)
            targets_all.append(targets)

    outputs_all = torch.cat(outputs_all, dim=0).cpu()
    targets_all = torch.cat(targets_all, dim=0).cpu()
    vars_all = torch.cat(vars_all, dim=0).cpu()
    return outputs_all, targets_all, vars_all


def evaluate_predictions(
    config,
    preds,
    labels,
    alea_vars=None,  # TODO to decide whether to include this in the metrics part
    model_type="ensemble",
    logger=None,
    epi_vars=None,
    wandb_push=False,
):
    data_name = config.get("data_name", "papyrus")
    activity_type = config.get("activity_type", "xc50")
    n_targets = config.get("n_targets", -1)
    multitask = config.get("MT", False)
    data_specific_path = get_data_specific_path(config, logger=logger)
    #     config.get(
    #     "data_specific_path", Path(data_name) / activity_type / get_topx(n_targets)
    # ))
    # model_name = config.get("model_name", model_type)
    # model_name += "_MT" if multitask else ""

    model_name = get_model_name(config)

    uct_metrics_logger = MetricsTable(
        model_type=model_type,
        config=config,
        add_plots_to_table=True,  # * we can turn on if we want to see them in wandb * #
        logger=logger,
    )

    # preds, labels = predict(model, dataloaders["test"], return_targets=True)
    y_true, y_pred, y_std, y_err, y_alea = process_preds(
        preds, labels, alea_vars, epi_vars, None
    )
    _ = create_df_preds(
        y_true,
        y_pred,
        y_std,
        y_err,
        y_alea,
        True,
        data_specific_path,
        model_name,
        logger,
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
            y_true, y_pred, y_std, y_err, y_alea = process_preds(
                preds, labels, alea_vars, epi_vars, task_idx
            )
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

    # * log the metrics and plots to wandb * #
    if wandb_push:
        uct_metrics_logger.wandb_log()

    return metrics, plots, uct_metrics_logger


def initial_evaluation(
    model,
    train_loader,
    val_loader,
    loss_fn,
    device=DEVICE,
    epoch=0,
    lossfname="",
    tracker="wandb",  # pbar=None,
):
    val_loss = evaluate(
        model,
        val_loader,
        loss_fn,
        device,
        False,
        "val",
        epoch,
        lossfname,
        tracker=tracker,
    )
    train_loss = evaluate(
        model,
        train_loader,
        loss_fn,
        device,
        False,
        "train",
        epoch,
        lossfname,
        tracker=tracker,
    )
    return train_loss, val_loss


def run_one_epoch(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    lr_scheduler,
    epoch=0,
    device=DEVICE,
    max_norm=None,
    lossfname="",
    tracker="wandb",
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
    max_norm : float, optional
        Maximum norm value for gradient clipping. Default is None.
    lossfname : str, optional
        Loss function name. Default is "".
    tracker : str, optional
        Tracker to log the results. Default is "wandb". Other option is "tensor".

    Returns
    -------
    float
        Validation loss for the epoch.
    """
    # pbar = None
    if epoch == 0:
        # TODO fix values returned to accomodate the array from tensor tracker.
        train_loss, val_loss = initial_evaluation(
            model, train_loader, val_loader, loss_fn, device, epoch, lossfname, tracker
        )

    else:
        train_loss = train(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            epoch,
            max_norm=max_norm,
            lossfname=lossfname,
            tracker=tracker,
        )
        val_loss = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            False,
            "val",
            epoch,
            lossfname,
            tracker=tracker,
        )

        if lr_scheduler is not None:
            # check if val_loss in np array or not
            vloss = val_loss if not isinstance(val_loss, np.ndarray) else val_loss[1]
            # v = val_loss if not isinstance(val_loss, )
            # Update the learning rate
            lr_scheduler.step(vloss)

    return epoch, train_loss, val_loss


# def wandb_epoch_logger(
#     epoch,
#     train_loss,
#     train_rmse,
#     train_r2,
#     train_evs,
#     val_loss,
#     val_rmse,
#     val_r2,
#     val_evs,
# ):
#     wandb.log(
#         data={
#             "epoch": epoch,
#             "train/loss": train_loss,
#             "train/rmse": train_rmse,
#             "train/r2": train_r2,
#             "train/evs": train_evs,
#             "val/loss": val_loss,
#             "val/rmse": val_rmse,
#             "val/r2": val_r2,
#             "val/evs": val_evs,
#         }
#     )
#
#
# def wandb_test_logger(
#     test_loss,
#     test_rmse,
#     test_r2,
#     test_evs,
#     tasks_rmse=None,
#     tasks_r2=None,
#     tasks_evs=None,
# ):
#     wandb.log(
#         data={
#             "test/loss": test_loss,
#             "test/rmse": test_rmse,
#             "test/r2": test_r2,
#             "test/evs": test_evs,
#         }
#     )
#     if tasks_rmse is not None:
#         for task_idx in range(len(tasks_rmse)):
#             wandb.log(
#                 data={
#                     f"test/task_{task_idx}/rmse": tasks_rmse[task_idx],
#                     f"test/task_{task_idx}/r2": tasks_r2[task_idx],
#                     f"test/task_{task_idx}/evs": tasks_evs[task_idx],
#                 }
#             )


def train_model(
    model,
    config,
    train_loader,
    val_loader,
    n_targets=-1,
    seed=42,
    device=DEVICE,
    logger=None,
    max_norm=None,  # 10.0
    tracker="wandb",
):
    try:
        set_seed(seed)
        multitask = n_targets > 1

        model = model.to(device)
        # best_model = model
        # best_model_params = model.state_dict()
        best_val_loss = float("inf")
        early_stop_counter = 0
        early_stop_criteria = int(config.get("early_stop", 10))
        optimizer = build_optimizer(
            model,
            config.get("optimizer", "adam"),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.0),
        )

        loss_fn = build_loss(
            config.get("loss", "mse"),
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

        # start empty np array
        # results_arr = np.array([], dtype=np.float32)
        results_arr = []
        # "none", "mean", "sum"
        # get a random number to add to the name of the best modelfor checkpointing
        random_num = np.random.randint(0, 10000000000)

        # create ckpting path here

        for epoch in tqdm(range(config.get("epochs", 10)), desc="Epochs"):
            # for epoch in range(config.epochs + 1):
            try:
                lossfname = config.get("loss", "mse")
                (
                    epoch,
                    train_loss,  # this can be an array if tracker is tensor
                    val_loss,
                ) = run_one_epoch(
                    model,
                    train_loader,
                    val_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    epoch=epoch,
                    device=device,
                    max_norm=max_norm,
                    lossfname=lossfname,
                    tracker=tracker,
                )
                vloss = (
                    val_loss if not isinstance(val_loss, np.ndarray) else val_loss[1]
                )
                if vloss < best_val_loss:
                    best_val_loss = vloss
                    early_stop_counter = 0
                    config = ckpt(model, config)
                    # best_model = model
                    # save the best model state dict to var
                    # torch.save(model.state_dict(), MODELS_DIR / "best_model_ckpt.pth")
                    # best_model_params = copy.deepcopy(model.state_dict())

                else:
                    early_stop_counter += 1
                    if early_stop_counter > early_stop_criteria:
                        break
                if tracker.lower() == "tensor":
                    results_arr.append(np.append(train_loss, val_loss[1:]))
                    # results_tensor.append(train_loss)
                    # results_tensor.append(val_loss[1:])

            except Exception as e:
                raise RuntimeError(
                    f"The following exception occurred inside the epoch loop {e}"
                )

        # TODO train once on validation set - check b and best_model before and after
        # b = best_model.state_dict()
        best_model = load_ckpt(model, config)
        # TODO : check here how the hell trainig on validation increases the loss
        # 1st: 2.250059547168868
        # 2nd: 2.280138745903969
        #
        # # best_model = model.load_state_dict(best_model_params)
        # # Just for the testing - One iteration is not sufficient.
        # At some point doing cross-validation here is better.
        # val_loss = evaluate(
        #     best_model,
        #     val_loader,
        #     loss_fn,
        #     device,
        #     False,
        #     "val",
        #     epoch + 1,
        #     lossfname,
        #     tracker=tracker,
        # )
        # print(f"1st: {val_loss}")
        # _ = train(
        #     best_model,
        #     val_loader,
        #     loss_fn,
        #     optimizer,
        #     device,
        #     epoch + 2,
        #     max_norm=max_norm,
        #     lossfname=lossfname,
        #     tracker=tracker,
        #     subset="val",
        # )
        # # Just for the testing
        # val_loss = evaluate(
        #     best_model,
        #     val_loader,
        #     loss_fn,
        #     device,
        #     False,
        #     "val",
        #     epoch + 3,
        #     lossfname,
        #     tracker=tracker,
        # )
        # print(f"2nd: {val_loss}")
        if tracker.lower() == "tensor":
            # stack all the list of arrays on dim 1
            results_arr = np.stack(results_arr, axis=0)

        return best_model, loss_fn, results_arr
    except Exception as e:
        raise RuntimeError(f"The following exception occurred in train_model {e}")


def run_model(
    config,
    model,
    dataloaders,
    device=DEVICE,
    logger=None,  # create_logger("run_model"),
    max_norm=None,  # 10.0
    tracker="wandb",
):
    if logger is None:
        logger = create_logger("run_model")
    seed = config.get("seed", 42)
    n_targets = config.get("n_targets", -1)
    # aleatoric = config.get("aleatoric", False)
    mt = n_targets > 1
    # Train the model
    best_model, loss_fn, results_arr = train_model(
        model,
        config,
        dataloaders["train"],
        dataloaders["val"],
        n_targets,
        seed,
        device,
        logger=logger,
        max_norm=max_norm,
        tracker=tracker,
    )

    # Testing metrics on the best model
    # TODO I think this might need another look when the tracker is not wandb but tensor
    test_loss = evaluate(
        best_model,
        dataloaders["test"],
        loss_fn,
        device,
        metrics_per_task=mt,
        subset="test",
        epoch=None,
        lossfname=config.get("loss", "mse"),
        tracker=tracker,
    )

    # TODO FOR TESTING
    # outputs_all, targets_all, vars_all = predict(best_model, dataloaders["test"], aleatoric=aleatoric, device=device)

    return best_model, test_loss, results_arr


# def model_vmap(models, x):
#     # https://pytorch.org/tutorials//intermediate/ensembling.html#using-vmap-to-vectorize-the-ensemble
#     from torch.func import stack_module_state
#     params, buffers = stack_module_state(models)
#
#     from torch.func import functional_call
#     import copy
#
#     base_model = copy.deepcopy(models[0])
#     base_model = base_model.to('meta')
#
#     def fmodel(params, buffers, x):
#         return functional_call(base_model, (params, buffers), (x,))
#
#     from torch import vmap
#     predictions1_vmap = vmap(fmodel)(params, buffers, minibatches)


def assign_wandb_tags(run, config):
    median_scaling = config.get("median_scaling", False)
    m_tag = "median_scaling" if median_scaling else "no_median_scaling"
    mt = config.get("MT", False)
    mt_tag = "MT" if mt else "ST"
    wandb_tags = [
        config.get("model_type", "baseline"),
        config.get("data_name", "papyrus"),
        config.get("activity_type", "xc50"),
        config.get("descriptor_protein", None),
        config.get("descriptor_chemical", None),
        config.get("split_type", "random"),
        config.get("task_type", "regression"),
        m_tag,
        mt_tag,
        f"max_norm={config.get('max_norm', None)}",
        TODAY,
        config.get("tags", None),
        f"ev_lamb={config.get('lamb', None)}",
    ]
    # filter out None values
    wandb_tags = [tag for tag in wandb_tags if tag]
    run.tags += tuple(wandb_tags)
    return run


def get_dataloader(
    config,
    device=DEVICE,
    logger=None,
):
    data_name = config.get("data_name", "papyrus")
    activity_type = config.get("activity_type", "xc50")
    n_targets = config.get("n_targets", -1)
    descriptor_protein = config.get("descriptor_protein", None)
    descriptor_chemical = config.get("descriptor_chemical", None)
    median_scaling = config.get("median_scaling", False)
    split_type = config.get("split_type", "random")
    ext = config.get("ext", "pkl")
    task_type = config.get("task_type", "regression")

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
        device=device,
    )

    batch_size = config.get("batch_size", 128)
    dataloaders = build_loader(datasets, batch_size, shuffle=False)

    return dataloaders


def post_training_save_model(
    model,
    config,
    model_type="baseline",
    onnx=True,
    tracker="wandb",
    run=None,
    logger=None,
    write_model=True,
):
    config["model_type"] = model_type
    model_name = get_model_name(config, run=run)
    data_specific_path = get_data_specific_path(config, logger=logger)
    config["data_specific_path"] = data_specific_path
    # data_name = config.get("data_name", "papyrus")
    # activity_type = config.get("activity_type", "xc50")
    # n_targets = config.get("n_targets", -1)

    # split_type = config.get("split_type", "random")
    #
    # model_name = (
    #     f"{TODAY}-{model_type}_{split_type}_{descriptor_protein}_{descriptor_chemical}"
    # )
    # if run:
    #     model_name += f"_{run.name}"
    # model_name += f"{run.name}" if tracker.lower() == "wandb" else ""
    descriptor_protein = config.get("descriptor_protein", None)
    descriptor_chemical = config.get("descriptor_chemical", None)
    desc_prot_len, desc_chem_len = get_desc_len(descriptor_protein, descriptor_chemical)

    # config["model_name"] = model_name
    # data_specific_path = Path(data_name) / activity_type / get_topx(n_targets)
    # if logger:
    #     logger.debug(f"Data specific path: {data_specific_path}")

    if write_model:
        save_model(
            config,
            model,
            model_name,
            data_specific_path,
            desc_prot_len,
            desc_chem_len,
            onnx=onnx,
            tracker=tracker,
        )

    return model_name


def get_tracker(config, tracker="wandb"):
    if tracker.lower() == "wandb":
        if config is not None:
            run = wandb.init(
                config=config,
                dir=WANDB_DIR,
                mode=WANDB_MODE,
                project=config.get("wandb_project_name", "test-project"),
                reinit=True,
            )
            config = wandb.config
            run = assign_wandb_tags(run, config)
        else:
            run = wandb.init(dir=WANDB_DIR, mode=WANDB_MODE)
            config = wandb.config
    else:
        run = None

    return run, config
    # if tracker.lower() == "wandb":
    #     if config is not None:
    #         wandb_project_name = config.get(
    #             "wandb_project_name", "test-project"
    #         )  # add it to config only in baseline not the sweep
    #         run = wandb.init(
    #             config=config, dir=WANDB_DIR, mode=WANDB_MODE, project=wandb_project_name, reinit=False
    #         )
    #     else:  # TODO check the effect here?
    #         run = wandb.init(config=config, dir=WANDB_DIR, mode=WANDB_MODE)
    #
    #     config = wandb.config


def train_model_e2e(
    config,
    model,
    model_type="baseline",
    model_kwargs=None,
    logger=None,
    seed=42,
    device=DEVICE,
    tracker="wandb",
    write_model=True,
):
    # start_time = datetime.now()
    # logger = (
    #     create_logger(name=model_type, file_level="debug", stream_level="info")
    #     if not logger
    #     else logger
    # )

    if model_kwargs is None:
        model_kwargs = {}

    run, config = get_tracker(config, tracker=tracker)

    n_targets = config.get("n_targets", -1)
    descriptor_protein = config.get("descriptor_protein", None)
    descriptor_chemical = config.get("descriptor_chemical", None)
    split_type = config.get("split_type", "random")
    max_norm = config.get("max_norm", None)
    seed = config.get("seed", seed)  # it has to be in that order here

    set_seed(seed)
    assert split_type in [
        "random",
        "scaffold",
        "time",
        "scaffold_cluster",
    ], "Split type must be either random or scaffold or scaffold_cluster or time"

    mt = n_targets > 1
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
            f"Setting split_type={split_type} to random"
        )
        config["split_type"] = "random"

    config["prot_input_dim"], config["chem_input_dim"] = get_desc_len(
        descriptor_protein, descriptor_chemical, logger=logger
    )

    model_ = model(config=config, **model_kwargs, logger=logger).to(device)

    dataloaders = get_dataloader(config, device=device, logger=logger)
    best_model, test_loss, results_arr = run_model(
        config,
        model_,
        dataloaders,
        device=device,
        logger=logger,
        max_norm=max_norm,
        tracker=tracker,
    )
    config["model_name"] = post_training_save_model(
        best_model,
        config,
        model_type=model_type,
        tracker=tracker,
        run=run,
        logger=logger,
        write_model=write_model,
    )

    return best_model, config, results_arr, test_loss
    # return best_model, dataloaders, config, logger, results_arr

    # logger.info(f"{model_type} - start time: {start_time}")
    # # Adding WANDB TAGS to the tracker if.
    # if tracker.lower() == "wandb":
    #     run = assign_wandb_tags(run, config)
    #     # m_tag = "median_scaling" if median_scaling else "no_median_scaling"
    #     # mt_tag = "MT" if mt else "ST"
    #     # wandb_tags = [
    #     #     model_type,
    #     #     data_name,
    #     #     activity_type,
    #     #     descriptor_protein,
    #     #     descriptor_chemical,
    #     #     split_type,
    #     #     task_type,
    #     #     m_tag,
    #     #     mt_tag,
    #     #     f"max_norm={max_norm}",
    #     #     TODAY,
    #     #     tags
    #     # ]
    #     # # filter out None values
    #     # wandb_tags = [tag for tag in wandb_tags if tag]
    #     # run.tags += tuple(wandb_tags)

    # datasets = build_datasets(
    #     data_name,
    #     n_targets,
    #     activity_type,
    #     split_type,
    #     descriptor_protein,
    #     descriptor_chemical,
    #     median_scaling,
    #     task_type,
    #     ext,
    #     logger,
    #     device=device,
    # )
    # batch_size = config.get("batch_size", 128)
    # dataloaders = build_loader(datasets, batch_size, shuffle=False)
    # logger.info(f"Model: {model_}")
    # logger.debug(f"{model_type} - end time: {datetime.now()}")
    # logger.debug(f"{model_type} - duration: {datetime.now() - start_time}")


def recalibrate_model(
    preds_val,
    labels_val,
    preds_test,
    labels_test,
    config,
    epi_val=None,
    epi_test=None,
    uct_logger=None,
):
    model_name = config.get("model_name", "ensemble")
    data_specific_path = config.get("data_specific_path", None)

    figures_path = FIGS_DIR / data_specific_path / model_name

    # Validation Set
    y_true_val, y_pred_val, y_std_val, y_err_val, _ = process_preds(
        preds_val, labels_val, epi_vars=epi_val
    )
    y_true_test, y_pred_test, y_std_test, y_err_test, _ = process_preds(
        preds_test, labels_test, epi_vars=epi_test
    )

    iso_recal_model, std_recal = recalibrate(
        y_true_val,
        y_pred_val,
        y_std_val,
        y_err_val,
        y_true_test,
        y_pred_test,
        y_std_test,
        y_err_test,
        n_subset=None,
        savefig=True,
        save_dir=figures_path,
        uct_logger=uct_logger,
    )
    # # TODO add task_name
    # model_dir = (
    #     MODELS_DIR / "saved_models" / data_specific_path
    #     if data_specific_path
    #     else MODELS_DIR / "saved_models"
    # )
    # model_dir.mkdir(exist_ok=True)
    # model_name = config.get("model_name", "ensemble") + "_recalibrate_model.pkl"
    # # pickle save the model to model_dir
    # save_pickle(recal_model, model_dir / model_name)

    # Test Set
    # return recal_model
    return iso_recal_model, std_recal  # , metrics, plots


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
