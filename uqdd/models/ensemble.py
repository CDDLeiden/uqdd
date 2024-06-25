import argparse
import logging
from multiprocessing import Pool, Manager, Queue, Lock
import time

import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from uqdd import DEVICE, WANDB_DIR, WANDB_MODE, DATASET_DIR
from uqdd.models.baseline import BaselineDNN
from uqdd.utils import create_logger, parse_list, save_pickle, split_list_by_sizes

from uqdd.models.utils_train import (
    train_model_e2e,
    evaluate_predictions,
    predict,
    recalibrate_model,
    assign_wandb_tags,
    get_dataloader,
    post_training_save_model,
)

from uqdd.models.utils_models import (
    get_model_config,
    get_sweep_config,
    set_seed,
)

mp.set_start_method("spawn", force=True)
# torch.cuda.memory._set_allocator_settings('expandable_segments:False')


class EnsembleDNN(nn.Module):
    def __init__(self, config=None, model_class=BaselineDNN, model_list=None, **kwargs):
        super(EnsembleDNN, self).__init__()
        if config is None:
            config = get_model_config(model_name="ensemble", **kwargs)
        # self.ensemble_size = ensemble_size
        self.config = config
        self.logger = create_logger(name="EnsembleDNN")
        self.ensemble_size = config.get("ensemble_size", 100)
        # self.aleatoric = config.get("aleatoric", False)
        # self.aleatoric = False
        if model_list is not None:
            models = model_list
        else:
            models = []
            seed = config.get("seed", 42)
            for _ in range(self.ensemble_size):
                set_seed(seed)
                seed += 1
                model = model_class(config, **kwargs)
                models.append(model)
        self.models = nn.ModuleList(models)
        # self.models = nn.ModuleList(
        #     [model_class(config, **kwargs) for _ in range(self.ensemble_size)]
        # )

    def forward(self, inputs):
        outputs = []
        vars_ = []
        for model in self.models:
            output, var_ = model(inputs)
            outputs.append(output)
            vars_.append(var_)
        outputs = torch.stack(
            outputs, dim=2
        )  # Shape: [batch_size, output_dim, ensemble_size]
        vars_ = torch.stack(
            vars_, dim=2
        )  # Shape: [batch_size, output_dim, ensemble_size]
        return outputs, vars_


def log_wandb_ensemble(results_tensor_avg, config):
    wandb_keys = [
        "epoch",
        "train/loss",
        "train/rmse",
        "train/r2",
        "train/evs",
        "train/alea_mean",
        "train/alea_var",
        "model/pnorm",
        "model/gnorm",
        "val/loss",
        "val/rmse",
        "val/r2",
        "val/evs",
        "val/alea_mean",
        "val/alea_var",
    ]
    n_targets = config.get("n_targets", -1)
    if n_targets > 1:
        for task in n_targets:
            wandb_keys += [f"val/rmse/{task}", f"val/r2/{task}", f"val/evs/{task}"]

    # iterate over the metrics and log them to wandb
    num_epochs, num_metrics = results_tensor_avg.shape

    for epoch in range(num_epochs):
        wandb.log(
            # all data except epoch
            data=dict(zip(wandb_keys[1:], results_tensor_avg[epoch, 1:])),
            step=epoch,  # int(results_tensor_avg[epoch, 0]),
        )
    # run.finish()


def log_wandb_test(test_tensor_avg, config):
    wandb_keys = [
        "test/loss",
        "test/rmse",
        "test/r2",
        "test/evs",
        "test/alea_mean",
        "test/alea_var",
    ]
    # multitask = config.get("MT", False)
    n_targets = config.get("n_targets", -1)

    if n_targets > 1:  # MT
        for task in n_targets:
            wandb_keys += [
                f"test/rmse/task_{task}",
                f"test/r2/task_{task}",
                f"test/evs/task_{task}",
            ]

    test_data = dict(zip(wandb_keys, test_tensor_avg[1:]))
    wandb.log(test_data)

    # # iterate over the metrics and log them to wandb
    # num_epochs, num_metrics = test_tensor_avg.shape
    #
    # for epoch in range(num_epochs):
    #     wandb.log(
    #         # all data except epoch
    #         data=dict(zip(wandb_keys, test_tensor_avg[epoch, :])),
    #         step=epoch,  # int(results_tensor_avg[epoch, 0]),
    #     )
    # # run.finish()


def fill_to_max_epochs(array, max_epochs):
    num_metrics = array.shape[1]
    filled_array = np.full((max_epochs, num_metrics), np.nan)
    filled_array[: array.shape[0], : array.shape[1]] = array
    return filled_array


def process_results_arrs(result_arrs, test_arrs, config, logger):
    try:
        # get the maximum number of epochs
        max_epochs = max([results_arr.shape[0] for results_arr in result_arrs])
        # fill the results to max epochs
        result_arrs = [
            fill_to_max_epochs(results_arr, max_epochs) for results_arr in result_arrs
        ]
        # now we stack result tensors on dim 2
        result_arrs = np.stack(result_arrs, axis=2)
        logger.debug(f"{result_arrs.shape=}")
        # this should equal to (num_epochs, metrics_collected, ensemble_size)

        # Take average across model metrics
        # results_tensor_avg = result_arrs.nanmean(2)
        results_tensor_avg = np.nanmean(result_arrs, 2)
        print(f"{results_tensor_avg.shape=}")
        # HERE we should report to wandb
        log_wandb_ensemble(results_tensor_avg, config)

        # Test Arrs
        test_arrs = np.stack(test_arrs, axis=1)
        test_tensor_avg = np.nanmean(test_arrs, 1)
        logger.debug(f"{test_tensor_avg.shape=}")

        log_wandb_test(test_tensor_avg, config)

    except Exception as e:
        logger.exception(f"Error in stacking results: {e}")

    finally:
        # Here we want to save a pkl file with the results tensor
        save_pickle(
            result_arrs,
            DATASET_DIR / config.get("data_specific_path") / "ensemble_results.pkl",
        )

    return result_arrs


# def gpu_manager(gpu_id, available_gpus, lock):
#     with lock:
#         gpu = available_gpus[gpu_id % len(available_gpus)]
#         while gpu in gpu_id:
#             gpu_id += 1
#             gpu = available_gpus[gpu_id % len(available_gpus)]
#         return gpu
# def gpu_manager(available_gpus, gpu_status, lock):
#     with lock:
#         for idx, gpu in enumerate(available_gpus):
#             if not gpu_status[idx]:  # If the GPU is not busy
#                 gpu_status[idx] = True
#                 return gpu, idx
#         return None, -1  # No GPU available
#
#
# def release_gpu(gpu_idx, gpu_status, lock):
#     with lock:
#         gpu_status[gpu_idx] = False

# def train_worker(rank, config, seed, results, device, logger, lock):
# # def train_worker(rank, config, seed, results, available_gpus, lock, logger):
#     # seed += rank
#     # device = gpu_manager(rank, available_gpus, lock)
#     with lock:
#         torch.cuda.set_device(device)
#
#         best_model, config_, results_arr = train_model_e2e(
#             config, model=BaselineDNN, model_type="baseline", logger=logger, seed=seed, device=device, tracker="tensor"
#         )
#         # print(rank)
#         results[rank] = (best_model.cpu(), config_, results_arr)
#         # clear memory cache on GPU
#         torch.cuda.empty_cache()
#
#
# def train_on_device(args):
#     # rank, config, results, available_gpus, lock, logger = args
#     # seed = config.get("seed", 42)
#     # train_worker(rank, config, seed + rank, results, available_gpus, lock, logger)
#     # rank, config, results, available_gpus, num_gpus, ens_size, logger = args
#     rank, config, results, device, logger, seed, lock, max_retries = args
#     # seed = config.get("seed", 42)
#     # device = available_gpus[rank % num_gpus]
#     # print(f"{device=}")
#     retries = 0
#     while retries < max_retries:
#         try:
#             train_worker(rank, config, seed, results, device, logger, lock)
#             break
#         except Exception as e:
#             logger.exception(f"Process {rank} failed with error: {e}. Retrying {retries}/{max_retries}...")
#             time.sleep(5)
#             retries += 1
#     # try:
#     #     train_worker(rank, config, seed + rank, results, device, logger, lock)
#     # except Exception as e:
#     #     logger.exception(f"Process {rank} failed with error: {e}. Retrying in 5 seconds...")
#     #     time.sleep(5)
#     #     train_on_device(args)
#
#
# def train_on_device_(args):
#     # rank, config, results, available_gpus, lock, logger = args
#     # seed = config.get("seed", 42)
#     # train_worker(rank, config, seed + rank, results, available_gpus, lock, logger)
#     # rank, config, results, available_gpus, num_gpus, ens_size, logger = args
#     seq, config, results, device, logger, seed, max_retries = args
#
#     torch.cuda.set_device(device)
#     for i, rank in enumerate(range(seq)):
#         retries = 0
#         while retries < max_retries:
#             try:
#                 best_model, config_, results_arr = train_model_e2e(
#                     config, model=BaselineDNN, model_type="baseline", logger=logger, seed=seed, device=device, tracker="tensor"
#                 )
#                 # print(rank)
#                 results[rank] = (best_model.cpu(), config_, results_arr)
#     # clear memory cache on GPU
#     torch.cuda.empty_cache()
#
#     retries = 0
#     while retries < max_retries:
#         try:
#             train_worker(rank, config, seed, results, device, logger, lock)
#             break
#         except Exception as e:
#             logger.exception(f"Process {rank} failed with error: {e}. Retrying {retries}/{max_retries}...")
#             time.sleep(5)
#             retries += 1


# def parallel_train_ensemble(ensemble_size, config, logger):
#     best_models = []
#     result_arrs = []
#     if torch.cuda.is_available():
#         logger.info("Parallel training on several GPUs")
#         # Get the available GPUs
#         available_devices = list(range(torch.cuda.device_count()))
#         num_processes = len(available_devices)
#         logger.info(f"Number of Available GPUs: {num_processes} GPUs")
#
#     else:
#         logger.info("Parallel training on several CPUs")
#         num_processes = mp.cpu_count()
#         # available_devices = list(range(num_processes))
#         available_devices = num_processes * ['cpu']
#         logger.info(f"Number of Available CPUs: {num_processes} CPUs")
#
#     # how many models to run on each gpu
#     seq_models = [ensemble_size // num_processes] * num_processes
#
#     for i in range(ensemble_size % num_processes):
#         seq_models[i] += 1
#
#     manager = Manager()
#     results = manager.dict()
#     # args_queue = Queue()
#     # lock = manager.Lock()
#     # for i in range(num_processes):
#     #     args_queue.put(i)
#     args = []
#     seed = config.get("seed", 42)
#
#     seedings = list(range(seed, seed + ensemble_size))
#     rankings = list(range(ensemble_size))
#     rank_seed = list(zip(rankings, seedings))
#     model_per_device = split_list_by_sizes(rank_seed, seq_models)
#
#     max_retries = 3
#     for i, s in enumerate(seq_models):
#         device = available_devices[i]
#         args.append((s, config, results, device, logger, seed, max_retries))
#         seed += s
#         config["seed"] = seed
#
#     # # chunking
#     # chunk_size = ensemble_size // num_processes * [num_processes]
#     # chunk_size.append(ensemble_size % num_processes)
#     # for rank in range(ensemble_size):
#     #     print(seed)
#     #     device = args_queue.get()
#     #     # device = available_devices[rank % num_processes]
#     #     args.append((rank, config, results, device, logger, seed, lock, max_retries))
#     #     # args_queue.put(device)
#     #     seed += 1
#     #     config["seed"] = seed
#
#     with Pool(processes=num_processes) as pool:
#         pool.map(train_on_device, args)
#         pool.map(train_on_device_, args)
#
#     for rank in range(ensemble_size):
#         # best_model, dataloaders, config_, results_arr = results[rank]
#         best_model, config_, results_arr = results[rank]
#         best_models.append(best_model)
#         result_arrs.append(results_arr)
#     if len(best_models) < ensemble_size:
#         # get how many are left
#         num_models_left = ensemble_size - len(best_models)
#         b_models_left, res_arrs_left, config_ = parallel_train_ensemble(num_models_left, config, logger)
#         best_models.extend(b_models_left)
#         result_arrs.extend(res_arrs_left)
#
#     return best_models, result_arrs, config_


def run_ensemble(config=None):
    ensemble_size = config.get("ensemble_size", 100)
    # parallelize = config.get("parallelize", False)
    logger = LOGGER
    best_models = []
    result_arrs = []
    test_arrs = []

    # Here we should init the wandb to track the resources
    # start wandb run
    run = wandb.init(
        config=config,
        dir=WANDB_DIR,
        mode=WANDB_MODE,
        project=config.get("wandb_project_name", "ensemble_test"),
        reinit=True,
    )

    assign_wandb_tags(run, config)

    # if not parallelize:
    for _ in range(ensemble_size):
        # best_model, dataloaders, config_, logger, results_arr = train_model_e2e(
        # For debugging of different sizes results
        # config["epochs"] += 1
        best_model, config_, results_arr, test_arr = train_model_e2e(
            config,
            model=BaselineDNN,
            model_type="ensemble",
            logger=logger,
            tracker="tensor",
            write_model=False,
        )
        best_models.append(best_model)
        config["seed"] += 1

        result_arrs.append(results_arr)
        test_arrs.append(test_arr)

    # else:
    #     best_models, result_arrs, config_ = parallel_train_ensemble(ensemble_size, config, logger)del model
    # gc.collect()
    # torch.cuda.empty_cache()

    process_results_arrs(result_arrs, test_arrs, config_, logger)

    logger.debug(f"{len(best_models)=}")
    ensemble_model = EnsembleDNN(config_, model_list=best_models).to(DEVICE)

    # we should save the best_models here
    config_["model_name"] = post_training_save_model(
        ensemble_model,
        config_,
        model_type="ensemble",
        tracker="wandb",
        run=run,
        logger=logger,
        write_model=True,
    )

    dataloaders = get_dataloader(config, device=DEVICE, logger=LOGGER)

    preds, labels, alea_vars = predict(
        ensemble_model, dataloaders["test"], device=DEVICE
    )

    # Then comes the predict metrics part
    metrics, plots, uct_logger = evaluate_predictions(
        config_, preds, labels, alea_vars, "ensemble", logger, wandb_push=False
    )

    # RECALIBRATION # Get Calibration / Validation Set
    preds_val, labels_val, alea_vars_val = predict(
        ensemble_model, dataloaders["val"], device=DEVICE
    )
    iso_recal_model, std_recal = recalibrate_model(
        preds_val, labels_val, preds, labels, config_, uct_logger=uct_logger
    )

    uct_logger.wandb_log()
    wandb.finish()

    return ensemble_model, iso_recal_model, std_recal, metrics, plots


def run_ensemble_wrapper(**kwargs):
    global LOGGER
    LOGGER = create_logger(name="ensemble", file_level="debug", stream_level="info")
    config = get_model_config(model_name="ensemble", **kwargs)
    return run_ensemble(config)

    # best_model, recal_model, metrics, plots = run_ensemble(config)
    # return best_model, recal_model, metrics, plots


def run_ensemble_hyperparm(**kwargs):
    global LOGGER
    LOGGER = create_logger(
        name="ensemble-sweep", file_level="debug", stream_level="info"
    )

    sweep_count = kwargs.pop("sweep_count")
    wandb_project_name = kwargs.pop("wandb_project_name")

    config = get_sweep_config("ensemble", **kwargs)
    config["project"] = wandb_project_name
    sweep_id = wandb.sweep(
        config,
        project=wandb_project_name,
    )
    print(f"Running sweep with SWEEP_ID: {sweep_id}")
    wandb.agent(sweep_id, function=run_ensemble, count=sweep_count)


# if __name__ == "__main__":
#     ensemble_model, iso_recal_model, std_recal, metrics, plots = run_ensemble_wrapper(
#         data_name="papyrus",
#         activity_type="xc50",
#         n_targets=-1,
#         descriptor_protein="ankh-large",
#         descriptor_chemical="ecfp2048",
#         median_scaling=False,
#         split_type="random",
#         ext="pkl",
#         task_type="regression",
#         wandb_project_name="ensemble-test",
#         ensemble_size=5,
#         epochs=5,
#         seed=440,
#     )
#     #
#     print("Done!")
# TEST
# run_ensemble(
#     data_name="papyrus",
#     activity_type="xc50",
#     n_targets=-1,
#     descriptor_protein="ankh-base",
#     descriptor_chemical="ecfp2048",
#     median_scaling=False,
#     split_type="random",
#     ext="pkl",
#     task_type="regression",
#     wandb_project_name=f"{TODAY}-ensemble-test",
#     ensemble_size=5,
# )
# if rank != ens_size - 1:
#     print("I AM HERE")
#     wandb.finish()
#     torch.cuda.empty_cache()
#     time.sleep(10)
#     print("I AM AWAKE")
# if torch.cuda.is_available():
#     logger.info("Parallel training on several GPUs")
#     # Get the available GPUs
#     available_gpus = list(range(torch.cuda.device_count()))
#     num_processes = len(available_gpus)
#     # num_processes = num_gpus
#     logger.info(f"Number of Available GPUs: {num_processes} GPUs")
#
# else:
#     logger.info("Parallel training on several CPUs")
#     num_processes = mp.cpu_count()
#     logger.info(f"Number of Available CPUs: {num_processes} CPUs")
# # lock = manager.Lock()
# # Prepare arguments for the pool workers
# # args = [(rank, config, results, available_gpus, num_gpus, ensemble_size, logger) for rank in range(ensemble_size)]
# # args = [(rank, config, results, available_gpus, lock, logger) for rank in range(ensemble_size)]
# # Create a pool of workers
# # Shared memory to store results from different processes
# manager = Manager()
# results = manager.dict()
# args_queue = Queue()
# lock = manager.Lock()
# for i in range(num_processes):
#     args_queue.put(i)
#
# args = []
# seed = config.get("seed", 42)
# max_retries = 3
# for rank in range(ensemble_size):
#     device = args_queue.get()
#     args.append((rank, config, results, device, logger, seed, lock, max_retries))
#     args_queue.put(device)
#     seed += 1
#     config["seed"] = seed
#
# with Pool(processes=num_processes) as pool:
#     pool.map(train_on_device, args)
#
# for rank in range(ensemble_size):
#     # best_model, dataloaders, config_, results_arr = results[rank]
#     best_model, config_, results_arr = results[rank]
#     best_models.append(best_model)
#     result_arrs.append(results_arr)
# if len(best_models) != ensemble_size:
#     # get how many are left
#     num_models_left = ensemble_size - len(best_models)
#
#     # raise ValueError(f"Number of models {len(best_models)} does not match ensemble size {ensemble_size}")
# full_runs = ensemble_size // num_gpus
# remaining_runs = ensemble_size % num_gpus
# for r in range(full_runs):
#     args = [(rank, config, seed, results, available_gpus, num_gpus, ensemble_size, logger) for rank in range(num_gpus)]
#     with Pool(processes=num_gpus) as pool:
#         pool.map(train_on_device, args)
#        #
# # Spawn processes for parallel training
# processes = []
# for rank in range(ensemble_size):
#     device = available_gpus[rank % ]
#     print(f"Rank: {rank}, Device: {device}")
#     p = mp.Process(target=train_worker, args=(rank, config, seed, results, device), )
#     p.start()
#     processes.append(p)
#
# # Ensure all processes have finished
# for p in processes:
#     p.join()

# Collect results from shared memory
# # Spawn processes for parallel training
# processes = []
# for rank in range(ensemble_size):
#     device = available_gpus[rank % ]
#     print(f"Rank: {rank}, Device: {device}")
#     p = mp.Process(target=train_worker, args=(rank, config, seed, results, device), )
#     p.start()
#     processes.append(p)
#
# # Ensure all processes have finished
# for p in processes:
#     p.join()

# Collect results from shared memory
# def _run_ensemble(config=None):
#
#     best_model, dataloaders, config, logger = train_model_e2e(
#         config,
#         model=EnsembleDNN,
#         model_type="ensemble",
#         model_kwargs={
#             "model_class": BaselineDNN,
#             # "ensemble_size": config.get("ensemble_size", 100),
#         },
#         logger=LOGGER,
#     )
#
#     aleatoric = config.get("aleatoric", False)
#     preds, labels, alea_vars = predict(
#         best_model, dataloaders["test"], device=DEVICE  #aleatoric=aleatoric,
#     )
#
#     # Then comes the predict metrics part
#     metrics, plots, uct_logger = evaluate_predictions(
#         config,
#         preds,
#         labels,
#         alea_vars,
#         "ensemble",
#         logger
#     )
#
#     # RECALIBRATION # Get Calibration / Validation Set
#     preds_val, labels_val, alea_vars_val = predict(
#         best_model, dataloaders["val"], device=DEVICE  # aleatoric=aleatoric,
#     )
#     recal_model = recalibrate_model(preds_val, labels_val, preds, labels, config, uct_logger=uct_logger)
#
#     return best_model, recal_model, metrics, plots


# def main():
#     parser = argparse.ArgumentParser(description="Run Ensemble Model")
#     parser.add_argument(
#         "--parallelize",
#         type=bool,
#         default=False,
#         help="Parallelize training"
#     )
#     parser.add_argument(
#         "--aleatoric",
#         type=bool,
#         default=True,
#         help="Aleatoric inference"
#     )
#     parser.add_argument(
#         "--ensemble_size",
#         type=int,
#         default=100,
#         help="Size of the ensemble",
#     )
#     parser.add_argument(
#         "--data_name",
#         type=str,
#         default="papyrus",
#         choices=["papyrus", "tdc", "other"],
#         help="Data name argument",
#     )
#     parser.add_argument(
#         "--activity_type",
#         type=str,
#         default="xc50",
#         choices=["xc50", "kx"],
#         help="Activity argument",
#     )
#     parser.add_argument(
#         "--n_targets",
#         type=int,
#         default=-1,
#         help="Number of targets argument (default=-1 for all targets)",
#     )
#     parser.add_argument(
#         "--descriptor_protein",
#         type=str,
#         default=None,
#         choices=[
#             None,
#             "ankh-base",
#             "ankh-large",
#             "unirep",
#             "protbert",
#             "protbert_bfd",
#             "esm1_t34",
#             "esm1_t12",
#             "esm1_t6",
#             "esm1b",
#             "esm_msa1",
#             "esm_msa1b",
#             "esm1v",
#         ],
#         help="Protein descriptor argument",
#     )
#     parser.add_argument(
#         "--descriptor_chemical",
#         type=str,
#         default="ecfp2048",
#         choices=[
#             "ecfp1024",
#             "ecfp2048",
#             "mold2",
#             "mordred",
#             "cddd",
#             "fingerprint",  # "moldesc"
#         ],
#         help="Chemical descriptor argument",
#     )
#     parser.add_argument(
#         "--median_scaling",
#         action="store_true",
#         help="Use median scaling",
#     )
#     parser.add_argument(
#         "--split_type",
#         type=str,
#         default="random",
#         choices=["random", "scaffold", "time", "scaffold_cluster"],
#         help="Split argument",
#     )
#     parser.add_argument(
#         "--ext",
#         type=str,
#         default="pkl",
#         choices=["pkl", "parquet", "csv", "feather"],
#         help="File extension argument",
#     )
#     parser.add_argument(
#         "--task_type",
#         type=str,
#         default="regression",
#         choices=["regression", "classification"],
#         help="Task type argument",
#     )
#     parser.add_argument(
#         "--wandb-project-name",
#         type=str,
#         default="ensemble-test",
#         help="Wandb project name argument",
#     )
#     parser.add_argument(
#         "--sweep-count",
#         type=int,
#         default=None,
#         help="Sweep count argument",
#     )
#     # take chem layers as list input
#     parser.add_argument(
#         "--chem_layers",
#         type=parse_list,
#         default=None,
#         help="Chem layers sizes",
#     )
#     parser.add_argument(
#         "--prot_layers", type=parse_list, default=None, help="Prot layers sizes"
#     )
#     parser.add_argument(
#         "--regressor_layers",
#         type=parse_list,
#         default=None,
#         help="Regressor layers sizes",
#     )
#     parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
#     parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
#     parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
#     parser.add_argument(
#         "--early_stop", type=int, default=None, help="Early stopping patience"
#     )
#     parser.add_argument("--loss", type=str, default=None, help="Loss function")
#     parser.add_argument(
#         "--loss_reduction", type=str, default=None, help="Loss reduction method"
#     )
#     parser.add_argument("--optimizer", type=str, default=None, help="Optimizer")
#     parser.add_argument("--lr", type=float, default=None, help="Learning rate")
#     parser.add_argument(
#         "--weight_decay", type=float, default=None, help="Weight decay rate"
#     )
#     parser.add_argument(
#         "--lr_scheduler", type=str, default=None, help="LR scheduler type"
#     )
#     parser.add_argument(
#         "--lr_scheduler_patience", type=int, default=None, help="LR scheduler patience"
#     )
#     parser.add_argument(
#         "--lr_scheduler_factor", type=float, default=None, help="LR scheduler factor"
#     )
#     parser.add_argument(
#         "--max_norm", type=float, default=None, help="Max norm for gradient clipping"
#     )
#     args = parser.parse_args()
#     # Construct kwargs, excluding arguments that were not provided
#     kwargs = {k: v for k, v in vars(args).items() if v is not None}
#
#     sweep_count = args.sweep_count
#     if sweep_count is not None and sweep_count > 0:
#         run_ensemble_hyperparm(
#             **kwargs,
#         )
#     else:
#         run_ensemble_wrapper(
#             **kwargs,
#         )
#
#


# def build_ensemble(config=wandb.config):
#     ensemble_models = []
#     try:
#         seed = config.seed
#     except AttributeError:
#         seed = 42
#     # deterministic cuda algorithms
#     torch.backends.cudnn.deterministic = True
#
#     for _ in range(config.ensemble_size):
#         set_seed(seed)
#         model = BaselineDNN(
#             config.input_dim,
#             config.hidden_dim_1,
#             config.hidden_dim_2,
#             config.hidden_dim_3,
#             config.output_dim,
#             config.dropout,
#         )
#         ensemble_models.append(model)
#         seed += 1
#
#     return ensemble_models
#
#
# def train_model(
#     model_idx,
#     ensemble_models,
#     train_loader,
#     val_loader,
#     test_loader,
#     config,
#     **kwargs,
#     # config=None,
# ):
#     """
#     Train a single model from the ensemble.
#
#     Parameters
#     ----------
#     model_idx : int
#         Index of the model in the ensemble.
#     ensemble_models : list
#         List of models in the ensemble.
#     train_loader : torch.utils.data.DataLoader
#         Training data loader.
#     val_loader : torch.utils.data.DataLoader
#         Validation data loader.
#     test_loader : torch.utils.data.DataLoader
#         Test data loader.
#     config : wandb.config
#         Configuration object containing hyperparameters and settings.
#     **kwargs : dict, optional
#         Additional keyword arguments.
#
#     Returns
#     -------
#     best_model : torch.nn.Module
#         Best trained model based on validation loss.
#     """
#     # config = wandb.config if config is None else config
#
#     # Get the model for the given model_idx
#     model = ensemble_models[model_idx]
#     # Move the model to the device
#     model.to(device)
#
#     # Define the loss function
#     loss_fn = MultiTaskLoss(
#         loss_type=config.loss,
#         reduction="none",
#     )
#     # Define the optimizer with weight decay and learning rate scheduler
#     optimizer = build_optimizer(
#         model, config.optimizer, config.learning_rate, config.weight_decay
#     )
#     lr_scheduler = ReduceLROnPlateau(
#         optimizer,
#         mode="min",
#         factor=config.lr_factor,
#         patience=config.lr_patience,
#         verbose=True,
#     )
#
#     # Train the model
#     best_val_loss = float("inf")
#     early_stop_counter = 0
#     best_model = None
#     for epoch in tqdm(range(config.num_epochs), desc="Epochs"):
#         try:
#             epoch, train_loss, val_loss, val_rmse, val_r2, val_evs = run_epoch(
#                 model,
#                 train_loader,
#                 val_loader,
#                 loss_fn,
#                 optimizer,
#                 lr_scheduler,
#                 epoch=epoch,
#             )
#             wandb.log(
#                 data={
#                     f"epoch": epoch,
#                     f"model{model_idx}/train_loss": train_loss,
#                     f"model{model_idx}/val_loss": val_loss,
#                     f"model{model_idx}/val_rmse": val_rmse,
#                     f"model{model_idx}/val_r2": val_r2,
#                     f"model{model_idx}/val_evs": val_evs,
#                 },
#                 # step=epoch
#             )
#             # Early stopping
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 early_stop_counter = 0
#                 # Save the best model - dropped to avoid memory issues
#                 # Update the best model and its performance
#                 best_model = model
#
#             else:
#                 early_stop_counter += 1
#                 if early_stop_counter > config.early_stop:
#                     break
#
#         except Exception as e:
#             raise Exception(
#                 f"The following exception occurred inside the epoch loop {e}"
#             )
#
#         # Save the best model
#         # save_model(config, best_model, model_idx)
#
#     predictions = predict(best_model, test_loader, return_targets=False)
#
#     return best_model, predictions

#
# def run_ensemble(
#     datasets=None,
#     config: Union[str, dict] = "uqdd/config/ensemble/ensemble.json",
#     activity: str = "xc50",
#     split: str = "random",
#     wandb_project_name: str = "multitask-learning-ensemble",
#     ensemble_size: int = 100,
#     seed: int = 42,
#     **kwargs,
# ):
#     # Load the config
#     config = get_model_config(
#         config=config,
#         activity=activity,
#         split=split,
#         ensemble_size=ensemble_size,
#         **kwargs,
#     )
#     # Load the dataset
#     if datasets is None:
#         datasets = get_datasets(activity=activity, split=split)
#     # Get tasks names:
#     tasks = get_tasks(activity=activity, split=split)
#
#     # Initialize wandb for the ensemble models
#     with wandb.init(
#         dir=LOG_DIR,
#         mode=wandb_mode,
#         project=wandb_project_name,
#         config=config,
#         name=f"{today}_ensemble_{activity}_{split}",
#     ):
#         config = wandb.config
#
#         # Initialize the table to store the metrics
#         uct_metrics_logger = MetricsTable(model_type="ensemble", config=config)
#
#         # Define the data loaders
#         train_loader, val_loader, test_loader = build_loader(
#             datasets, config.batch_size, config.input_dim
#         )
#         # Define the ensemble models
#         config.seed = seed  # TODO FIX THIS
#         ensemble_models = build_ensemble(config=config)
#         # Initialize lists to store the results
#         best_models = []
#         predictions = []
#         # results = []
#         _, targets = predict(
#             ensemble_models[0].to(device), test_loader, return_targets=True
#         )
#         # Train the ensemble models
#         for model_idx in tqdm(range(len(ensemble_models)), desc="Ensemble models"):
#             # Train the model
#             best_model, preds = train_model(
#                 model_idx,
#                 ensemble_models,
#                 train_loader,
#                 val_loader,
#                 test_loader,
#                 config,
#             )
#             # Store the results of the model
#             predictions.append(preds)
#             best_models.append(best_model)
#
#         # Save Best Ensemble Models
#         best_models = nn.ModuleList(best_models)
#         save_model(config, best_models, f"{wandb.run.name}_ensemble_model", onnx=False)
#
#         # Ensemble the predictions
#         ensemble_preds = torch.stack(predictions, dim=2)
#         # Process ensemble predictions
#         y_pred, y_std, y_true = process_preds(ensemble_preds, targets, None)
#
#         # Calculate and log the metrics
#         # task_name =
#         metrics = uct_metrics_logger(
#             y_pred=y_pred, y_std=y_std, y_true=y_true, task_name="All 20 Targets"
#         )
#         for task_idx in range(len(tasks)):
#             task_y_pred, task_y_std, task_y_true = process_preds(
#                 ensemble_preds, targets, task_idx=task_idx
#             )
#             # Calculate and log the metrics
#             task_name = tasks[task_idx]
#             metrics = uct_metrics_logger(
#                 y_pred=task_y_pred,
#                 y_std=task_y_std,
#                 y_true=task_y_true,
#                 task_name=task_name,
#             )
#
#         uct_metrics_logger.wandb_log()
#
# #
# # if __name__ == "__main__":
# #     # datasets = get_datasets('xc50', 'random')
# #     test_config = {
# #         "activity": "xc50",
# #         "batch_size": 64,
# #         "dropout": 0.1,
# #         "early_stop": 100,
# #         "hidden_dim_1": 2048,
# #         "hidden_dim_2": 256,
# #         "hidden_dim_3": 256,
# #         "input_dim": 2048,
# #         "learning_rate": 0.01,
# #         "loss": "huber",
# #         "lr_scheduler": "ReduceLROnPlateau",  # TODO not necessary anymore
# #         "lr_factor": 0.5,
# #         "lr_patience": 20,
# #         "num_epochs": 3,
# #         "num_tasks": 20,
# #         "optimizer": "SGD",
# #         "output_dim": 20,
# #         "weight_decay": 0.001,
# #         "seed": 42,
# #         "split": "random",
# #         "ensemble_size": 3,
# #     }
# #
# #     # test_loss, test_predictions = \
# #     run_ensemble(
# #         config=test_config,  # os.path.join(CONFIG_DIR, 'ensemble/ensemble.json'),
# #         activity="xc50",
# #         split="random",
# #         ensemble_size=5,
# #         # ensemble_method='fusion',
# #         wandb_project_name="mtl-ensemble-test",
# #         seed=42,
# #     )
# #
# # print(test_loss)
# # print(test_predictions)
#
#
# # chem_input_dim=None,
# # prot_input_dim=None,
# # task_type="regression",
# # n_targets=-1,
# # seed = 42
# # set_seed(seed)
# #
# # logger = create_logger(name="ensemble", file_level="debug", stream_level="info")
# #
# # start_time = datetime.now()
# # logger.info(f"Ensemble - start time: {start_time}")
# #
# # # get datasets
# # datasets = build_datasets(
# #     data_name=data_name,
# #     n_targets=n_targets,
# #     activity_type=activity_type,
# #     split_type=split_type,
# #     desc_prot=descriptor_protein,
# #     desc_chem=descriptor_chemical,
# #     label_scaling_func=label_scaling_func,
# #     ext=ext,
# #     logger=logger,
# # )
# # desc_prot_len, desc_chem_len = get_desc_len_from_dataset(datasets["train"])
# # logger.info(f"Chemical descriptor {descriptor_chemical} of length: {desc_chem_len}")
# # logger.info(f"Protein descriptor {descriptor_protein} of length: {desc_prot_len}")


# def _run_ensemble(
#     config=None,
# ):
#     if config is not None:
#         wandb_project_name = config.get(
#             "wandb_project_name"
#         )  # add it to config only in baseline not the sweep
#         run = wandb.init(
#             config=config, dir=WANDB_DIR, mode=WANDB_MODE, project=wandb_project_name
#         )
#     else:
#         run = wandb.init(config=config, dir=WANDB_DIR, mode=WANDB_MODE)
#     config = wandb.config
#
#     data_name = config.get("data_name")
#     activity_type = config.get("activity_type")
#     n_targets = config.get("n_targets")
#     descriptor_protein = config.get("descriptor_protein")
#     descriptor_chemical = config.get("descriptor_chemical")
#     median_scaling = config.get("median_scaling")
#     split_type = config.get("split_type")
#     ext = config.get("ext")
#     task_type = config.get("task_type")
#
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
#         "ensemble",
#         data_name,
#         activity_type,
#         n_targets,
#         descriptor_protein,
#         descriptor_chemical,
#         split_type,
#         median_scaling,
#         task_type,
#         ext,
#         LOGGER,
#     )
#
#     m_tag = "median_scaling" if median_scaling else "no_median_scaling"
#     mt = n_targets > 1
#     mt_tag = "MT" if mt else "ST"
#     wandb_tags = [
#         "ensemble",
#         data_name,
#         activity_type,
#         descriptor_protein,
#         descriptor_chemical,
#         split_type,
#         task_type,
#         m_tag,
#         mt_tag,
#     ]
#     with wandb.init(
#         dir=WANDB_DIR,
#         mode=WANDB_MODE,
#         project=wandb_project_name,
#         config=config,
#         tags=wandb_tags,
#     ):
#         config = wandb.config
#
#         # Define the ensemble models
#         ensemble_model = EnsembleDNN(
#             config=config,
#             model_class=BaselineDNN,
#             ensemble_size=ensemble_size,
#             chem_input_dim=desc_chem_len,
#             prot_input_dim=desc_prot_len,
#             task_type=task_type,
#             n_targets=n_targets,
#             logger=logger,
#         ).to(DEVICE)
#
#         # Train the ensemble model
#         best_model, test_loss = run_model(
#             config,
#             ensemble_model,
#             dataloaders,
#             n_targets=n_targets,
#             device=DEVICE,
#             logger=logger,
#         )
#         model_name = (
#             f"{TODAY}-ensemble_{split_type}_{descriptor_protein}_{descriptor_chemical}"
#         )
#         # Save the best model
#         save_model(
#             config,
#             best_model,
#             model_name,
#             data_specific_path,
#             desc_prot_len,
#             desc_chem_len,
#             onnx=True,
#         )
#         # Predictions on Test set
#         # Initialize the table to store the metrics
#         config.activity = activity_type
#         config.split = split_type
#         uct_metrics_logger = MetricsTable(
#             model_type="ensemble",
#             config=config,
#             desc_prot=descriptor_protein,
#             desc_chem=descriptor_chemical,
#             multitask=mt,
#             task_type=task_type,
#             data_specific_path=data_specific_path,
#             model_name=model_name,
#             logger=logger,
#         )
#         ensemble_preds, targets = predict(
#             best_model, dataloaders["test"], return_targets=True
#         )
#
#         if mt:
#             tasks = get_tasks(
#                 data_name=data_name, activity=activity_type, n_targets=n_targets
#             )
#
#             y_true, y_pred, y_std, y_err = process_preds(ensemble_preds, targets, None)
#             df = create_df_preds(
#                 y_true,
#                 y_pred,
#                 y_std,
#                 y_err,
#                 True,
#                 data_specific_path,
#                 model_name + "_MT_AllTargets",
#             )
#             logger.debug(
#                 f"Ensemble - predictions saved to Dataframe with shape {df.shape}"
#             )
#             metrics, plots = uct_metrics_logger(
#                 y_pred=y_pred,
#                 y_std=y_std,
#                 y_true=y_true,
#                 y_err=y_err,
#                 task_name=f"All {n_targets} Targets",
#             )
#
#             for task_idx in range(len(tasks)):
#                 task_y_true, task_y_pred, task_y_std, task_y_err = process_preds(
#                     ensemble_preds, targets, task_idx=task_idx
#                 )
#                 # Calculate and log the metrics
#                 task_name = tasks[task_idx]
#                 taskmetrics, taskplots = uct_metrics_logger(
#                     y_pred=task_y_pred,
#                     y_std=task_y_std,
#                     y_true=task_y_true,
#                     y_err=task_y_err,
#                     task_name=task_name,
#                 )
#                 metrics[taskmetrics] = taskmetrics
#                 plots[taskplots] = taskplots
#
#         else:  # ST
#             task_name = f"PCM {task_type}"
#             # Process the predictions
#             y_true, y_pred, y_std, y_err = process_preds(ensemble_preds, targets, None)
#             df = create_df_preds(
#                 y_true, y_pred, y_std, y_err, True, data_specific_path, model_name
#             )
#             logger.debug(
#                 f"Ensemble - predictions saved to Dataframe with shape {df.shape}"
#             )
#
#             # Calculate and log the metrics
#             metrics, plots = uct_metrics_logger(
#                 y_pred=y_pred,
#                 y_std=y_std,
#                 y_true=y_true,
#                 y_err=y_err,
#                 task_name=task_name,
#             )
#
#         uct_metrics_logger.wandb_log()
#
#     logger.info(f"Ensemble - end time: {datetime.now()}")
#     logger.info(f"Ensemble - duration: {datetime.now() - start_time}")
#     return test_loss, ensemble_preds, metrics, plots
#
