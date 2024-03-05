import argparse
import sys
from datetime import datetime
from functools import partial
import wandb
import torch
import torch.nn as nn
from uqdd import TODAY, DEVICE, LOGS_DIR, WANDB_MODE
from uqdd.utils import create_logger
from uqdd.models.utils_models import (
    build_datasets,
    get_desc_len,
    get_model_config,
    get_sweep_config,
    set_seed,
)
from uqdd.models.utils_train import run_model


class BaselineDNN(nn.Module):
    def __init__(
        self,
        config=None,
        chem_input_dim=None,
        prot_input_dim=None,
        task_type="regression",
        n_targets=-1,
        **kwargs,
    ):
        super(BaselineDNN, self).__init__()
        assert task_type in [
            "regression",
            "classification",
        ], "task_type must be either 'regression' or 'classification'"
        self.task_type = task_type
        self.MT = n_targets > 0
        self.feature_extractors = {}
        self.regressor_or_classifier = None

        self.log = create_logger(
            name="baseline", file_level="debug", stream_level="info"
        )
        if config is None:
            config = get_model_config(model_name="baseline", **kwargs)

        n_targets = 1 if not self.MT else n_targets
        # active inactive per each target if classification
        output_dim = n_targets if task_type == "regression" else 2 * n_targets

        # Initialize feature extractors
        self.init_layers(config, chem_input_dim, prot_input_dim, output_dim)

        self.apply(self.init_wt)

    @staticmethod
    def init_wt(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, chem_input, prot_input):
        chem_features = self.feature_extractors["chem"](chem_input)
        if not self.MT:
            prot_features = self.feature_extractors["prot"](prot_input)
            combined_features = torch.cat((chem_features, prot_features), dim=1)
        else:
            combined_features = chem_features

        output = self.regressor_or_classifier(combined_features)
        return output

    @staticmethod
    def create_feature_extractor(input_dim, layer_dims, dropout):
        modules = [nn.Linear(input_dim, layer_dims[0]), nn.ReLU()]
        for i in range(len(layer_dims) - 1):
            modules += [nn.Linear(layer_dims[i], layer_dims[i + 1]), nn.ReLU()]
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
        return nn.Sequential(*modules)

    def init_layers(self, config, chem_input_dim, prot_input_dim, output_dim):
        # Chemical feature extractor
        chem_layers = config["chem_layers"]
        self.feature_extractors["chem"] = self.create_feature_extractor(
            chem_input_dim, chem_layers, config["dropout"]
        )
        self.log.debug(f"Chemical feature extractor: {chem_input_dim} -> {chem_layers}")

        if not self.MT:
            # Protein feature extractor (only for single-task learning)
            prot_layers = config["prot_layers"]
            self.feature_extractors["prot"] = self.create_feature_extractor(
                prot_input_dim, prot_layers, config["dropout"]
            )
            self.log.debug(
                f"Protein feature extractor: {prot_input_dim} -> {prot_layers}"
            )

            # Combined input dimension for STL
            chem_dim = config["chem_layers"][-1]
            prot_dim = config["prot_layers"][-1]
            combined_input_dim = chem_dim + prot_dim

        else:
            # Only chemical features for MTL
            combined_input_dim = config["chem_layers"][-1]

        self.log.debug(f"Combined input dimension: {combined_input_dim}")
        self.log.debug(f"Output dimension: {output_dim}")

        self.regressor_or_classifier = nn.Linear(combined_input_dim, output_dim)


def run_baseline(
    datasets,
    config=None,
    chem_input_dim=None,
    prot_input_dim=None,
    task_type="regression",
    n_targets=-1,
    wandb_project_name=f"{TODAY}-baseline",
    seed=42,
    **kwargs,
):
    set_seed(seed)
    config = get_model_config("baseline", **kwargs) if not config else config
    with wandb.init(
        dir=LOGS_DIR, mode=WANDB_MODE, project=wandb_project_name, config=config
    ):
        config = wandb.config

        # Initiate the model
        model = BaselineDNN(
            config=config,
            chem_input_dim=chem_input_dim,
            prot_input_dim=prot_input_dim,
            task_type=task_type,
            n_targets=n_targets,
        ).to(DEVICE)

        best_model, test_loss = run_model(
            config,
            datasets,
            model,
            model_name="baseline",
            n_targets=n_targets,
            seed=42,
            device=DEVICE,
        )

        return best_model, test_loss


def run_baseline_sweeper(
    datasets,
    chem_input_dim=None,
    prot_input_dim=None,
    task_type="regression",
    n_targets=-1,
    wandb_project_name=f"{TODAY}-baseline",
    sweep_count=1,
    seed=42,
    **kwargs,
):
    sweep_config = get_sweep_config("baseline", **kwargs)
    sweep_id = wandb.sweep(
        sweep_config,
        project=wandb_project_name,
    )
    # Recursion to run the sweep
    wandb_train_func = partial(
        run_baseline,
        datasets=datasets,
        config=sweep_config,
        chem_input_dim=chem_input_dim,
        prot_input_dim=prot_input_dim,
        task_type=task_type,
        n_targets=n_targets,
        wandb_project_name=wandb_project_name,
        seed=seed,
    )
    wandb.agent(sweep_id, function=wandb_train_func, count=sweep_count)


def main():
    parser = argparse.ArgumentParser(description="Baseline Model Running")
    parser.add_argument(
        "--data-name",
        type=str,
        default="papyrus",
        choices=["papyrus", "tdc", "other"],
        help="Data name argument",
    )
    parser.add_argument(
        "--n-targets",
        type=int,
        default=-1,
        help="Number of targets argument (default=-1 for all targets)",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type argument",
    )
    parser.add_argument(
        "--activity",
        type=str,
        default="xc50",
        choices=["xc50", "kx"],
        help="Activity argument",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="random",
        choices=["random", "scaffold", "time"],
        help="Split argument",
    )
    parser.add_argument(
        "--desc-prot",
        type=str,
        default="unirep",
        choices=[
            "ankh-base",
            "ankh-large",
            "unirep",
            "protbert",
            "protbert_bfd",
            "esm1_t34",
            "esm1_t12",
            "esm1_t6",
            "esm1b",
            "esm_msa1",
            "esm_msa1b",
            "esm1v",
        ],
        help="Protein descriptor argument",
    )
    parser.add_argument(
        "--desc-chem",
        type=str,
        default="ecfp2048",
        choices=[
            "ecfp1024",
            "ecfp2048",
            "mold2",
            "mordred",
            "cddd",
            "fingerprint",  # "moldesc"
        ],
        help="Chemical descriptor argument",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--baseline", action="store_true", help="Run baseline")
    group.add_argument(
        "--hyperparam", action="store_true", help="Run hyperparameter search"
    )
    parser.add_argument(
        "--sweep-count",
        type=int,
        default=250,
        help="Sweep count argument",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="baseline-test",
        help="Wandb project name argument",
    )

    args = parser.parse_args()
    start_time = datetime.now()

    datasets = build_datasets(
        data_name=args.data_name,
        n_targets=args.n_targets,
        activity_type=args.activity,
        split_type=args.split,
        desc_prot=args.desc_prot,
        desc_chem=args.desc_chem,
        ext=args.ext,
    )
    desc_prot_len, desc_chem_len = get_desc_len(datasets)  # TODO fix this for MT
    if args.baseline:
        _, test_loss = run_baseline(
            datasets,
            chem_input_dim=desc_chem_len,
            prot_input_dim=desc_prot_len,
            task_type=args.task_type,
            n_targets=args.n_targets,
            wandb_project_name=f"{TODAY}-{args.wandb_project_name}",
            seed=42,
        )
        print(f"Test loss: {test_loss}")

    elif args.hyperparam:
        run_baseline_sweeper(
            datasets,
            chem_input_dim=desc_chem_len,
            prot_input_dim=desc_prot_len,
            task_type=args.task_type,
            n_targets=args.n_targets,
            wandb_project_name=f"{TODAY}-{args.wandb_project_name}",
            sweep_count=args.sweep_count,
        )

    else:
        sys.exit("Error: Please specify either --baseline or --hyperparam")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Script execution time: {duration}")


if __name__ == "__main__":
    # main()
    data_name = "papyrus"
    n_targets = -1
    task_type = "regression"
    activity = "xc50"
    split = "random"
    desc_prot = "ankh-base"
    desc_chem = "ecfp2048"
    ext = "pkl"
    wandb_project_name = f"{TODAY}-baseline-test"
    sweep_count = 250

    datasets = build_datasets(
        data_name=data_name,
        n_targets=n_targets,
        activity_type=activity,
        split_type=split,
        desc_prot=desc_prot,
        desc_chem=desc_chem,
        ext=ext,
    )
    desc_prot_len, desc_chem_len = get_desc_len(datasets["train"])
    run_baseline(
        datasets,
        chem_input_dim=desc_chem_len,
        prot_input_dim=desc_prot_len,
        task_type=task_type,
        n_targets=n_targets,
        wandb_project_name=wandb_project_name,
        seed=42,
    )

    run_baseline_sweeper(
        datasets,
        chem_input_dim=desc_chem_len,
        prot_input_dim=desc_prot_len,
        task_type=task_type,
        n_targets=n_targets,
        wandb_project_name=wandb_project_name + "_sweeper",
        sweep_count=sweep_count,
    )
    print("Done")


#
# def _run_baseline(
#     datasets=None,
#     config=None,
#     activity="xc50",
#     split="random",
#     wandb_project_name=f"{TODAY}-baseline",
#     seed=42,
#     **kwargs,
# ):
#     set_seed(seed)
#     # Load config
#     config = get_model_config(config=config, activity=activity, split=split, **kwargs)
#
#     if datasets is None:
#         # TODO fix this
#         datasets = get_datasets(activity=activity, split=split)
#
#     with wandb.init(
#         dir=LOGS_DIR, mode=WANDB_MODE, project=wandb_project_name, config=config
#     ):
#         config = wandb.config
#         # Load the dataset
#         train_loader, val_loader, test_loader = build_loader(
#             datasets, config.batch_size, config.input_dim
#         )
#
#         # Train the model
#         best_model, loss_fn = train_model(
#             train_loader, val_loader, config=config, seed=seed
#         )
#
#         # Testing metrics on the best model
#         test_loss, test_rmse, test_r2, test_evs = evaluate(
#             best_model, test_loader, loss_fn
#         )
#
#
# def _run_baseline_hyperparam(
#     config=None,
#     activity="xc50",
#     split="random",
#     wandb_project_name=f"{TODAY}-baseline-hyperparam",
#     sweep_count=1,
#     seed=42,
#     **kwargs,
# ):
#     set_seed(seed)
#     datasets = get_datasets(activity=activity, split=split)
#     sweep_config = get_sweep_config(
#         config=config, activity=activity, split=split, **kwargs
#     )
#
#     sweep_id = wandb.sweep(
#         sweep_config,
#         project=wandb_project_name,
#     )
#
#     wandb_train_func = partial(
#         run_baseline,
#         datasets=datasets,
#         activity=activity,
#         split=split,
#         config=sweep_config,
#         wandb_project_name=wandb_project_name,
#         seed=42,
#     )
#     wandb.agent(sweep_id, function=wandb_train_func, count=sweep_count)
#
#
# if __name__ == "__main__":
#     test_loss = run_baseline()

# if data_name == "papyrus":
#     from uqdd.data.data_papyrus import get_datasets
# elif data_name == "tdc":
#     from uqdd.data.data_tdc import get_datasets
# elif data_name == "other":
#     raise NotImplementedError
# else:
#     raise ValueError("Invalid data_name")

# class MTBaselineDNN(nn.Module):
#     def __init__(self, config, **kwargs):
#         super(MTBaselineDNN, self).__init__()
#         input_dim = config["input_dim"]
#         layers = config["model_config"]["layers"]
#         dropout = config["model_config"]["dropout"]
#         num_tasks = config["num_tasks"]
#
#         modules = [nn.Linear(input_dim, layers[0]), nn.ReLU()]
#         for i in range(len(layers) - 1):
#             modules.append(nn.Linear(layers[i], layers[i + 1]))
#             modules.append(nn.ReLU())
#             if dropout > 0:
#                 modules.append(nn.Dropout(dropout))
#
#         self.feature_extractor = nn.Sequential(*modules)
#         self.task_specific = nn.Linear(layers[-1], num_tasks)
#         self.apply(self.init_wt)
#
#     @staticmethod
#     def init_wt(module):
#         if isinstance(module, nn.Linear):
#             nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("relu"))
#
#     def forward(self, x):
#         features = self.feature_extractor(x)
#         outputs = self.task_specific(features)
#         return outputs


# class BaselineDNN(nn.Module):
#     def __init__(
#         self,
#         chem_input_dim=None,
#         prot_input_dim=None,
#         task_type="regression",
#         n_targets=-1,
#         **kwargs,
#     ):
#         super(BaselineDNN, self).__init__()
#         assert task_type in [
#             "regression",
#             "classification",
#         ], "task_type must be either 'regression' or 'classification'"
#         self.task_type = task_type
#         self.MT = n_targets > 0
#         self.feature_extractors = {}
#
#         self.log = create_logger(
#             name="baseline", file_level="debug", stream_level="info"
#         )
#         config = get_config("baseline", **kwargs)
#
#         # Initialize feature extractors
#         self.init_feature_extractors(config, chem_input_dim, prot_input_dim)
#
#         # Initialize regressor or classifier
#         self.init_regressor_or_classifier(config, n_targets)
#
#         # Unpack configuration for chemical and protein branches
#         chem_input_dim = (
#             config["chem_input_dim"] if not chem_input_dim else chem_input_dim
#         )
#         chem_layers = config["chem_model_config"]["layers"]
#
#         prot_input_dim = (
#             config["prot_input_dim"] if not prot_input_dim else prot_input_dim
#         )
#         prot_layers = config["prot_model_config"]["layers"]
#         dropout = config["model_config"]["dropout"]
#         regressor_layers = config["regressor_config"]["layers"]
#
#         # Chemical compound feature extractor
#         chem_modules = [nn.Linear(chem_input_dim, chem_layers[0]), nn.ReLU()]
#         for i in range(len(chem_layers) - 1):
#             chem_modules.append(nn.Linear(chem_layers[i], chem_layers[i + 1]))
#             chem_modules.append(nn.ReLU())
#             if dropout > 0:
#                 chem_modules.append(nn.Dropout(dropout))
#         self.chem_feature_extractor = nn.Sequential(*chem_modules)
#         self.log.debug(f"Chemical feature extractor: {chem_input_dim} -> {chem_layers}")
#
#         if not self.MT:
#             # Protein feature extractor
#             prot_modules = [nn.Linear(prot_input_dim, prot_layers[0]), nn.ReLU()]
#             for i in range(len(prot_layers) - 1):
#                 prot_modules.append(nn.Linear(prot_layers[i], prot_layers[i + 1]))
#                 prot_modules.append(nn.ReLU())
#                 if dropout > 0:
#                     prot_modules.append(nn.Dropout(dropout))
#             self.prot_feature_extractor = nn.Sequential(*prot_modules)
#
#             # Regressor construction
#             combined_input_dim = chem_layers[-1] + prot_layers[-1]
#             output_dim = 1 if task_type == "regression" else 2
#             regressor_modules = [
#                 nn.Linear(combined_input_dim, regressor_layers[0]),
#                 nn.ReLU(),
#             ]
#             for i in range(len(regressor_layers) - 1):
#                 regressor_modules.append(
#                     nn.Linear(regressor_layers[i], regressor_layers[i + 1])
#                 )
#                 regressor_modules.append(nn.ReLU())
#                 if (
#                     dropout > 0 and i < len(regressor_layers) - 2
#                 ):  # No dropout before final layer
#                     regressor_modules.append(nn.Dropout(dropout))
#
#             # Final layer
#             regressor_modules.append(nn.Linear(regressor_layers[-1], output_dim))
#
#             self.regressor = nn.Sequential(*regressor_modules)
#
#         else:
#             self.regressor = nn.Linear(chem_layers[-1], n_targets)
#
#         self.apply(self.init_wt)
#
#     @staticmethod
#     def init_wt(module):
#         if isinstance(module, nn.Linear):
#             nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("relu"))
#
#     def forward(self, chem_input, prot_input):
#         chem_features = self.chem_feature_extractor(chem_input)
#         if not self.MT:
#             prot_features = self.prot_feature_extractor(prot_input)
#             combined_features = torch.cat((chem_features, prot_features), dim=1)
#         else:
#             combined_features = chem_features
#         output = self.regressor(combined_features)
#         return output
#
#     @staticmethod
#     def create_feature_extractor(input_dim, layer_dims, dropout):
#         modules = [nn.Linear(input_dim, layer_dims[0]), nn.ReLU()]
#         for i in range(len(layer_dims) - 1):
#             modules += [nn.Linear(layer_dims[i], layer_dims[i + 1]), nn.ReLU()]
#             if dropout > 0:
#                 modules.append(nn.Dropout(dropout))
#         return nn.Sequential(*modules)
#
#     def init_feature_extractors(self, config, chem_input_dim, prot_input_dim):
#         # Chemical feature extractor
#         chem_layers = config["chem_model_config"]["layers"]
#         self.feature_extractors["chem"] = self.create_feature_extractor(
#             chem_input_dim, chem_layers, config["model_config"]["dropout"]
#         )
#
#         if not self.MT:
#             # Protein feature extractor (only for single-task learning)
#             prot_layers = config["prot_model_config"]["layers"]
#             self.feature_extractors["prot"] = self.create_feature_extractor(
#                 prot_input_dim, prot_layers, config["model_config"]["dropout"]
#             )
#
#     def init_regressor_or_classifier(self, config, n_targets):
#
#         output_dim = n_targets if self.MT else 1 if self.task_type == "regression" else 2
#         if not self.MT:
#             # Combined input dimension for STL
#             chem_dim = config["chem_model_config"]["layers"][-1]
#             prot_dim = config["prot_model_config"]["layers"][-1]
#             combined_input_dim = chem_dim + prot_dim
#         else:
#             # Only chemical features for MTL
#             combined_input_dim = config["chem_model_config"]["layers"][-1]
#
#         self.regressor_or_classifier = nn.Linear(combined_input_dim, output_dim)
#
#     # if data_name == "papyrus":
#     #     from uqdd.data.data_papyrus import get_datasets
#     # elif data_name == "tdc":
#     #     from uqdd.data.data_tdc import get_datasets
#     # elif data_name == "other":
#     #     raise NotImplementedError
#     # else:
#     #     raise ValueError("Invalid data_name")
