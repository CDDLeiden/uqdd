import argparse

import wandb
import torch
import torch.nn as nn
from uqdd.utils import create_logger, parse_list
from uqdd.models.utils_models import (
    get_model_config,
    get_sweep_config,
)
from uqdd.models.utils_train import (
    train_model_e2e,
)


class BaselineDNN(nn.Module):
    def __init__(
        self,
        config=None,
        logger=None,
        **kwargs,
    ):
        super(BaselineDNN, self).__init__()

        chem_input_dim = config.get("chem_input_dim", None)
        prot_input_dim = config.get("prot_input_dim", None)
        task_type = config.get("task_type", "regression")
        n_targets = config.get("n_targets", -1)
        self.MT = config.get("MT", n_targets > 1)
        self.aleatoric = config.get("aleatoric", False)

        assert task_type in [
            "regression",
            "classification",
        ], "task_type must be either 'regression' or 'classification'"

        self.task_type = task_type
        # memory placeholders
        self.prot_feature_extractor = None
        self.chem_feature_extractor = None
        self.regressor_or_classifier = None
        self.logvar_layer = None
        self.output_layer = None
        self.logger = (
            create_logger(name="baseline", file_level="debug", stream_level="info")
            if not logger
            else logger
        )

        if config is None:
            config = get_model_config(model_name="baseline", **kwargs)
        self.config = config
        n_targets = 1 if not self.MT else n_targets
        # active inactive per each target if classification
        self.output_dim = n_targets if task_type == "regression" else 2 * n_targets

        # Initialize feature extractors
        self.init_layers(config, chem_input_dim, prot_input_dim, self.output_dim)

        self.apply(self.init_wt)

    @staticmethod
    def init_wt(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, inputs):
        prot_input, chem_input = inputs
        chem_features = self.chem_feature_extractor(chem_input)
        if not self.MT:
            # prot_input, chem_input = inputs
            # chem_features = self.chem_feature_extractor(chem_input)
            prot_features = self.prot_feature_extractor(prot_input)
            combined_features = torch.cat((chem_features, prot_features), dim=1)
        else:
            # chem_input = inputs
            # combined_features = self.chem_feature_extractor(chem_input)
            combined_features = chem_features

        _output = self.regressor_or_classifier(combined_features)
        if self.aleatoric:
            output = self.output_layer(_output)
            logvar = self.logvar_layer(_output)
            return output, logvar
        else:
            output = self.output_layer(_output)
        # TODO : check this with classification if necessary:
        self.output_layer(_output)
        return output

    @staticmethod
    def create_mlp(input_dim, layer_dims, dropout): # , output_dim=None
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
        # if output_dim:
        #     modules.append(nn.Linear(layer_dims[-1], output_dim))
        return nn.Sequential(*modules)  # , layer_dims[-1]

    def init_layers(self, config, chem_input_dim, prot_input_dim, output_dim):
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
        if self.aleatoric:
            self.output_layer = nn.Linear(regressor_layers[-1], output_dim)
            self.logvar_layer = nn.Sequential(
                nn.Linear(regressor_layers[-1], output_dim),
                nn.Softplus()
            )

        else:
            self.output_layer = nn.Linear(regressor_layers[-1], output_dim)
        self.logger.debug(f"Output dimension: {output_dim}")


def run_baseline(config=None):
    best_model, _, _, _ = train_model_e2e(
        config, model=BaselineDNN, model_type="baseline", logger=LOGGER
    )

    return best_model


def run_baseline_wrapper(
    **kwargs,
):
    global LOGGER
    LOGGER = create_logger("baseline", file_level="debug", stream_level="info")

    config = get_model_config(
        "baseline",
        **kwargs,
    )
    run_baseline(config=config)


# args mentioned for readibility


# data_name = (data_name,)
# activity_type = (activity_type,)
# n_targets = (n_targets,)
# descriptor_protein = (descriptor_protein,)
# descriptor_chemical = (descriptor_chemical,)
# median_scaling = (median_scaling,)
# split_type = (split_type,)
# ext = (ext,)
# task_type = (task_type,)
# wandb_project_name = (wandb_project_name,)
def run_baseline_hyperparam(**kwargs):
    global LOGGER
    LOGGER = create_logger(
        name="baseline-sweep", file_level="debug", stream_level="info"
    )

    sweep_count = kwargs.pop("sweep_count")
    wandb_project_name = kwargs.pop("wandb_project_name")
    config = get_sweep_config("baseline", **kwargs)
    config["project"] = wandb_project_name
    sweep_id = wandb.sweep(
        config,
        project=wandb_project_name,
    )
    print(f"Running sweep with SWEEP_ID: {sweep_id}")
    wandb.agent(sweep_id, function=run_baseline, count=sweep_count)


#
# def _run_baseline_sweeper(
#     sweep_config=None,
#     sweep_count=1,
#     data_name="papyrus",
#     activity_type="xc50",
#     n_targets=-1,
#     descriptor_protein=None,
#     descriptor_chemical=None,
#     median_scaling=False,
#     split_type="random",
#     ext="pkl",
#     task_type="regression",
#     wandb_project_name=f"{TODAY}-baseline-sweep",
#     **kwargs,
# ):
#     logger = create_logger(
#         name="baseline-sweep", file_level="debug", stream_level="info"
#     )
#     sweep_config = (
#         get_sweep_config("baseline", **kwargs) if not sweep_config else sweep_config
#     )
#     wandb.init(dir=WANDB_DIR, mode=WANDB_MODE)
#     sweep_id = wandb.sweep(
#         sweep_config,
#         project=wandb_project_name,
#     )
#     wandb_train_func = partial(
#         run_baseline,
#         # config=sweep_config,
#         data_name=data_name,
#         activity_type=activity_type,
#         n_targets=n_targets,
#         descriptor_protein=descriptor_protein,
#         descriptor_chemical=descriptor_chemical,
#         median_scaling=median_scaling,
#         # label_scaling_func=label_scaling_func,
#         split_type=split_type,
#         ext=ext,
#         task_type=task_type,
#         wandb_project_name=wandb_project_name,
#         logger=logger,
#     )
#
#     wandb.agent(sweep_id, function=wandb_train_func, count=sweep_count)
#     # wandb.agent(sweep_id, function=run_baseline, count=sweep_count,  )
#
#     wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Baseline Model Running")
    parser.add_argument(
        "--data_name",
        type=str,
        default="papyrus",
        choices=["papyrus", "tdc", "other"],
        help="Data name argument",
    )
    parser.add_argument(
        "--activity_type",
        type=str,
        default="xc50",
        choices=["xc50", "kx"],
        help="Activity argument",
    )
    parser.add_argument(
        "--n_targets",
        type=int,
        default=-1,
        help="Number of targets argument (default=-1 for all targets)",
    )
    parser.add_argument(
        "--descriptor_protein",
        type=str,
        default=None,
        choices=[
            None,
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
        "--descriptor_chemical",
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
    parser.add_argument(
        "--median_scaling",
        type=bool,
        default=False,
        help="Label Median scaling function argument",
    )
    # parser.add_argument(
    #     "--label-scaling-func",
    #     type=str,
    #     default=None,
    #     choices=[None, "median", "standard", "minmax", "robust"],
    #     help="Label scaling function argument",
    # )
    parser.add_argument(
        "--split_type",
        type=str,
        default="random",
        choices=["random", "scaffold", "time"],
        help="Split argument",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="pkl",
        choices=["pkl", "parquet", "csv", "feather"],
        help="File extension argument",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type argument",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="baseline-test",
        help="Wandb project name argument",
    )

    # group = parser.add_mutually_exclusive_group()
    # group.add_argument("--baseline", action="store_true", help="Run baseline")
    # group.add_argument(
    #     "--hyperparam", action="store_true", help="Run hyperparameter search"
    # )
    parser.add_argument(
        "--sweep-count",
        type=int,
        default=None,
        help="Sweep count argument",
    )
    # take chem layers as list input
    parser.add_argument(
        "--chem_layers",
        type=parse_list,
        default=None,
        help="Chem layers sizes",
    )  #  nargs="+",
    parser.add_argument(
        "--prot_layers", type=parse_list, default=None, help="Prot layers sizes"
    )
    parser.add_argument(
        "--regressor_layers",
        # nargs="+",
        type=parse_list,
        default=None,
        help="Regressor layers sizes",
    )
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument(
        "--early_stop", type=int, default=None, help="Early stopping patience"
    )
    parser.add_argument("--loss", type=str, default=None, help="Loss function")
    parser.add_argument(
        "--loss_reduction", type=str, default=None, help="Loss reduction method"
    )
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=None, help="Weight decay rate"
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default=None, help="LR scheduler type"
    )
    parser.add_argument(
        "--lr_scheduler_patience", type=int, default=None, help="LR scheduler patience"
    )
    parser.add_argument(
        "--lr_scheduler_factor", type=float, default=None, help="LR scheduler factor"
    )

    args = parser.parse_args()
    # Construct kwargs, excluding arguments that were not provided
    kwargs = {k: v for k, v in vars(args).items() if v is not None}

    sweep_count = args.sweep_count
    if sweep_count is not None and sweep_count > 0:
        run_baseline_hyperparam(
            **kwargs,
        )

    else:
        run_baseline_wrapper(
            **kwargs,
        )


if __name__ == "__main__":
    main()
    #
    # data_name = "papyrus"
    # n_targets = -1
    # task_type = "regression"
    # activity = "xc50"
    # split = "random"
    # desc_prot = "ankh-base"
    # desc_chem = "ecfp2048"
    # median_scaling = False
    # ext = "pkl"
    # wandb_project_name = "baseline-test-272"
    # sweep_count = 0  # 250
    # aleatoric = True
    # # epochs=1
    #
    # run_baseline_wrapper(
    #     data_name=data_name,
    #     activity_type=activity,
    #     n_targets=n_targets,
    #     descriptor_protein=desc_prot,
    #     descriptor_chemical=desc_chem,
    #     median_scaling=median_scaling,
    #     split_type=split,
    #     aleatoric=aleatoric,
    #     ext=ext,
    #     task_type=task_type,
    #     wandb_project_name=wandb_project_name,
    #     logger=None,
    # )
    #
    # sweep_count = 10
    # run_baseline_hyperparam(
    #     sweep_count=sweep_count,
    #     data_name=data_name,
    #     activity_type=activity,
    #     n_targets=n_targets,
    #     descriptor_protein=desc_prot,
    #     descriptor_chemical=desc_chem,
    #     split_type=split,
    #     ext=ext,
    #     task_type=task_type,
    #     wandb_project_name=wandb_project_name,
    # )
    # print("Done")


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
# seed = 42
# set_seed(seed)
# config = get_model_config("baseline", **kwargs) if not config else config
# logger = (
#     create_logger(name="baseline", file_level="debug", stream_level="info")
#     if not logger
#     else logger
# )
#
# start_time = datetime.now()
# logger.info(f"Baseline - start time: {start_time}")
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
# # build dataloaders
#
# # get descriptor lengths
# desc_prot_len, desc_chem_len = get_desc_len(descriptor_protein, descriptor_chemical)
# # desc_prot_len, desc_chem_len = get_desc_len_from_dataset(datasets["train"])
# logger.info(f"Chemical descriptor {descriptor_chemical} of length: {desc_chem_len}")
# logger.info(f"Protein descriptor {descriptor_protein} of length: {desc_prot_len}")
#
# def run_baseline(
#     config=None,
#     data_kwargs=None,
#     wandb_project_name=f"{TODAY}-baseline",
#     logger=None,
#     **kwargs,
# ):
#     config = get_model_config("baseline", **kwargs) if not config else config
#     desc_chem_len, desc_prot_len = get_desc_len()
#     # Initiate the model
#     model = BaselineDNN(
#         config=config,
#         chem_input_dim=desc_chem_len,
#         prot_input_dim=desc_prot_len,
#         task_type=task_type,
#         n_targets=n_targets,
#     ).to(DEVICE)
#     baseline_model =
#     best_model, test_loss = run_model_e2e(
#         model,
#         "baseline",
#         config,
#         data_kwargs,
#         wandb_project_name,
#         logger,
#     )
#
