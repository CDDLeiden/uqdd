from functools import partial
import wandb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from uqdd import TODAY, DEVICE, LOGS_DIR, WANDB_MODE
from uqdd.models.utils_models import (
    get_model_config,
    get_sweep_config,
    build_loader,
    build_optimizer,
    save_models,
    calc_regr_metrics,
    set_seed,
    MultiTaskLoss,
)

# get_datasets,


class MTBaselineDNN(nn.Module):
    def __init__(self, config, **kwargs):
        super(MTBaselineDNN, self).__init__()
        input_dim = config["input_dim"]
        layers = config["model_config"]["layers"]
        dropout = config["model_config"]["dropout"]
        num_tasks = config["num_tasks"]

        modules = [nn.Linear(input_dim, layers[0]), nn.ReLU()]
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            modules.append(nn.ReLU())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))

        self.feature_extractor = nn.Sequential(*modules)
        self.task_specific = nn.Linear(layers[-1], num_tasks)
        self.apply(self.init_wt)

    @staticmethod
    def init_wt(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = self.task_specific(features)
        return outputs


class PCMBaselineDNN(nn.Module):
    def __init__(self, config):
        super(PCMBaselineDNN, self).__init__()

        # Unpack configuration for chemical and protein branches
        chem_input_dim = config["chem_input_dim"]
        prot_input_dim = config["prot_input_dim"]
        chem_layers = config["chem_model_config"]["layers"]
        prot_layers = config["prot_model_config"]["layers"]
        dropout = config["model_config"]["dropout"]
        regressor_layers = config["regressor_config"]["layers"]

        # Chemical compound feature extractor
        chem_modules = [nn.Linear(chem_input_dim, chem_layers[0]), nn.ReLU()]
        for i in range(len(chem_layers) - 1):
            chem_modules.append(nn.Linear(chem_layers[i], chem_layers[i + 1]))
            chem_modules.append(nn.ReLU())
            if dropout > 0:
                chem_modules.append(nn.Dropout(dropout))
        self.chem_feature_extractor = nn.Sequential(*chem_modules)

        # Protein feature extractor
        prot_modules = [nn.Linear(prot_input_dim, prot_layers[0]), nn.ReLU()]
        for i in range(len(prot_layers) - 1):
            prot_modules.append(nn.Linear(prot_layers[i], prot_layers[i + 1]))
            prot_modules.append(nn.ReLU())
            if dropout > 0:
                prot_modules.append(nn.Dropout(dropout))
        self.prot_feature_extractor = nn.Sequential(*prot_modules)

        # Regressor construction
        combined_input_dim = chem_layers[-1] + prot_layers[-1]
        regressor_modules = [
            nn.Linear(combined_input_dim, regressor_layers[0]),
            nn.ReLU(),
        ]
        for i in range(len(regressor_layers) - 1):
            regressor_modules.append(
                nn.Linear(regressor_layers[i], regressor_layers[i + 1])
            )
            regressor_modules.append(nn.ReLU())
            if (
                dropout > 0 and i < len(regressor_layers) - 2
            ):  # No dropout before final layer
                regressor_modules.append(nn.Dropout(dropout))
        self.regressor = nn.Sequential(*regressor_modules)

        self.apply(self.init_wt)

    @staticmethod
    def init_wt(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, chem_input, prot_input):
        chem_features = self.chem_feature_extractor(chem_input)
        prot_features = self.prot_feature_extractor(prot_input)
        combined_features = torch.cat((chem_features, prot_features), dim=1)
        output = self.regressor(combined_features)
        return output


def run_baseline(
    datasets=None,
    config=None,
    activity="xc50",
    split="random",
    wandb_project_name=f"{TODAY}-baseline",
    seed=42,
    **kwargs,
):
    set_seed(seed)
    # Load config
    config = get_model_config(config=config, activity=activity, split=split, **kwargs)

    if datasets is None:
        # TODO fix this
        datasets = get_datasets(activity=activity, split=split)

    with wandb.init(
        dir=LOGS_DIR, mode=WANDB_MODE, project=wandb_project_name, config=config
    ):
        config = wandb.config
        # Load the dataset
        train_loader, val_loader, test_loader = build_loader(
            datasets, config.batch_size, config.input_dim
        )

        # Train the model
        best_model, loss_fn = train_model(
            train_loader, val_loader, config=config, seed=seed
        )

        # Testing metrics on the best model
        test_loss, test_rmse, test_r2, test_evs = evaluate(
            best_model, test_loader, loss_fn
        )

        # Log the final test metrics
        wandb.log(
            {
                "test/loss": test_loss,
                "test/rmse": test_rmse,
                "test/r2": test_r2,
                "test/evs": test_evs,
            }
        )


def run_baseline_hyperparam(
    config=None,
    activity="xc50",
    split="random",
    wandb_project_name=f"{TODAY}-baseline-hyperparam",
    sweep_count=1,
    seed=42,
    **kwargs,
):
    set_seed(seed)
    datasets = get_datasets(activity=activity, split=split)
    sweep_config = get_sweep_config(
        config=config, activity=activity, split=split, **kwargs
    )

    sweep_id = wandb.sweep(
        sweep_config,
        project=wandb_project_name,
    )

    wandb_train_func = partial(
        run_baseline,
        datasets=datasets,
        activity=activity,
        split=split,
        config=sweep_config,
        wandb_project_name=wandb_project_name,
        seed=42,
    )
    wandb.agent(sweep_id, function=wandb_train_func, count=sweep_count)


if __name__ == "__main__":
    test_loss = run_baseline()
