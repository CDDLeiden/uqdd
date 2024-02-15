from functools import partial

import wandb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from uqdd.models.models_utils import get_datasets, get_model_config, get_sweep_config, build_loader, build_optimizer, \
    save_models, calc_regr_metrics, set_seed, MultiTaskLoss

from uqdd import TODAY, DEVICE, LOGS_DIR, WANDB_MODE


class MTBaselineDNN(nn.Module):
    def __init__(
            self,
            config,
            **kwargs
    ):
        super(MTBaselineDNN, self).__init__()
        input_dim = config['input_dim']
        layers = config['model_config']['layers']
        dropout = config['model_config']['dropout']
        num_tasks = config['num_tasks']

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
            nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = self.task_specific(features)
        return outputs


class PCMBaselineDNN(nn.Module):
    pass


def run_baseline(
        datasets=None,
        config=None,
        activity="xc50",
        split="random",
        wandb_project_name=f"{TODAY}-baseline",
        seed=42,
        **kwargs
):
    set_seed(seed)
    # Load config
    config = get_model_config(config=config, activity=activity, split=split, **kwargs)

    if datasets is None:
        # TODO fix this
        datasets = get_datasets(activity=activity, split=split)

    with wandb.init(
            dir=LOGS_DIR,
            mode=WANDB_MODE,
            project=wandb_project_name,
            config=config
    ):
        config = wandb.config
        # Load the dataset
        train_loader, val_loader, test_loader = build_loader(datasets, config.batch_size, config.input_dim)

        # Train the model
        best_model, loss_fn = train_model(train_loader, val_loader, config=config, seed=seed)

        # Testing metrics on the best model
        test_loss, test_rmse, test_r2, test_evs = evaluate(best_model, test_loader, loss_fn)

        # Log the final test metrics
        wandb.log({
            'test/loss': test_loss,
            'test/rmse': test_rmse,
            'test/r2': test_r2,
            'test/evs': test_evs
        })


def run_baseline_hyperparam(
        config=None,
        activity="xc50",
        split="random",
        wandb_project_name=f"{TODAY}-baseline-hyperparam",
        sweep_count=1,
        seed=42,
        **kwargs
):
    set_seed(seed)
    datasets = get_datasets(activity=activity, split=split)
    sweep_config = get_sweep_config(config=config, activity=activity, split=split, **kwargs)

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
        seed=42
    )
    wandb.agent(sweep_id, function=wandb_train_func, count=sweep_count)


if __name__ == '__main__':
    test_loss = run_baseline()
