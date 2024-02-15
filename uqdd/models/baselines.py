from functools import partial

import wandb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from uqdd.models.models_utils import get_datasets, get_config, get_sweep_config, build_loader, build_optimizer, \
    save_models, calc_regr_metrics, set_seed, MultiTaskLoss

from uqdd import TODAY, DEVICE, LOGS_DIR, WANDB_MODE


class BaselineDNN(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim_1,
            hidden_dim_2=None,
            hidden_dim_3=None,
            num_tasks=1,
            dropout=0.2
    ):
        if hidden_dim_2 is None:
            hidden_dim_2 = hidden_dim_1
        if hidden_dim_3 is None:
            hidden_dim_3 = hidden_dim_1

        super(BaselineDNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )
        self.task_specific = nn.Linear(hidden_dim_3, num_tasks)
        self.apply(self.init_wt)

    @staticmethod
    def init_wt(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        features = self.feature_extractor(x)
        outputs = self.task_specific(features)
        return outputs


def train(
        model,
        loader,
        optimizer,
        loss_fn,
):
    model.train()
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # Zero the parameter gradients
        optimizer.zero_grad()
        # ➡ Forward pass
        outputs = model(inputs)

        # loss calculation
        loss = loss_fn(outputs, targets)

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    total_loss /= len(loader)
    return total_loss


def evaluate(
        model,
        loader,
        loss_fn,
):
    model.eval()
    total_loss = 0.0
    targets_all = []
    outputs_all = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            targets_all.append(targets)
            outputs_all.append(outputs)

        total_loss /= len(loader)
        targets_all = torch.cat(targets_all, dim=0)
        outputs_all = torch.cat(outputs_all, dim=0)
        # Calculate metrics
        rmse, r2, evs = calc_regr_metrics(targets_all, outputs_all)

    return total_loss, rmse, r2, evs


def predict(
        model,
        loader,
        return_targets=False,
):
    model.eval()
    outputs_all = []
    targets_all = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            outputs_all.append(outputs)
            if return_targets:
                targets_all.append(targets)
    outputs_all = torch.cat(outputs_all, dim=0)
    if return_targets:
        targets_all = torch.cat(targets_all, dim=0)
        return outputs_all, targets_all
    return outputs_all


def initial_evaluation(model, train_loader, val_loader, loss_fn):
    val_loss, val_rmse, val_r2, val_evs = evaluate(model, val_loader, loss_fn)
    train_loss, _, _, _ = evaluate(model, train_loader, loss_fn)
    return train_loss, val_loss, val_rmse, val_r2, val_evs


def run_epoch(model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler, epoch=0):
    """
    Run a single epoch of training and evaluation.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained and evaluated.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data.
    loss_fn : torch.nn.Module
        Loss function used for training and evaluation.
    optimizer : torch.optim.Optimizer
        Optimizer for model parameter updates.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    epoch : int, optional
        Current epoch number. Default is 0.

    Returns
    -------
    float
        Validation loss for the epoch.
    """
    if epoch == 0:
        # Perform evaluation before training starts (epoch 0)
        train_loss, val_loss, val_rmse, val_r2, val_evs = initial_evaluation(model, train_loader, val_loader, loss_fn)

    else:
        train_loss = train(model, train_loader, optimizer, loss_fn)
        val_loss, val_rmse, val_r2, val_evs = evaluate(model, val_loader, loss_fn)
        # Update the learning rate
        lr_scheduler.step(val_loss)

    return epoch, train_loss, val_loss, val_rmse, val_r2, val_evs


def train_model(
        train_loader,
        val_loader,
        config=wandb.config,
        seed=42
):
    try:
        # set a random seed for reproducibility
        # torch.manual_seed(seed)
        set_seed(seed)
        # deterministic cuda algorithms
        torch.backends.cudnn.deterministic = True

        # Load the model
        model = BaselineDNN(
            input_dim=config.input_dim,
            hidden_dim_1=config.hidden_dim_1,
            hidden_dim_2=config.hidden_dim_2,
            hidden_dim_3=config.hidden_dim_3,
            num_tasks=config.num_tasks,
            dropout=config.dropout
        )
        model = model.to(DEVICE)

        # Temporarily initialize best_model
        best_model = model

        # Define the loss function
        loss_fn = MultiTaskLoss(loss_type=config.loss, reduction='none')

        # Define the optimizer with weight decay and learning rate scheduler
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)

        # Define Learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.lr_factor,
            patience=config.lr_patience,
            verbose=True
        )

        # Train the model
        best_val_loss = float('inf')
        early_stop_counter = 0
        # Train the model
        for epoch in tqdm(range(config.num_epochs + 1)):
            try:
                epoch, train_loss, val_loss, val_rmse, val_r2, val_evs = run_epoch(
                    model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler, epoch=epoch
                )
                # Log the metrics
                wandb.log(
                    data={
                        'epoch': epoch,
                        'train/loss': train_loss,
                        'val/loss': val_loss,
                        'val/rmse': val_rmse,
                        'val/r2': val_r2,
                    }
                )
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    # Save the best model - dropped to avoid memory issues
                    # Update the best model and its performance
                    best_model = model

                else:
                    early_stop_counter += 1
                    if early_stop_counter > config.early_stop:
                        break

            except Exception as e:
                raise RuntimeError(f"The following exception occurred inside the epoch loop {e}")

        # Save the best model
        save_models(config, best_model)
        return best_model, loss_fn

    except Exception as e:
        raise Exception(f"The following exception occurred in train_model {e}")


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
    config = get_config(config=config, activity=activity, split=split, **kwargs)

    if datasets is None:
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
