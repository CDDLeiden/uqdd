__author__ = "Bola Khalil"
__copyright__ = "Copyright 2022, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__license__ = "All rights reserved, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__version__ = "0.0.1"
__maintainer__ = "Bola Khalil"
__email__ = "bkhalil@its.jnj.com"
__status__ = "Development"

import os
import copy
import random
import wandb
import pandas as pd
from tqdm import tqdm

from papyrus import PapyrusDataset # build_top_dataset,
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader # , random_split

# from rdkit import Chem
from chemutils import smi_to_pil_image
# get today's date as yyyy/mm/dd format
from datetime import date
today = date.today()
today = today.strftime("%Y%m%d")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
print(torch.version.cuda) if device == 'cuda' else None

wandb_dir = 'logs/'
wandb_mode = 'online'
data_dir = 'data/' # 'data/papyrus_filtered_high_quality_xc50_01_standardized.csv'
dataset_dir = 'data/dataset/'
sweep_count = 50


class DNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim_1,
                 hidden_dim_2=None,
                 hidden_dim_3=None,
                 num_tasks=1,
                 dropout=0.2):
        if hidden_dim_2 is None:
            hidden_dim_2 = hidden_dim_1
        if hidden_dim_3 is None:
            hidden_dim_3 = hidden_dim_1

        super(DNN, self).__init__()
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
        # L1 or MSE loss
        # MSE is good if the target decreases error < 1 -> small gradients
        # something between L1 and MSE is smoothed L1
        # one linear layer - multi outputs faster to train
        # Taking init from nn.Linear
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
        dataloader,
        optimizer,
        loss_fn,
        # device=device
):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_fn, log="all", log_freq=100)

    model.train()
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Create a mask for Nan targets
        mask = torch.isnan(targets)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # ➡ Forward pass
        outputs = model(inputs)
        targets[mask], outputs[mask] = 0.0, 0.0
        # targets = targets[mask]
        # outputs = outputs * (mask)
        # outputs = outputs[mask]
        # Compute the MSE loss only for non-NaN values
        # TODO multiply outputs * inversed mask to get rid of NaN values
        # this multiplication will stop the gradients of thse nans
        # TODO replace the NaN values with 0.0 in the targets -> then use default reduction
        # all final loss will be small.
        # outpuot * inversed mask & targets slice or replace nan with 0.0
        # TODO check the ranges of the targets that they are technically the same.
        # mean of the values should be similar. if not --> then rescale the targets
        # or to use loss function with autoscaling -> kytorch loss function
        loss_per_task = loss_fn(outputs, targets)
        loss = torch.mean(loss_per_task, dim=0)
        loss = torch.sum(loss)
        total_loss += loss.item()
        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()
        # if i % 200 == 0:
        #     # log Batch loss and Batch entropy every 25 batches
        #     wandb.log({"batch loss": loss.item()})
    return total_loss / len(dataloader)


def evaluate(
        model,
        loader,
        loss_fn,
        # last_batch_log=False,
        # device=device
):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            mask = torch.isnan(targets)
            outputs = model(inputs)
            targets[mask], outputs[mask] = 0.0, 0.0
            loss_per_task = loss_fn(outputs, targets)
            task_losses = torch.mean(loss_per_task, dim=0)
            loss = torch.sum(task_losses)
            total_loss += loss.item()
        # if last_batch_log:
        #     targets_names = loader.dataset.target_col
        #     smiles = smiles.to(device)
        #     log_mol_table(smiles, inputs, targets, outputs, targets_names)

    return total_loss / len(loader)


def build_loader(config=wandb.config):
    d_dir = os.path.join(dataset_dir, config.activity)

    train_path = os.path.join(d_dir, "train.pkl")
    val_path = os.path.join(d_dir, "val.pkl")
    test_path = os.path.join(d_dir, "test.pkl")

    train_set = PapyrusDataset(train_path, input_col=f"ecfp{config.input_dim}", device=device)
    val_set = PapyrusDataset(val_path, input_col=f"ecfp{config.input_dim}", device=device)
    test_set = PapyrusDataset(test_path, input_col=f"ecfp{config.input_dim}", device=device)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True) # , pin_memory=True
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False) # , pin_memory=True
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False) # , pin_memory=True

    return train_loader, val_loader, test_loader


def build_optimizer(model, optimizer, lr, weight_decay):
    if optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Unknown optimizer: {}'.format(optimizer))

    return optimizer


def build_loss(loss, reduction='none'):
    if loss.lower() == 'mse':
        loss_fn = nn.MSELoss(reduction=reduction)
    elif loss.lower() in ['mae', 'l1']:
        loss_fn = nn.L1Loss(reduction=reduction)
    elif loss.lower() in ['huber', 'smoothl1']:
        loss_fn = nn.SmoothL1Loss(reduction=reduction)
    else:
        raise ValueError('Unknown loss: {}'.format(loss))
    return loss_fn


def model_pipeline():  # config=None
    # Initialize wandb
    with wandb.init(dir=wandb_dir, mode=wandb_mode):  # project='multitask-learning', config=config
        config = wandb.config

        # Load the dataset
        train_loader, val_loader, test_loader = build_loader(config)

        # Load the model
        model = DNN(
            input_dim=config.input_dim,
            hidden_dim_1=config.hidden_dim_1,
            hidden_dim_2=config.hidden_dim_2,
            hidden_dim_3=config.hidden_dim_3,
            # hidden_dim=config.hidden_dim,
            num_tasks=config.num_tasks,
            dropout=config.dropout
        )
        model = model.to(device)

        # Define the loss function
        loss_fn = build_loss(config.loss, reduction='none')

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
        for epoch in tqdm(range(config.num_epochs)):
            # Training
            train_loss = train(model, train_loader, optimizer, loss_fn)
            # Validation
            val_loss = evaluate(model, val_loader, loss_fn)
            # Log the metrics
            wandb.log(
                data={
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
            )

            # Update the learning rate
            lr_scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # Save the best model - dropped to avoid memory issues
                # Update the best model and its performance
                best_model = copy.deepcopy(model)

                # optional: save model at the end to view in wandb
                # inputs, _ = next(iter(train_loader)) # Takes so much time # TODO: put it before epochs loop
                # inputs = inputs.to(device)
                # x = torch.zeros((config.batch_size, config.input_dim), dtype=torch.float32, device=device,
                #                      requires_grad=False)
                # torch.onnx.export(model, x, f'models/{today}_best_model.onnx')
                # wandb.save(f"models/{today}_best_model.onnx")

            else:
                early_stop_counter += 1
                if early_stop_counter > config.early_stop:
                    break

        # Save the best model
        torch.save(best_model.state_dict(), f'models/{today}_best_model.pt')
        wandb.save(f"models/{today}_best_model.pt")

        # Load the best model
        # model.load_state_dict(torch.load(f'models/{today}_best_model.pt'))
        # Test
        test_loss = evaluate(best_model, test_loader, loss_fn)  # , last_batch_log=True
        # Log the final test metrics
        wandb.log({
            'test_loss': test_loss
        })

        return test_loss
        # Log the final hyperparameters
        # wandb.config.update({
        #     'best_val_loss': best_val_loss,
        #     'test_loss': test_loss
        # })


def run_pipeline(sweep=False):
    if sweep:
        # with wandb.init(dir=wandb_dir, mode=wandb_mode):
        sweep_config = get_sweep_config()
        sweep_id = wandb.sweep(
            sweep_config,
            project='2023-06-02-mtl-testing-hyperparam'
        )
        wandb.agent(sweep_id, function=model_pipeline, count=sweep_count)
        test_loss = None

    else:
        config = get_config()
        wandb.init(
            project='2023-06-02-mtl-testing',
            dir=wandb_dir,
            mode=wandb_mode,
            config=config
        )
        test_loss = model_pipeline()

    return test_loss


def get_config():
    config = {
        'input_dim': 1024,
        'hidden_dim_1': 512,
        'hidden_dim_2': 256,
        'hidden_dim_3': 64,
        'num_tasks': 20,
        'batch_size': 32,
        'loss': 'huber',
        'learning_rate': 0.001,
        'weight_decay': 0.01,  # 1e-5,
        'dropout': 0.2,
        'lr_factor': 0.1,
        'lr_patience': 10,
        'num_epochs': 10,  # 20,
        'optimizer': 'AdamW',
        'early_stop': 5,
        # 'n_tasks': 20,
        'output_dim': 20,
        'activity': "xc50"
    }

    return config


def get_sweep_config():
    # # Initialize wandb
    # wandb.init(project='multitask-learning')
    # Sweep configuration
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'input_dim': {
                'values': [1024, 2048]
            },
            # TODO : add hidden_dim_1, hidden_dim_2, hidden_dim_3
            # 'hidden_dim': {
            #     'values': [128, 256, 512]
            # },
            'hidden_dim_1': {
                'values': [512, 1024, 2048]
            },
            'hidden_dim_2': {
                'values': [256, 512]
            },
            'hidden_dim_3': {
                'values': [128, 256]
            },
            'num_tasks': {
                'value': 20
            },
            'batch_size': {
                'values': [64, 128, 256]
                # 'distribution': 'q_log_uniform',
                # 'q': 1,
                # 'min': math.log(32),
                # 'max': math.log(256)
            },
            'loss': {
                'values': ['huber', 'mse']
            },
            'learning_rate': {
                'values': [0.001, 0.01]
                # 'distribution': 'uniform',  # 'log_uniform',
                # 'min': 0,  # 0.001, # e-4
                # 'max': 0.01  # e-2
            },
            'ensemble_size': {
                'value': 100
                # 'values': [5, 10, 20]
            },
            'weight_decay': {
                'value': 0.001
            },
            'dropout': {
                # 'value': 0.2
                'values': [0.1, 0.2]
            },
            'lr_factor': {
                'value': 0.1
                # 'values': [0.1, 0.5]
            },
            'lr_patience': {
                'value': 10
                # 'values': [5, 10]
            },
            'num_epochs': {
                'value': 3000
                # 'values': [200, 3000] # [50, 100, 200]
                # 'values': [1]  # [50, 100, 200]
            },
            'early_stop': {
                'value': 100
            },
            'optimizer': {
                # 'value': 'AdamW'
                'values': ['adamw', 'sgd']
            },
            'output_dim': {
                'value': 20
            },
            'activity': {
                'value': "xc50"  # or kx
                # 'values': ['xc50', 'kx']
            },
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10
        }
    }
    # 576 combinations
    return sweep_config
    # sweep_id = wandb.sweep(
    #     sweep_config,
    #     project='multitask-learning-hyperparam'
    # )
    #
    # wandb.agent(sweep_id, function=model_pipeline, count=50)


def log_mol_table(smiles, inputs, targets, outputs, targets_names):
    # targets_cols = targets.columns()
    # table_cols = ['smiles', 'mol', 'mol_2D', 'ECFP', 'fp_length']
    # for t in targets_cols:
    #     table_cols.append(f'{t}_label')
    #     table_cols.append(f'{t}_predicted')
    # table = wandb.Table(columns=table_cols)
    # with wandb.init(dir=wandb_dir, mode=wandb_mode):
    data = []
    for smi, inp, tar, out in zip(smiles, inputs.to("cpu"), targets.to("cpu"), outputs.to("cpu")):
        row = {
            "smiles": smi,
            "molecule": wandb.Molecule.from_smiles(smi),
            "molecule_2D": wandb.Image(smi_to_pil_image(smi)),
            "ECFP": inp,
            "fp_length": len(inp),
        }

        # Iterate over each pair of output and target
        for targetName, target, output in zip(targets_names, tar, out):
            row[f'{targetName}_label'] = target.item()
            row[f'{targetName}_predicted'] = output.item()

        data.append(row)

    dataframe = pd.DataFrame.from_records(data)
    table = wandb.Table(dataframe=dataframe)
    wandb.log({"mols_table": table}, commit=False)


    # "molecules": [substance.get("molecule") for substance in data]

    # table = wandb.Table(data=data)
    #
    #     table.add_data(
    #         smi,
    #         wandb.Molecule.from_smiles(smi),
    #         wandb.Image(smi_to_pil_image(smi)),
    #         inp,
    #         len(inp),
    #         # *[t, o for t, o in zip(tar, out)]
    #     )
    # wandb.log({"mols_table": table}, commit=False)
    #
    #
    # for smiles, pred, label, prob in zip(smiles_list.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
    #     # TODO SMILES TO MOL IMAGE logging here
    #     # Convert SMILES string to RDKit molecule object
    #     mol = Chem.MolFromSmiles(smiles)
    #
    #     # Generate PIL image of the molecule
    #     mol_img = Chem.Draw.MolToImage(mol)
    #     # TODO : Need modifications because of MultiTask Learning.
    #     table.add_data(mol_img, pred, label, *prob.numpy())
    #
    # wandb.log({"mols_table": table}, commit=False)
# def log_mol_table(smiles_list, predicted, labels, probs, targets):
#     data = []
#
#     table = wandb.Table(columns=['mol', 'predicted', 'labels'] + [f'probs_{t}' for t in targets])
#     for smiles, pred, label, prob in zip(smiles_list.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
#         # TODO SMILES TO MOL IMAGE logging here
#         # Convert SMILES string to RDKit molecule object
#         mol = Chem.MolFromSmiles(smiles)
#
#         # Generate PIL image of the molecule
#         mol_img = Chem.Draw.MolToImage(mol)
#         # TODO : Need modifications because of MultiTask Learning.
#         table.add_data(mol_img, pred, label, *prob.numpy())
#
#
#     wandb.log({"mols_table": table}, commit=False)

#
# def train_model(
#         model,
#         train_loader,
#         val_loader,
#         num_epochs=3000,
#         lr=0.01,
#         lr_decay=0.0,
#         momentum=0.9,
#         nesterov=True,
#         early_stop=200,
#         # device="cuda"
# ):
#     #### similar to deep confidence
#     run = wandb.init(
#         project="uqdd",
#         # Track hyperparameters and run metadata
#         config={
#             "learning_rate": lr,
#             "epochs": num_epochs,
#         },
#         dir=wandb_dir,
#         mode=wandb_mode
#     )
#     model.to(device)
#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
#     early_stop = early_stop
#     early_stop_counter = 0
#     best_val_loss = float('inf')
#     for epoch in range(num_epochs):
#         # Training
#         model.train()
#         train_loss = 0.0
#         # train_entropy = 0.0
#         for inputs, targets in train_loader:
#             inputs = inputs.to(device)
#             targets = targets.to(device)
#
#             optimizer.zero_grad()
#             logits = model(inputs)
#             loss = loss_fn(logits, targets)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#         train_loss /= len(train_loader)
#
#         # Validation
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for inputs, targets in val_loader:
#                 outputs = model(inputs)
#                 targets = targets.float().unsqueeze(1)
#                 loss = loss_fn(outputs, targets)
#                 val_loss += loss.item()
#             val_loss /= len(val_loader)
#
#         # Early Stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             early_stop_counter = 0
#         else:
#             early_stop_counter += 1
#             if early_stop_counter == early_stop:
#                 print(f"Stopped early after {epoch + 1} epochs")
#                 break
#
#         # Print progress
#         if (epoch + 1) % 100 == 0:
#             print(f"Epoch {epoch + 1}/{num_epochs} -- Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
#
#         # Learning Rate Decay
#         if (epoch + 1) % 200 == 0 and lr_decay > 0:
#             lr *= lr_decay
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr
#
#     return model
#
#
# def test_model(model, test_loader):
#     criterion = nn.MSELoss()
#     test_loss = 0.0
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             outputs = model(inputs)
#             targets = targets.float().unsqueeze(1)
#             loss = criterion(outputs, targets)
#             test_loss += loss.item()
#         test_loss /= len(test_loader)
#     rmse = torch.sqrt(torch.tensor(test_loss)).item()
#     return rmse
#
#
# def train_ensemble(
#         train_loader, val_loader, test_loader,
#         input_size, hidden_size1, hidden_size2, hidden_size3,
#         output_size, num_epochs=3000):
#     ensemble = []
#     learning_rate = 0.005
#     learning_rate_decay = 0.4
#     early_stop = 200
#     for _ in range(100):
#         # Set random seed for reproducibility
#         seed = random.randint(0, 10000)
#         torch.manual_seed(seed)
#         random.seed(seed)
#
#         # Create and train model
#         model = DNN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
#         model = train_model(
#             model,
#             train_loader,
#             val_loader,
#             num_epochs=num_epochs,
#             lr=learning_rate,
#             lr_decay=learning_rate_decay,
#             momentum=0.9,
#             nesterov=True,
#             early_stop=early_stop
#         )
#
#         # Test model and calculate uncertainties
#         val_rmse = test_model(model, val_loader)
#         test_rmse = test_model(model, test_loader)
#
#         if val_rmse < 1.2:
#             ensemble.append((model, val_rmse, test_rmse))
#
#     return ensemble
#

if __name__ == '__main__':
    best_model, test_loss = run_pipeline()
    # run_model()
    # hyperparam_search()

# class MultiTaskLossWrapper(nn.Module):
#     def __init__(self, task_num, model):
#         super(MultiTaskLossWrapper, self).__init__()
#         self.model = model
#         self.task_num = task_num
#         self.log_vars = nn.Parameter(torch.zeros((task_num)))
#
#     def forward(self, input, targets):
#         outputs = self.model(input)
#
#         loss = 0.0
#         for task_index in range(self.task_num):
#             precision = torch.exp(-self.log_vars[task_index])
#             loss += torch.sum(precision * (targets[task_index] - outputs[task_index]) ** 2. + self.log_vars[task_index], -1)
#
#         precision1 = torch.exp(-self.log_vars[0])
#         loss = torch.sum(precision1 * (targets[0] - outputs[0]) ** 2. + self.log_vars[0], -1)
#
#         precision2 = torch.exp(-self.log_vars[1])
#         loss += torch.sum(precision2 * (targets[1] - outputs[1]) ** 2. + self.log_vars[1], -1)
#
#         loss = torch.mean(loss)
#
#         return loss, self.log_vars.data.tolist()


#
# def train(
#         model,
#         dataloader,
#         optimizer,
#         loss_fn,
#         # device=device
# ):
#     # Tell wandb to watch what the model gets up to: gradients, weights, and more!
#     wandb.watch(model, loss_fn, log="all", log_freq=10)
#
#     model.train()
#     total_loss = 0.0
#     # total_entropy = 0.0
#     for i, (inputs, targets) in enumerate(dataloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         # Create a mask for Nan targets
#         mask = torch.isnan(targets)
#         # Zero the parameter gradients
#         optimizer.zero_grad()
#         # ➡ Forward pass
#         logits = model(inputs)
#         # Compute the MSE loss only for non-NaN values
#         loss_per_task = loss_fn(logits, targets)  # , reduction='none'
#         loss_per_task[mask] = 0.0
#         task_losses = torch.mean(loss_per_task, dim=0)
#         loss = torch.sum(task_losses)
#         total_loss += loss.item()
#
#         # ⬅ Backward pass + weight update
#         loss.backward()
#         optimizer.step()
#         # TODO : wrapper of the loss function - Multiple tasks - Multiple losses add up the dimension
#         #  and average over the batch dimension
#         #  reduction = 'none'
#         #  https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
#         #  use none --> (batch, tasks) torch.sum dim 1 --> torch.mean dim 0 --> scalar
#
#         # batch entropy
#         # TODO : doesnt make sense to calculate entropy
#         #  - only for classification - in RL they want to know how commited an action is.
#         # prob = torch.softmax(logits, dim=-1)
#         # entropy = -torch.sum(prob * torch.log(prob), dim=-1)
#
#         if i % 25 == 0:
#             # log Batch loss and Batch entropy every 25 batches
#             wandb.log({"batch loss": loss.item()})
#             # wandb.log({"batch entropy": torch.mean(entropy).item()})
#         # total_entropy += torch.mean(entropy).item()
#     return total_loss / len(dataloader)  # , total_entropy / len(dataloader)


# config_keys = ["input_dim", "hidden_size", "learning_rate", "weight_decay", "dropout", "lr_factor", "lr_patience",
#                "batch_size", "num_epochs", "n_tasks", "activity", "optimizer"]

# Define a list of configurations to search over
# configurations = {
#
#     'input_dim': 1024,
#     'hidden_size': [128, 256, 512],
#     'learning_rate': [0.001, 0.01],
#     'weight_decay': [1e-5, 1e-4],
#     'dropout': [0.1, 0.2],
#     'lr_factor': [0.1, 0.5],
#     'lr_patience': [5, 10],
#     'batch_size': [32, 64],
#     'num_epochs': [50, 100]
# }

# def run_model():
#     # Initialize wandb
#     config = get_config()
#     print(config)
#     # wandb.init(
#     #     project='multitask-learning-2',
#     #     config={
#     #         'input_dim': 1024,
#     #         'hidden_dim_1': 512,
#     #         'hidden_dim_2': 256,
#     #         'hidden_dim_3': 64,
#     #         'batch_size': 64,
#     #         'learning_rate': 0.001,
#     #         'weight_decay': 0.01, # 1e-5,
#     #         'dropout': 0.2,
#     #         'lr_factor': 0.1,
#     #         'lr_patience': 10,
#     #         'num_epochs': 20,
#     #         'optimizer': 'adam',
#     #         'early_stop': 5,
#     #         'n_tasks': 20,
#     #         'output_dim': 20,
#     #         'activity': "xc50"
#     #     }
#     # )
#     model_pipeline()
#



### DEPRECATED
#
# def _build_loader_(config=wandb.config, dataset=None):
#     if dataset is None:
#         dataset = build_top_dataset(
#             data_path=config.data_dir,
#             activity=config.activity,  # "xc50",
#             n_top=config.num_tasks,
#             multitask=True
#         )
#
#     columns = dataset.columns
#     columns = columns[1:]
#
#     # train_size = int(0.7 * len(dataset))
#     # val_size = int(0.15 * len(dataset))
#     # test_size = len(dataset) - train_size - val_size
#     # train_set, val_set, test_set = random_split(dataset, lengths=[train_size, val_size, test_size])
#
#     train_data, test_data = train_test_split(
#         dataset, test_size=0.3, shuffle=True, random_state=42
#     )
#     val_data, test_data = train_test_split(
#         test_data, test_size=0.5, shuffle=True, random_state=42
#     )
#
#     train_set = PapyrusDataset(train_data, input_col='smiles', target_col=list(columns)) # , length=config.input_dim
#     val_set = PapyrusDataset(val_data, input_col='smiles', target_col=list(columns))
#     test_set = PapyrusDataset(test_data, input_col='smiles', target_col=list(columns))
#
#     train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
#     test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
#
#     return train_loader, val_loader, test_loader
#
