__author__ = "Bola Khalil"
__supervisor__ = "Kajetan Schweighofer"
__copyright__ = "Copyright 2022, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__license__ = "All rights reserved, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__version__ = "0.0.1"
__maintainer__ = "Bola Khalil"
__email__ = "bkhalil@its.jnj.com"
__status__ = "Development"

import copy
import wandb

from tqdm import tqdm
from functools import partial
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.models_utils import build_loader, build_optimizer, build_loss, save_models, calc_loss_notnan, calc_regr_metrics, log_mol_table


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
        'activity': "xc50",
        # 'wandb_name': "2023-06-02-mtl-testing",
    }

    return config


def get_sweep_config():
    # # Initialize wandb
    # wandb.init(project='multitask-learning')
    # Sweep configuration
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_rmse', # 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'input_dim': {
                'values': [1024, 2048]
            },
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
            },
            'loss': {
                'values': ['huber', 'mse']
            },
            'learning_rate': {
                'values': [0.001, 0.01]
            },
            'ensemble_size': {
                'value': 100
            },
            'weight_decay': {
                'value': 0.001
            },
            'dropout': {
                'values': [0.1, 0.2]
            },
            'lr_factor': {
                'value': 0.5
            },
            'lr_patience': {
                'value': 20
            },
            'num_epochs': {
                'value': 3000
            },
            'early_stop': {
                'value': 100
            },
            'optimizer': {
                'values': ['adamw', 'sgd']
            },
            'output_dim': {
                'value': 20
            },
            'activity': {
                'value': "xc50"
            },
        },
    }
    # 576 combinations
    return sweep_config


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
    # wandb.watch(model, loss_fn, log="all", log_freq=100)

    model.train()
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        # ➡ Forward pass
        outputs = model(inputs)
        # Create a mask for Nan targets
        nan_mask = torch.isnan(targets)
        # Loss calculation without nan in the mean
        loss = calc_loss_notnan(outputs, targets, nan_mask, loss_fn)
        total_loss += loss.item()
        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()
        # if i % 100 == 0:
        #     # Log the loss in your run history
        #     wandb.log({"batch loss": loss})
            
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
    targets_all = []
    outputs_all = []
    # nan_masks_all = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Create a mask for Nan targets
            nan_mask = torch.isnan(targets)
            # Loss calculation without nan in the mean
            loss = calc_loss_notnan(outputs, targets, nan_mask, loss_fn)
            total_loss += loss.item()

            targets_all.append(targets)
            outputs_all.append(outputs)
            # nan_masks_all.append(nan_mask)

        total_loss /= len(loader)
        targets_all = torch.cat(targets_all, dim=0)
        outputs_all = torch.cat(outputs_all, dim=0)
        # nan_masks_all = torch.cat(nan_masks_all, dim=0)

        # Calculate metrics
        rmse, r2, evs = calc_regr_metrics(targets_all, outputs_all)

        # mask = torch.isnan(targets)
        # valid_targets = targets[~mask]
        # valid_outputs = outputs[~mask]

        # targets[mask], outputs[mask] = 0.0, 0.0
        # loss_per_task = loss_fn(outputs, targets)
        # task_losses = torch.mean(loss_per_task, dim=0)
        # loss = torch.sum(task_losses)
        # total_loss += loss.item()

        # if last_batch_log:
        #     targets_names = loader.dataset.target_col
        #     smiles = smiles.to(device)
        #     log_mol_table(smiles, inputs, targets, outputs, targets_names)

    return total_loss, rmse, r2, evs


def model_pipeline(config=wandb.config, wandb_project_name="test-project"):  #
    with wandb.init(
            dir=wandb_dir,
            mode=wandb_mode,
            project=wandb_project_name,
            config=config
    ):
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
            val_loss, val_rmse, val_r2, val_evs = evaluate(model, val_loader, loss_fn)
            # Log the metrics
            wandb.log(
                data={
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'val_evs': val_evs
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
                best_model = model
                # best_model = copy.deepcopy(model)

                # Save the best model
                # save_models(config, model)

                # optional: save model at the end to view in wandb
                # inputs, _ = next(iter(train_loader)) # Takes so much time # TODO: put it before epochs loop
                # inputs = inputs.to(device)
                # x = torch.zeros((config.batch_size, config.input_dim), dtype=torch.float32, device=device,
                #                      requires_grad=False)
                # torch.onnx.export(model, x, f'models/{today}_best_model.onnx')
                # wandb.save(f"models/{today}_best_model.onnx")

            else:
                early_stop_counter += 1
                # print(config)
                if early_stop_counter > config.early_stop:
                    break

        # Save the best model
        save_models(config, best_model)
        # saved_model_path = f'models/saved_models/{config.activity}/'
        # if not os.path.exists(saved_model_path):
        # torch.save(best_model.state_dict(), f'models/{today}_best_model.pt')
        # wandb.save(f"models/saved_models/{today}_best_model.pt")

        # Load the best model
        # model.load_state_dict(torch.load(f'models/{today}_best_model.pt'))
        # Test
        test_loss, test_rmse, test_r2, test_evs = evaluate(best_model, test_loader, loss_fn)  # , last_batch_log=True
        # Log the final test metrics
        wandb.log({
            'test_loss': test_loss,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_evs': test_evs
        })

        return test_loss, test_rmse, test_r2, test_evs
        # Log the final hyperparameters
        # wandb.config.update({
        #     'best_val_loss': best_val_loss,
        #     'test_loss': test_loss
        # })


def run_baseline(wandb_project_name=f"{today}-baseline"):
    config = get_config()
    test_loss, test_rmse, test_r2, test_evs = model_pipeline(
        config,
        wandb_project_name=wandb_project_name
    )
    return test_loss, test_rmse, test_r2, test_evs


def run_baseline_hyperparam(wandb_project_name=f"{today}-baseline-hyperparam", sweep_count=1):
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(
        sweep_config,
        project=wandb_project_name,
    )
    wandb_train_func = partial(
        model_pipeline,
        config=sweep_config,
        wandb_project_name=wandb_project_name,
    )
    wandb.agent(sweep_id, function=wandb_train_func, count=sweep_count)
    # test_loss = None
    # return test_loss


# def run_pipeline(wandb_project_name="test-project", sweep=False, sweep_count=1):
#
#     if sweep:
#         # with wandb.init(dir=wandb_dir, mode=wandb_mode):
#         sweep_config = get_sweep_config()
#         sweep_id = wandb.sweep(
#             sweep_config,
#             project=wandb_project_name,
#         )
#         wandb_train_func = partial(
#             model_pipeline,
#             config=sweep_config,
#             wandb_project_name=wandb_project_name,
#             # wandb_project_name="2023-06-02-mtl-testing-hyperparam"
#         )
#         wandb.agent(sweep_id, function=wandb_train_func, count=sweep_count)
#         test_loss = None
#
#     else:
#         config = get_config()
#         # wandb.init(
#         #     project='2023-06-02-mtl-testing',
#         #     dir=wandb_dir,
#         #     mode=wandb_mode,
#         #     config=config
#         # )
#         test_loss = model_pipeline(
#             config,
#             wandb_project_name=wandb_project_name,
#             # wandb_project_name="2023-06-02-mtl-testing"
#         )
#
#     return test_loss




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
    test_loss = run_baseline()
    # best_model, test_loss = run_pipeline()
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
