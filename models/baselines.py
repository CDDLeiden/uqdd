__author__ = "Bola Khalil"
__copyright__ = "Copyright 2022, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__license__ = "All rights reserved, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__version__ = "0.0.1"
__maintainer__ = "Bola Khalil"
__email__ = "bkhalil@its.jnj.com"
__status__ = "Development"

import math
import random
from tqdm import tqdm
from papyrus import build_top_dataset, PapyrusDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from torch.utils.data import DataLoader, random_split

from rdkit import Chem
from sklearn.model_selection import train_test_split, KFold
from papyrus_scripts.modelling import pcm, qsar
import xgboost


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(DNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.task_specific = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(output_dim)
        ])
        self.apply(self.init_wt)

    @staticmethod
    def init_wt(n):
        if isinstance(n, nn.Linear):
            nn.init.xavier_uniform_(n.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.feature_extractor(x)
        outputs = []
        for task in self.task_specific:
            outputs.append(task(x))
        return torch.cat(outputs, dim=1)


def train(
        model,
        dataloader,
        optimizer,
        loss_fn,
        # device=device
):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_fn, log="all", log_freq=10)

    model.train()
    total_loss = 0.0
    total_entropy = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Create a mask for non-Nan targets
        mask = ~torch.isnan(targets)

        # Zero the parameter gradients
        optimizer.zero_grad()
        # ➡ Forward pass
        logits = model(inputs)
        # Compute the MSE loss only for non-NaN values
        loss = loss_fn(logits[mask], targets[mask])
        total_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        # batch entropy
        prob = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(prob * torch.log(prob), dim=-1)
        if i % 25 == 0:
            # log Batch loss and Batch entropy every 25 batches
            wandb.log({"batch loss": loss.item()})
            wandb.log({"batch entropy": torch.mean(entropy).item()})
        total_entropy += torch.mean(entropy).item()
    return total_loss / len(dataloader), total_entropy / len(dataloader)


def evaluate(
        model,
        loader,
        loss_fn,
        # device=device
):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            mask = ~torch.isnan(targets)

            logits = model(inputs)
            loss = loss_fn(logits[mask], targets[mask])
            total_loss += loss.item()
            # MSE = ((torch.pow((logits - targets), 2)).sum()) / total

    return total_loss / len(loader)


def build_loader(config=None):

    dataset = build_top_dataset(
        data_path="data/papyrus_filtered_high_quality_xc50_01_standardized.csv",
        activity=config.activity, #"xc50",
        n_top=config.n_tasks,
        multitask=True
    )
    columns = dataset.columns
    columns = columns[1:]

    # train_size = int(0.7 * len(dataset))
    # val_size = int(0.15 * len(dataset))
    # test_size = len(dataset) - train_size - val_size
    # train_set, val_set, test_set = random_split(dataset, lengths=[train_size, val_size, test_size])

    train_data, test_data = train_test_split(
        dataset, test_size=0.3, shuffle=True, random_state=42
    )
    val_data, test_data = train_test_split(
        test_data, test_size=0.5, shuffle=True, random_state=42
    )

    train_set = PapyrusDataset(train_data, input_col='smiles', target_col=list(columns))
    val_set = PapyrusDataset(val_data, input_col='smiles', target_col=list(columns))
    test_set = PapyrusDataset(test_data, input_col='smiles', target_col=list(columns))

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def build_optimizer(model, optimizer, lr, weight_decay):
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Unknown optimizer: {}'.format(optimizer))

    return optimizer


def model_pipeline(): #config=None
    # Initialize wandb
    with wandb.init(): #project='multitask-learning', config=config
        config = wandb.config

        # Load the dataset
        train_loader, val_loader, test_loader = build_loader(config) # wandb.config.config.batch_size

        # Load the model
        model = DNN(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            dropout=config.dropout
        ).to(device)

        # Define the loss function
        loss_fn = nn.MSELoss()

        # Define the optimizer with weight decay and learning rate scheduler
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
        # optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

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
            train_loss, train_entropy = train(model, train_loader, optimizer, loss_fn)
            # Validation
            val_loss = evaluate(model, val_loader, loss_fn)
            # Log the metrics
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_entropy': train_entropy,
                'val_loss': val_loss
            })
            # Update the learning rate
            lr_scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # Save the best model
                torch.save(model.state_dict(), 'models/best_model.pt')
                # optional: save model at the end to view in wandb
                torch.onnx.export(model, 'models/best_model.onnx')
                wandb.save("models/best_model.onnx")

            else:
                early_stop_counter += 1
                if early_stop_counter > config.early_stop:
                    break

        # Load the best model
        model.load_state_dict(torch.load('models/best_model.pt'))
        # Test
        test_loss = evaluate(model, test_loader, loss_fn)
        # Log the final test metrics
        wandb.log({
            'test_loss': test_loss
        })

        # Log the final hyperparameters
        # wandb.config.update({
        #     'best_val_loss': best_val_loss,
        #     'test_loss': test_loss
        # })


def hyperparam_search():
    # # Initialize wandb
    # wandb.init(project='multitask-learning')

    # Sweep configuration
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'test_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'input_dim': {
                'value': 1024
            },
            'hidden_dim': {
                'values': [128, 256, 512]
            },
            'batch_size': {
                'distribution': 'q_log_uniform',
                'q': 1,
                'min': math.log(32),
                'max': math.log(256)
            },
            'learning_rate': {
                'distribution': 'uniform',
                'min': 0, # 0.001,
                'max': 0.01
            },
            'weight_decay': {
                'values': [1e-5, 1e-4]
            },
            'dropout': {
                'values': [0.1, 0.2]
            },
            'lr_factor': {
                'values': [0.1, 0.5]
            },
            'lr_patience': {
                'values': [5, 10]
            },
            'num_epochs': {
                'values': [1] # [50, 100, 200]
            },
            'optimizer': {
                'values': ['adam', 'sgd']
            },
            'early_stop': {
                'value': 10
            },
            'n_tasks': {
                'value': 20
            },
            'output_dim': {
                'value': 20
            },
            'activity': {
                'value': "xc50"
            },
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10
        }
    }

    sweep_id = wandb.sweep(
        sweep_config,
        project='multitask-learning-hyperparam'
    )

    wandb.agent(sweep_id, function=model_pipeline, count=50)


def run_model():
    # Initialize wandb
    wandb.init(
        project='multitask-learning-2',
        config= {
            'input_dim': 1024,
            'hidden_dim': 256,
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'dropout': 0.1,
            'lr_factor': 0.1,
            'lr_patience': 10,
            'num_epochs': 20,
            'optimizer': 'adam',
            'early_stop': 5,
            'n_tasks': 20,
            'output_dim': 20,
            'activity': "xc50"
        }
    )
    model_pipeline()


def log_mol_table(smiles_list, predicted, labels, probs, targets):
    table = wandb.Table(columns=['mol', 'predicted', 'labels'] + [f'probs_{t}' for t in targets])
    for smiles, pred, label, prob in zip(smiles_list.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        # TODO SMILES TO MOL IMAGE logging here
        # Convert SMILES string to RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)

        # Generate PIL image of the molecule
        mol_img = Chem.Draw.MolToImage(mol)
        # TODO : Need modifications because of MultiTask Learning.
        table.add_data(mol_img, pred, label, *prob.numpy())

    wandb.log({"mols_table": table}, commit=False)







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



def train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=3000,
        lr=0.01,
        lr_decay=0.0,
        momentum=0.9,
        nesterov=True,
        early_stop=200,
        device="cuda"
):
    run = wandb.init(
        project="uqdd",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": num_epochs,

        })

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    early_stop = early_stop
    early_stop_counter = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_entropy = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # inputs = torch.tensor(inputs, dtype=torch.float)
            # targets = targets.float().unsqueeze(1) # TODO check this line if it is important

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                targets = targets.float().unsqueeze(1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            val_loss /= len(val_loader)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter == early_stop:
                print(f"Stopped early after {epoch + 1} epochs")
                break

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} -- Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Learning Rate Decay
        if (epoch + 1) % 200 == 0 and lr_decay > 0:
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    return model


def test_model(model, test_loader):
    criterion = nn.MSELoss()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            targets = targets.float().unsqueeze(1)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
        test_loss /= len(test_loader)
    rmse = torch.sqrt(torch.tensor(test_loss)).item()
    return rmse


def train_ensemble(
        train_loader, val_loader, test_loader,
        input_size, hidden_size1, hidden_size2, hidden_size3,
        output_size, num_epochs=3000):
    ensemble = []
    learning_rate = 0.005
    learning_rate_decay = 0.4
    early_stop = 200
    for _ in range(100):
        # Set random seed for reproducibility
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)
        random.seed(seed)

        # Create and train model
        model = DNN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
        model = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            lr=learning_rate,
            lr_decay=learning_rate_decay,
            momentum=0.9,
            nesterov=True,
            early_stop=early_stop
        )

        # Test model and calculate uncertainties
        val_rmse = test_model(model, val_loader)
        test_rmse = test_model(model, test_loader)

        if val_rmse < 1.2:
            ensemble.append((model, val_rmse, test_rmse))

    return ensemble


if __name__ == '__main__':
    run_model()
    # hyperparam_search()