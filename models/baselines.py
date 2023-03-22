__author__ = "Bola Khalil"
__copyright__ = "Copyright 2022, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__license__ = "All rights reserved, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__version__ = "0.0.1"
__maintainer__ = "Bola Khalil"
__email__ = "bkhalil@its.jnj.com"
__status__ = "Development"

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from papyrus_scripts.modelling import pcm, qsar
import xgboost
import random
from tqdm import tqdm

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x.float()))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=3000,
        lr=0.01,
        lr_decay=0.0,
        momentum=0.9,
        nesterov=True,
        early_stop=200
):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    early_stop = early_stop
    early_stop_counter = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            # inputs = torch.tensor(inputs, dtype=torch.float)
            targets = targets.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
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