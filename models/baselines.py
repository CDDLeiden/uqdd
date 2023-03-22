
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

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=3000, lr=0.01, momentum=0.9, nesterov=True, early_stop=200):
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
                print(f"Stopped early after {epoch+1} epochs")
                break

        # Print progress
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} -- Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    return model

def test_model(model, test_loader):
    criterion = nn.MSELoss()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
        test_loss /= len(test_loader)
    rmse = torch.sqrt(torch.tensor(test_loss)).item()
    return rmse

import random

def train_ensemble(train_data, val_data, test_data, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, num_epochs=3000):
    ensemble = []
    learning_rate = 0.005
    learning_rate_decay = 0.4
    early_stop = 200
    for i in range(100):
        # Set random seed for reproducibility
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)
        random.seed(seed)

        # Create and train model
        model = Net(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        early_stop_counter = 0
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = nn.MSELoss()(outputs, targets)
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
                    loss = nn.MSELoss()(outputs, targets)
                    val_loss += loss.item()
                val_loss /= len(val_loader)

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter == early_stop:
                    break

            # Learning Rate Decay
            if (epoch+1) % 200 == 0:
                learning_rate *= learning_rate_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

        # Test model and calculate uncertainties
        val_rmse = test_model(model, val_data)
        test_rmse = test_model(model, test_data)
        if val_rmse < 1.2:
            ensemble.append((model, val_rmse, test_rmse))

    return ensemble







net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_data = [] # TODO
targets_perf = {}
for i, (X_train, y_train) in enumerate(train_data):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []
    # for train_idx, val_idx in

# CHATGPT CODE
from sklearn.model_selection import train_test_split

targets = data['target_accession'].unique()

train_data = []
valid_data = []
test_data = []

for target in targets:
    target_data = data[data['target_accession'] == target]
    if len(target_data) < cutoff:
        continue
    X = target_data[['ecfp', 'physchem_properties']]
    y = target_data['pchembl']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    train_data.append((X_train, y_train))
    valid_data.append((X_valid, y_valid))
    test_data.append((X_test, y_test))

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data.csv')

# Filter by the number of compounds data points
cut_off = 100
data = data.groupby('target_accession').filter(lambda x: len(x) >= cut_off)

# Split the filtered dataset into training, validation, and test sets
train, test = train_test_split(data, test_size=0.2, random_state=42)
valid, test = train_test_split(test, test_size=0.5, random_state=42)

# Build a baseline model for each target
models = {}
for target in data['target_accession'].unique():
    # Extract the data for the current target
    train_data = train[train['target_accession'] == target]
    valid_data = valid[valid['target_accession'] == target]
    test_data = test[test['target_accession'] == target]

    # Extract the features and labels for the current target
    X_train = train_data[['ecfp', 'physchem_properties']]
    y_train = train_data['pchembl']
    X_valid = valid_data[['ecfp', 'physchem_properties']]
    y_valid = valid_data['pchembl']
    X_test = test_data[['ecfp', 'physchem_properties']]
    y_test = test_data['pchembl']

    # Train a linear regression model on the training set
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set using cross-validation
    scores = cross_val_score(model, X_valid, y_valid, cv=5)
    mean_score = scores.mean()
    print(f'Cross-validation score for {target}: {mean_score}')

    # Store the model and its validation score
    models[target] = {'model': model, 'score': mean_score}

# Evaluate the performance of the chosen models on the test set
test_scores = []
for target, model_info in models.items():
    # Extract the data and model for the current target
    test_data = test[test['target_accession'] == target]
    X_test = test_data[['ecfp', 'physchem_properties']]
    y_test = test_data['pchembl']
    model = model_info['model']

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    test_score = mean_squared_error(y_test, y_pred)
    print
