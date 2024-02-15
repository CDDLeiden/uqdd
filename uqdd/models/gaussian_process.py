import os
import sys

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from uqdd.models.gp_kernels import Tanimoto, SSK

from typing import Union
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from uqdd.models.models_utils import set_seed, get_model_config, get_datasets, get_tasks
from uqdd.models.models_utils import build_loader, build_optimizer, MultiTaskLoss, save_models
from uqdd.models.models_utils import UCTMetricsTable, process_preds

from uqdd.models.baselines import MTBaselineDNN, run_epoch, predict

# get today's date as yyyy/mm/dd format
from datetime import date

today = date.today()
today = today.strftime("%Y%m%d")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))
print(torch.version.cuda) if device == 'cuda' else None

LOG_DIR = os.environ.get('LOG_DIR')
DATA_DIR = os.environ.get('DATA_DIR')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
CONFIG_DIR = os.environ.get('CONFIG_DIR')

wandb_mode = 'online'  # 'data/papyrus_filtered_high_quality_xc50_01_standardized.csv'

def build_gp_model(config=wandb.config, kernel="tanimoto"):

    def objective_closure():
        return -m.log_marginal_likelihood()

    if kernel.lower() == "tanimoto":
        k = Tanimoto()

    elif kernel.lower() == "ssk":
        # kernel choices
        max_subsequence_length = 5
        alphabet = list(set("".join([x[0] for x in X_train])))

        k = SSK(batch_size=4000, gap_decay=0.46, match_decay=0.99, alphabet=alphabet,
                max_subsequence_length=max_subsequence_length, maxlen=85)
        cst = gpflow.kernels.Constant(2.75)

    m = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(np.mean(y_train)), kernel=k, noise_variance=1)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))
    print_summary(m)  # Model summary



    m = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(-1.7), kernel=cst * k, noise_variance=0.056)
    loss = m.log_marginal_likelihood()

    # fit model
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(m.training_loss, m.trainable_variables, options=dict(ftol=0.00001), compile=False)

    # mean and variance GP prediction
    y_pred, y_var = m.predict_f(X_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_test = y_scaler.inverse_transform(y_test)
    # Compute R^2, RMSE and MAE on test set molecules
    score = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)


    pass

def train_gp_model():
    pass

def run_gp_model():
    pass

if __name__ == '__main__':
    test_config = {}

    run_gp_model()

#
#
#
# # get today's date as yyyy/mm/dd format
# import os
# import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from datetime import date
# from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF, PairwiseKernel
# from functools import partial
#
# import torch
# import torch.nn as nn
# import wandb
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from tqdm import tqdm
# from uqdd.models.models_utils import get_datasets, get_config, get_sweep_config, build_loader, build_optimizer, \
#     save_models, calc_regr_metrics, set_seed, MultiTaskLoss
#
# today = date.today()
# today = today.strftime("%Y%m%d")
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device: " + str(device))
# print(torch.version.cuda) if device == 'cuda' else None
#
# LOG_DIR = os.environ.get('LOG_DIR')
# DATA_DIR = os.environ.get('DATA_DIR')
#
# wandb_mode = 'online'  # 'offline'
#
# import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, Kernel
# from scipy.spatial.distance import pdist, squareform
#
# # Define the Tanimoto kernel
# class TanimotoKernel(Kernel):
#     def __call__(self, X, Y=None, eval_gradient=False):
#         if Y is None:
#             Y = X
#         # Compute the Tanimoto similarity
#         dists = pdist(X, Y, metric='jaccard')
#         mat = np.exp(-dists)
#         return squareform(mat)
#
#     def diag(self, X):
#         return np.diag(self(X))
#
#     def is_stationary(self):
#         return True
#
#
# # Extract features from the baseline DNN model
# def extract_features(model, loader):
#     model.eval()
#     features = []
#     targets = []
#     with torch.no_grad():
#         for inputs, target in loader:
#             inputs = inputs.to(device)
#             feature = model.feature_extractor(inputs)
#             features.append(feature.cpu().numpy())
#             targets.append(target.cpu().numpy())
#     return np.vstack(features), np.vstack(targets)
#
#
# # Train the GPR
# def train_gpr(train_loader, model, kernel='RBF'):
#     X, y = extract_features(model, train_loader)
#     if kernel == 'RBF':
#         kernel = 1.0 * RBF()
#     elif kernel == 'Tanimoto':
#         kernel = TanimotoKernel()
#     gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
#     gpr.fit(X, y)
#     return gpr
#
#
# # Predict using the GPR
# def predict_gpr(gpr, loader, model):
#     X, _ = extract_features(model, loader)
#     y_pred, y_std = gpr.predict(X, return_std=True)
#     return y_pred, y_std
#
#
# # Example usage
# if __name__ == '__main__':
#     # Assuming train_loader and val_loader are already defined
#     # and the baseline model is trained
#     model = BaselineDNN(...)  # Define your model parameters
#     model.to(device)
#
#     # Train the GPR with RBF kernel
#     gpr_rbf = train_gpr(train_loader, model, kernel='RBF')
#     y_pred_rbf, y_std_rbf = predict_gpr(gpr_rbf, val_loader, model)
#
#     # Train the GPR with Tanimoto kernel
#     gpr_tanimoto = train_gpr(train_loader, model, kernel='Tanimoto')
#     y_pred_tanimoto, y_std_tanimoto = predict_gpr(gpr_tanimoto, val_loader, model)
