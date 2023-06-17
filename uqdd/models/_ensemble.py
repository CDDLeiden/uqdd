# get today's date as yyyy/mm/dd format
import os
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from uqdd.models.models_utils import get_config, set_seed, build_loader, build_optimizer, build_loss, save_models, \
    calc_loss_notnan, calc_regr_metrics, get_datasets, MultiTaskLoss
from uqdd.models.baselines import BaselineDNN, train, evaluate

from torchensemble import FusionRegressor, VotingRegressor, BaggingRegressor, GradientBoostingRegressor, \
    NeuralForestRegressor, SnapshotEnsembleRegressor, AdversarialTrainingRegressor, FastGeometricRegressor, \
    SoftGradientBoostingRegressor


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

methods = {
    'fusion': FusionRegressor,
    'voting': VotingRegressor,
    'bagging': BaggingRegressor,
    'gbdt': GradientBoostingRegressor,
    'neuralforest': NeuralForestRegressor,
    'snapshot': SnapshotEnsembleRegressor,
    'adversarial': AdversarialTrainingRegressor,
    'fastgeometric': FastGeometricRegressor,
    'softgbdt': SoftGradientBoostingRegressor
}


def build_ensemble(config=wandb.config):
    ensemble_models = []
    try:
        seed = config.seed
    except AttributeError:
        seed = 42
    # deterministic cuda algorithms
    torch.backends.cudnn.deterministic = True

    for _ in range(config.ensemble_size):
        set_seed(seed)
        model = BaselineDNN(
            config.input_dim,
            config.hidden_dim_1,
            config.hidden_dim_2,
            config.hidden_dim_3,
            config.output_dim,
            config.dropout
        )
        ensemble_models.append(model)
        seed += 1

    return ensemble_models


def run_ensemble(
        datasets=None,
        config='uqdd/config/ensemble/ensemble.json',
        activity='xc50',
        split='random',
        ensemble_size=100,
        ensemble_method='fusion',
        wandb_project_name='multitask-learning-ensemble',
        seed=42,
        **kwargs
        # optimizer,
        # loss_fn,
):
    # Load the config
    config = get_config(config=config, activity=activity, split=split, ensemble_size=ensemble_size, **kwargs)

    # Load the dataset
    if datasets is None:
        datasets = get_datasets(activity=activity, split=split)


    # Initialize wandb for the ensemble models
    with wandb.init(
            dir=LOG_DIR,
            mode=wandb_mode,
            project=wandb_project_name,
            config=config
    ):
        config = wandb.config

        # Define the data loaders
        train_loader, val_loader, test_loader = build_loader(datasets, config.batch_size, config.input_dim)
        # Define the ensemble models
        model = BaselineDNN(
            config.input_dim,
            config.hidden_dim_1,
            config.hidden_dim_2,
            config.hidden_dim_3,
            config.output_dim,
            config.dropout
        )
        model.to(device)
        # Get the ensemble method
        ensemble_method = methods[ensemble_method]
        ensemble_model = ensemble_method(
            model,
            n_estimators=config.ensemble_size,
            cuda=torch.cuda.is_available(),
        )
        # Define the loss function
        loss_fn = MultiTaskLoss(
            loss_type=config.loss,
            reduction='none',
        )

        # test_loader
        ensemble_model.set_criterion(loss_fn)

        # Define the optimizer with weight decay and learning rate scheduler
        ensemble_model.set_optimizer(config.optimizer, lr=config.learning_rate, weight_decay=config.weight_decay)

        ensemble_model.set_scheduler(
            config.lr_scheduler,
            mode='min',
            factor=config.lr_factor,
            patience=config.lr_patience,
            verbose=True
        )

        ensemble_model.fit(
            train_loader=train_loader,
            epochs=config.num_epochs,
            test_loader=val_loader,
            save_model=True,
            save_dir=LOG_DIR
        )

        # Evaluate the model
        test_loss = ensemble_model.evaluate(test_loader)
        test_predictions = ensemble_model.predict(test_loader)

        return test_loss, test_predictions

        # for i in tqdm(range(config.ensemble_size), desc='Ensemble models'):
        #     # Get the model
        #     model = ensemble_models[i]
        #     model.to(device)
        #
        #     # Define the loss function
        #     loss_fn = build_loss(config.loss, reduction='none')
        #     # Define the optimizer with weight decay and learning rate scheduler
        #     optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
        #
        # # Define the learning rate scheduler
        # scheduler = ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=config.lr_factor,
        #     patience=config.lr_patience,
        #     verbose=True
        # )
        #
        # # Train the model
        # best_val_loss = float('inf')
        # early_stop_counter = 0
        # for epoch in tqdm(range(config.num_epochs + 1)):
        #
        #     if epoch == 0:
        #         # epoch_0_eval(model, train_loader, val_loader, loss_fn,)
        #         continue
        #
        #     # Training
        #     train_loss = train(model, train_loader, optimizer, loss_fn)
        #     # Validation
        #     val_loss, val_rmse, val_r2, val_evs =  evaluate(model, val_loader, loss_fn)
        #     # Log the metrics
        #     wandb.log(
        #         data={
        #             'epoch': epoch,
        #             'train_loss': train_loss,
        #             'val_loss': val_loss,
        #             'val_rmse': val_rmse,
        #             'val_r2': val_r2,
        #             'val_evs': val_evs
        #         }
        #     )
        #
        #     # Update the learning rate
        #
        #
        #
        # # Train the model
        # train_loss = train(
        #     model,
        #     train_loader,
        #
        #     return_model=True
        # )





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
#         # seed = random.randint(0, 10000)
#         # torch.manual_seed(seed)
#         # random.seed(seed)
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

def train(
        model,
        dataloader,
        optimizer,
        loss_fn,
):
    model.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # Create a mask for Nan targets
        nan_mask = torch.isnan(targets)
        # Loss calculation without nan in the mean
        loss = calc_loss_notnan(outputs, targets, nan_mask, loss_fn)
        # loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(dataloader)
    return train_loss


def evaluate(
        model,
        dataloader,
        loss_fn,
):
    model.eval()
    total_loss = 0.0
    targets_all = []
    outputs_all = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Create a mask for Nan targets
            nan_mask = torch.isnan(targets)
            # Loss calculation without nan in the mean
            loss = calc_loss_notnan(outputs, targets, nan_mask, loss_fn)
            total_loss += loss.item()
            targets_all.append(targets)
            outputs_all.append(outputs)
        total_loss /= len(dataloader)
        targets_all = torch.cat(targets_all, dim=0)
        outputs_all = torch.cat(outputs_all, dim=0)

        # Calculate metrics for the ensemble
        ensemble_rmse,ensemble_r2, ensemble_evs = calc_regr_metrics(targets_all, outputs_all)

        # Calculate metrics for each individual model in the ensemble
        model_metrics = []
        for model_output in outputs_all:
            model_rmse, model_r2, model_evs = calc_regr_metrics(targets_all, model_output)
            model_metrics.append((model_rmse, model_r2, model_evs))

        # Calculate uncertainties or variances
        uncertainties = torch.var(outputs_all, dim=0)
    # return total_loss, rmse, r2, evs
    return total_loss, ensemble_rmse, ensemble_r2, ensemble_evs, model_metrics, uncertainties


def ensemble_pipeline(config=wandb.config, wandb_project_name="test-project"):
    with wandb.init(
            dir=LOG_DIR,
            mode=wandb_mode,
            project=wandb_project_name,
            config=config
    ):
        config = wandb.config

        # Load the dataset
        train_loader, val_loader, test_loader = build_loader(config)

        # Load the model
        model = BaselineDNN(
            input_dim=config.input_dim,
            hidden_dim_1=config.hidden_dim_1,
            hidden_dim_2=config.hidden_dim_2,
            hidden_dim_3=config.hidden_dim_3,
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
            val_loss, ensemble_rmse, ensemble_r2, ensemble_evs, model_metrics, uncertainties = evaluate(model, test_loader, loss_fn)
            # val_loss, val_rmse, val_r2, val_evs = evaluate(model, val_loader, loss_fn)
            # Log the metrics
            wandb.log(
                data={
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'ensemble_rmse': ensemble_rmse,
                    'ensemble_r2': ensemble_r2,
                    'ensemble_evs': ensemble_evs,
                    'model_metrics': model_metrics,
                    'uncertainties': uncertainties
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


if __name__ == '__main__':
    # datasets = get_datasets('xc50', 'random')
    test_loss, test_predictions = run_ensemble(
        config=os.path.join(CONFIG_DIR, 'ensemble/ensemble.json'),
        activity='xc50',
        split='random',
        ensemble_size=100,
        ensemble_method='fusion',
        wandb_project_name='mtl-ensemble-test',
        seed=42,
    )

    print(test_loss)
    print(test_predictions)


# class EnsembleDNN(nn.Module):
#     def __init__(self, base_model, num_of_ensemble):
#         super(EnsembleDNN, self).__init__()
#         # TODO: use different random seeds for initialization of each model
#
#         self.models = nn.ModuleList([base_model for _ in range(num_of_ensemble)])
#
#     def forward(self, x):
#         outputs = torch.stack([model(x) for model in self.models])
#         return outputs
#

# def get_config(activity='xc50', split='random'):
#     config = {
#         'activity': activity,
#         'batch_size': 128,
#         'dropout': 0.1,
#         'early_stop': 100,
#         'hidden_dim_1': 2048,
#         'hidden_dim_2': 256,
#         'hidden_dim_3': 256,
#         'input_dim': 2048,
#         'learning_rate': 0.01,
#         'loss': 'huber',
#         'lr_factor': 0.5,
#         'lr_patience': 20,
#         'num_epochs': 3000,  # 20,
#         'num_tasks': 20,
#         'optimizer': 'sgd',
#         'output_dim': 20,
#         'weight_decay': 0.001,
#         'seed': 42,
#         'split': split,
#     }
#
#     return config
#
#
# def get_sweep_config(activity='xc50', split='random'):
#     # # Initialize wandb
#     # wandb.init(project='multitask-learning')
#     # Sweep configuration
#     sweep_config = {
#         'method': 'random',
#         'metric': {
#             'name': 'val_rmse', # 'val_loss',
#             'goal': 'minimize'
#         },
#         'parameters': {
#             'input_dim': {
#                 'values': [1024, 2048]
#             },
#             'hidden_dim_1': {
#                 'values': [512, 1024, 2048]
#             },
#             'hidden_dim_2': {
#                 'values': [256, 512]
#             },
#             'hidden_dim_3': {
#                 'values': [128, 256]
#             },
#             'num_tasks': {
#                 'value': 20
#             },
#             'batch_size': {
#                 'values': [64, 128, 256]
#             },
#             'loss': {
#                 'values': ['huber', 'mse']
#             },
#             'learning_rate': {
#                 'values': [0.001, 0.01]
#             },
#             'ensemble_size': {
#                 'value': 100
#             },
#             'weight_decay': {
#                 'value': 0.001
#             },
#             'dropout': {
#                 'values': [0.1, 0.2]
#             },
#             'lr_factor': {
#                 'value': 0.5
#             },
#             'lr_patience': {
#                 'value': 20
#             },
#             'num_epochs': {
#                 'value': 3000
#             },
#             'early_stop': {
#                 'value': 100
#             },
#             'optimizer': {
#                 'values': ['adamw', 'sgd']
#             },
#             'output_dim': {
#                 'value': 20
#             },
#             'activity': {
#                 'value': "xc50"
#             },
#             'seed': {
#                 'value': 42
#             },
#         },
#     }
#     # 576 combinations
#     return sweep_config
#


