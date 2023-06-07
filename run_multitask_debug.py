# get today's date as yyyy/mm/dd format
from datetime import date

from models.baselines import run_baseline_hyperparam

today = date.today()
today = today.strftime("%Y%m%d")
# # preparing the xc50 dataset - scaffold split
# train_data_xc50_scaffold, val_data_xc50_scaffold, test_data_xc50_scaffold, df_xc50_scaffold = data_preparation(papyrus_path='data/', activity='xc50', organism='Homo sapiens (Human)', n_top=20, multitask=True, std_smiles=True, split_type='scaffold', output_path='data/dataset/xc50/scaffold/', verbose_files=True)
# #
# # # preparing the xc50 dataset
# train_data_xc50, val_data_xc50, test_data_xc50, df_xc50 = data_preparation(papyrus_path='data/', activity='xc50', organism='Homo sapiens (Human)', n_top=20, multitask=True, std_smiles=True, split_type='random', output_path='data/dataset/xc50/random/', verbose_files=True)
# #
# # # preparing the kx dataset
# train_data_kx, val_data_kx, test_data_kx, df_kx = data_preparation(papyrus_path='data/', activity='kx', organism='Homo sapiens (Human)', n_top=20, multitask=True, std_smiles=True, split_type='random', output_path='data/dataset/kx/random/', verbose_files=True)
#
# # preparing the kx dataset
# train_data_kx_scaffold, val_data_kx_scaffold, test_data_kx_scaffold, df_kx_scaffold = data_preparation(papyrus_path='data/', activity='kx', organism='Homo sapiens (Human)', n_top=20, multitask=True, std_smiles=True, split_type='scaffold', output_path='data/dataset/kx/scaffold/', verbose_files=True)


failed_config = {
    'activity': "xc50",
    'batch_size': 128,
    'dropout': 0.1,
    'early_stop': 100,
    'hidden_dim_1': 2048,
    'hidden_dim_2': 512,
    'hidden_dim_3': 256,
    'input_dim': 2048,
    'learning_rate': 0.001,
    'loss': 'huber',
    'lr_factor': 0.5,
    'lr_patience': 20,
    'num_epochs': 3000,  # 20,
    'num_tasks': 20,
    'optimizer': 'sgd',
    'output_dim': 20,
    'weight_decay': 0.001,  # 1e-5,
    'split': "random"
}
# running 10 epochs for multitask debugging
# run_baseline(config=failed_config, activity='xc50', split='random', wandb_project_name=f'2023-06-02-mtl-testing')

# running sweep for multitask
run_baseline_hyperparam(
    activity='xc50',
    split='random',
    wandb_project_name=f'2023-06-02-mtl-testing-hyperparam',
    sweep_count=10
)


