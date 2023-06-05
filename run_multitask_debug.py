from papyrus import data_preparation
from models.baselines import run_baseline, run_baseline_hyperparam

import wandb

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


# running 10 epochs for multitask debugging
test_loss, test_rmse, test_r2, test_evs = run_baseline()

# running sweep for multitask
run_baseline_hyperparam(sweep_count=100)

# running config for multitask debugging
# test_loss = run_pipeline(sweep=False)
#
# # running hyperparam search for multitask debugging
# test_loss = run_pipeline(sweep=True)

# run_model()

# hyperparam_search()
