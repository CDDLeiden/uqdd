from papyrus import data_preparation
from models.baselines import run_pipeline

import wandb
    # run_model, hyperparam_search

# preparing the xc50 dataset
# train_data_xc50, val_data_xc50, test_data_xc50, df_xc50 = data_preparation(papyrus_path='data/', activity='xc50', n_top=20, multitask=True, std_smiles=True, output_path='data/dataset/xc50/', verbose_files=True)
#
# # preparing the kx dataset
# train_data_kx, val_data_kx, test_data_kx, df_kx = data_preparation(papyrus_path='data/', activity='kx', n_top=20, multitask=True, std_smiles=True, output_path='data/dataset/kx/', verbose_files=True)

# running config for multitask debugging
test_loss = run_pipeline(wandb_project_name="2023-06-02-mtl-testing", sweep=False)
test_loss = run_pipeline(wandb_project_name="2023-06-02-mtl-testing-hyperparam", sweep=True, sweep_count=100)

print(test_loss)
# running config for multitask debugging
# test_loss = run_pipeline(sweep=False)
#
# # running hyperparam search for multitask debugging
# test_loss = run_pipeline(sweep=True)

# run_model()

# hyperparam_search()
