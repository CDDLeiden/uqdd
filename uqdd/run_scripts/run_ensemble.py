from datetime import date, datetime
import argparse
import os
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from uqdd.models.papyrus import data_preparation
from uqdd.models.ensemble import run_ensemble


DATA_DIR = os.environ.get('DATA_DIR')
CONFIG_DIR = os.environ.get('CONFIG_DIR')

def main():
    parser = argparse.ArgumentParser(description='Ensemble Model Running')
    parser.add_argument('--activity', type=str, default='xc50', choices=['xc50', 'kx'], help='Activity argument')
    parser.add_argument('--split', type=str, default='random', choices=['random', 'scaffold'], help='Split argument')
    parser.add_argument('--wandb-project-name', type=str, default='ensemble-test', help='Wandb project name argument')
    parser.add_argument('--ensemble-size', type=int, default=5, help='Ensemble size argument')
    # parser.add_argument('--config', type=str, default='config/ensemble.json', help='Config argument')
    # parser.add_argument('--ensemble-method', type=str, default='bagging', choices=['bagging', 'boosting'], help='Ensemble method argument')
    # parser.add_argument('--sweep-count', type=int, default=250, help='Sweep count argument')
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument('--baseline', action='store_true', help='Run baseline')
    # group.add_argument('--hyperparam', action='store_true', help='Run hyperparameter search')

    parser.add_argument('--prepare-dataset', action='store_true', help='Flag Prepare dataset')
    args = parser.parse_args()

    today = date.today()
    today = today.strftime("%Y%m%d")

    start_time = datetime.now()

    if args.prepare_dataset:
        output_path = os.path.join(DATA_DIR, 'dataset', args.activity, args.split)
            # f'data/dataset/{args.activity}/{args.split}/'
        _, _, _, _ = data_preparation(
            papyrus_path=DATA_DIR,
            activity=args.activity,
            organism='Homo sapiens (Human)',
            n_top=20,
            multitask=True,
            std_smiles=True,
            split_type=args.split,
            output_path=output_path,
            verbose_files=True
        )

    config_path = os.path.join(CONFIG_DIR, 'baseline', f'baseline_{args.activity}_{args.split}_best.json')

    run_ensemble(
        config=config_path,
        activity=args.activity,
        split=args.split,
        wandb_project_name=args.wandb_project_name,
        ensemble_size=args.ensemble_size,
    )

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Script execution time: {duration}")


if __name__ == '__main__':
    main()


