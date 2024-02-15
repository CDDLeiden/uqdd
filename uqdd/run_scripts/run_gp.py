from datetime import date, datetime
import argparse
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from uqdd.models.papyrus import data_preparation
from uqdd.models.gp_gauche import run_gp_model

DATA_DIR = os.environ.get("DATA_DIR")
CONFIG_DIR = os.environ.get("CONFIG_DIR")


def main():
    parser = argparse.ArgumentParser(description="GP Gauche Model Running")
    parser.add_argument(
        "--activity",
        type=str,
        default="xc50",
        choices=["xc50", "kx"],
        help="Activity argument",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="random",
        choices=["random", "scaffold"],
        help="Split argument",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="gp-gauche-test",
        help="Wandb project name argument",
    )
    parser.add_argument(
        "--prepare-dataset", action="store_true", help="Flag Prepare dataset"
    )

    args = parser.parse_args()

    today = date.today()
    today = today.strftime("%Y%m%d")

    start_time = datetime.now()

    if args.prepare_dataset:
        output_path = os.path.join(DATA_DIR, "dataset", args.activity, args.split)
        _, _, _, _ = data_preparation(
            papyrus_path=DATA_DIR,
            activity=args.activity,
            organism="Homo sapiens (Human)",
            n_top=20,
            multitask=True,
            std_smiles=True,
            split_type=args.split,
            output_path=output_path,
            verbose_files=True,
        )

    run_gp_model(
        config=os.path.join(CONFIG_DIR, "gp", "gp.json"),
    )
    # parser = argparse.ArgumentParser(description='MC Dropout Model Running')
    # parser.add_argument('--activity', type=str, default='xc50', choices=['xc50', 'kx'], help='Activity argument')
    # parser.add_argument('--split', type=str, default='random', choices=['random', 'scaffold'], help='Split argument')
    # parser.add_argument('--num-samples', type=int, default=100, help='Number of samples for MC Dropout')
    # parser.add_argument('--wandb-project-name', type=str, default='mcdropout-test', help='Wandb project name argument')
    # parser.add_argument('--prepare-dataset', action='store_true', help='Flag Prepare dataset')
    #
    # args = parser.parse_args()
    #
    # today = date.today()
    # today = today.strftime("%Y%m%d")
    #
    # start_time = datetime.now()
    #
    # if args.prepare_dataset:
    #     output_path = os.path.join(DATA_DIR, 'dataset', args.activity, args.split)
    #     _, _, _, _ = data_preparation(
    #         papyrus_path=DATA_DIR,
    #         activity=args.activity,
    #         organism='Homo sapiens (Human)',
    #         n_top=20,
    #         multitask=True,
    #         std_smiles=True,
    #         split_type=args.split,
    #         output_path=output_path,
    #         verbose_files=True
    #     )
    #
    # run_mcdropout(
    #     config=os.path.join(CONFIG_DIR, 'baseline', 'baseline_xc50_random_best.json'),
    #     activity=args.activity,
    #     split=args.split,
    #     wandb_project_name=f'{args.wandb_project_name}',
    #     num_samples=args.num_samples,
    # )
    #
    # end_time = datetime.now()
    # duration = end_time - start_time
    # print(f"Script execution time: {duration}")


if __name__ == "__main__":
    main()
