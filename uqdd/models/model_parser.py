import argparse

from uqdd.models.baseline import run_baseline_wrapper, run_baseline_hyperparam
from uqdd.models.ensemble import run_ensemble_wrapper, run_ensemble_hyperparm
from uqdd.models.mcdropout import run_mcdropout_wrapper, run_mcdropout_hyperparm
from uqdd.models.evidential import run_evidential_wrapper  # , run_evidential_hyperparam
from uqdd.utils import float_or_none, parse_list

query_dict = {
    "baseline": run_baseline_wrapper,
    "baseline_hyperparam": run_baseline_hyperparam,
    "ensemble": run_ensemble_wrapper,
    "ensemble_hyperparam": run_ensemble_hyperparm,
    "mcdropout": run_mcdropout_wrapper,
    "mcdropout_hyperparam": run_mcdropout_hyperparm,
    "evidential": run_evidential_wrapper,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model parser")

    # * DATA arguments * #
    parser.add_argument(
        "--data_name",
        type=str,
        default="papyrus",
        choices=["papyrus", "tdc", "other"],
        help="Data name argument",
    )
    parser.add_argument(
        "--activity_type",
        type=str,
        default="xc50",
        choices=["xc50", "kx"],
        help="Activity argument",
    )
    parser.add_argument(
        "--n_targets",
        type=int,
        default=-1,
        help="Number of targets argument (default=-1 for all targets)",
    )
    parser.add_argument(
        "--median_scaling",
        action="store_true",
        help="Use median scaling",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type argument",
    )

    # * SPLIT arguments * #
    parser.add_argument(
        "--split_type",
        type=str,
        default="random",
        choices=["random", "scaffold", "time", "scaffold_cluster"],
        help="Split argument",
    )

    parser.add_argument(
        "--ext",
        type=str,
        default="pkl",
        choices=["pkl", "parquet", "csv", "feather"],
        help="File extension argument",
    )

    # * Features args * #
    parser.add_argument(
        "--descriptor_protein",
        type=str,
        default="ankh-large",
        choices=[
            None,
            "ankh-base",
            "ankh-large",
            "unirep",
            "protbert",
            "protbert_bfd",
            "esm1_t34",
            "esm1_t12",
            "esm1_t6",
            "esm1b",
            "esm_msa1",
            "esm_msa1b",
            "esm1v",
        ],
        help="Protein descriptor argument",
    )
    parser.add_argument(
        "--descriptor_chemical",
        type=str,
        default="ecfp2048",
        choices=[
            "ecfp1024",
            "ecfp2048",
            "mold2",
            "mordred",
            "cddd",
            "fingerprint",  # "moldesc"
        ],
        help="Chemical descriptor argument",
    )

    # * Model args * #
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "ensemble", "mcdropout", "evidential"],
        help="Model name argument",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of sequential repeats of runs",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="model-test",
        help="Wandb project name argument",
    )
    parser.add_argument(
        "--sweep-count",
        type=int,
        default=None,
        help="Sweep count argument",
    )

    parser.add_argument(
        "--chem_layers",
        type=parse_list,
        default=None,
        help="Chem layers sizes",
    )  #  nargs="+",
    parser.add_argument(
        "--prot_layers", type=parse_list, default=None, help="Prot layers sizes"
    )
    parser.add_argument(
        "--regressor_layers",
        # nargs="+",
        type=parse_list,
        default=None,
        help="Regressor layers sizes",
    )
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument(
        "--early_stop", type=int, default=None, help="Early stopping patience"
    )
    parser.add_argument("--loss", type=str, default=None, help="Loss function")
    parser.add_argument(
        "--loss_reduction", type=str, default=None, help="Loss reduction method"
    )
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer")
    parser.add_argument("--lr", type=float_or_none, default=None, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float_or_none, default=None, help="Weight decay rate"
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default=None, help="LR scheduler type"
    )
    parser.add_argument(
        "--lr_scheduler_patience", type=int, default=None, help="LR scheduler patience"
    )
    parser.add_argument(
        "--lr_scheduler_factor",
        type=float_or_none,
        default=None,
        help="LR scheduler factor",
    )
    parser.add_argument(
        "--max_norm",
        type=float_or_none,
        default=50.0,
        help="Max norm for gradient clipping",  # , choices=[None, 10.0, 50.0]
    )

    # * uncertainty args * #
    parser.add_argument(
        "--aleatoric", type=bool, default=True, help="Aleatoric inference"
    )

    # * Ensemble args * #
    # parser.add_argument(
    #     "--parallelize",
    #     type=bool,
    #     default=False,
    #     help="Parallelize training"
    # )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=100,
        help="Size of the ensemble",
    )

    # * MCDropout args * #
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=100,
        help="Number of MC dropout samples",
    )

    # * Evidential args * #

    parser.add_argument("--tags", type=str, default=None, help="Extra Tags for wandb")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()
    # Construct kwargs, excluding arguments that were not provided
    kwargs = {k: v for k, v in vars(args).items() if v is not None}

    sweep_count = args.sweep_count
    repeats = args.repeats
    seed = args.seed

    if sweep_count is not None:
        if args.model in ["baseline", "ensemble", "mcdropout"]:
            query_dict[f"{args.model}_hyperparam"](**kwargs)
        else:
            print(
                "Sweep count only supported for baseline, ensemble, and mcdropout models."
            )
    else:
        if repeats > 1:
            for i in range(repeats):
                kwargs["seed"] = seed
                print(f"Run {i+1}/{repeats}")
                query_dict[args.model](**kwargs)
                seed += 1 if args.model != "ensemble" else int(args.ensemble_size)

        else:
            query_dict[args.model](**kwargs)
