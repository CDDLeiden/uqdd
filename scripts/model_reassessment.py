import argparse
import os

from uqdd.metrics import (
    preprocess_runs,
    reassess_metrics,
)
from uqdd.utils import create_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input parameters for the script.")
    parser.add_argument("--activity_type", type=str, required=True, help="The type of activity (e.g., 'kx', 'xc50').")
    parser.add_argument("--runs_file_name", type=str, required=True, help="The name of the runs file (without path).")
    parser.add_argument("--project_name", type=str, required=True, help="The name of the project.")
    args = parser.parse_args()

    activity_type = args.activity_type
    project_name = args.project_name
    runs_file_name = args.runs_file_name

    data_name = "papyrus"
    type_n_targets = "all"
    project_out_name = f"reassess-{project_name}"
    data_specific_path = f"{data_name}/{activity_type}/{type_n_targets}"
    descriptor_protein = "ankh-large"
    descriptor_chemical = "ecfp2048"
    prot_input_dim = 1536
    chem_input_dim = 2048

    # Build paths relative to repo
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, "../uqdd", "data")

    preds_dirpath = os.path.join(data_dir, "predictions", data_specific_path)
    models_dir = os.path.join(data_dir, "../..", "models", "saved_models", data_specific_path)
    runs_path = os.path.join(data_dir, "runs", runs_file_name)

    figs_out_path = os.path.join(repo_root, "../uqdd", "figures", data_specific_path, project_out_name)
    os.makedirs(figs_out_path, exist_ok=True)
    csv_out_path = os.path.join(figs_out_path, "metrics.csv")

    if not os.path.exists(runs_path):
        raise FileNotFoundError(f"File {runs_path} not found")

    logger = create_logger(name="reassess", file_level="debug", stream_level="debug")

    runs_df = preprocess_runs(
        runs_path,
        models_dir=models_dir,
        data_name=data_name,
        activity_type=activity_type,
        descriptor_protein=descriptor_protein,
        descriptor_chemical=descriptor_chemical,
        data_specific_path=data_specific_path,
        prot_input_dim=prot_input_dim,
        chem_input_dim=chem_input_dim,
    )
    print("Runs_df preprocessed")

    reassess_metrics(runs_df, figs_out_path, csv_out_path, project_out_name, logger)
