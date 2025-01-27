import os
import pandas as pd
import argparse

import glob
from uqdd import DEVICE, MODELS_DIR
from uqdd.models.emc import emc_predict

# from uqdd.utils import get_config
from uqdd.utils import load_df, create_logger
from uqdd.models.utils_models import load_model, get_model_config, calculate_means
from uqdd.models.baseline import BaselineDNN
from uqdd.models.ensemble import EnsembleDNN
from uqdd.models.mcdropout import mc_predict
from uqdd.models.evidential import EvidentialDNN, ev_predict
from uqdd.models.eoe import EoEDNN

# METRICS
from uqdd.models.utils_train import (
    predict,
    evaluate_predictions,
    recalibrate_model,
    get_dataloader,
)
from uqdd.models.utils_metrics import process_preds, create_df_preds

# # Importing models and their predict functions
# from uqdd.models.utils_train import predict # for ensemble
# import matplotlib.pyplot as plt
import ast


# Function to convert string representation of list to actual list of integers
def convert_to_list(val):
    if isinstance(val, str):
        return ast.literal_eval(val)
    return val


# RUNS preprocessing
def preprocess_runs(
    runs_path,
    models_dir=MODELS_DIR,
    data_name="papyrus",
    activity_type="xc50",
    descriptor_protein="ankh-large",
    descriptor_chemical="ecfp2048",
    data_specific_path="papyrus/xc50/all",
    prot_input_dim=1536,
    chem_input_dim=2048,
):
    # Load csv runs and model names file
    runs_df = load_df(
        runs_path,
        converters={
            "chem_layers": convert_to_list,
            "prot_layers": convert_to_list,
            "regressor_layers": convert_to_list,
        },
    )
    runs_df.rename(columns={"Name": "run_name"}, inplace=True)

    # DEALING WITH MODEL_NAME IF DOESN'T EXIST
    # runs_df["model_name"] = runs_df.apply(
    #     lambda row: (
    #         f"{data_name}_{activity_type}_{row['model_type']}_{row['split_type']}_{descriptor_protein}_{descriptor_chemical}_{row['run_name']}"
    #         if pd.isna(row["model_name"])
    #         else row["model_name"]
    #     ),
    #     axis=1,
    # )

    i = 1
    # Update and match model_name with model saved files
    for index, row in runs_df.iterrows():
        model_name = (
            row["model_name"] if not pd.isna(row["model_name"]) else row["run_name"]
        )
        model_file_pattern = os.path.join(models_dir, f"*{model_name}.pt")
        model_files = glob.glob(model_file_pattern)
        if model_files:
            model_file_path = model_files[0]
            model_name = os.path.basename(model_file_path).replace(".pt", "")
            # print(model_name)
            runs_df.at[index, "model_name"] = model_name
            # add model_file_path to the runs_df
            runs_df.at[index, "model_path"] = model_file_path
        else:
            print(
                f"{i} Model file(s) not found for {model_name} \n with pattern {model_file_pattern}"
            )
            runs_df.at[index, "model_path"] = ""
            i += 1

    # Ensure rest of variables set correctly across the runs
    runs_df["data_name"] = data_name
    runs_df["activity_type"] = activity_type
    runs_df["descriptor_protein"] = descriptor_protein
    runs_df["descriptor_chemical"] = descriptor_chemical
    runs_df["chem_input_dim"] = chem_input_dim
    runs_df["prot_input_dim"] = prot_input_dim
    runs_df["data_specific_path"] = data_specific_path
    runs_df["MT"] = runs_df["n_targets"].apply(lambda x: True if x > 1 else False)

    return runs_df


# Get model class and predict function


def get_model_class(model_type: str):
    if model_type.lower() in ["baseline", "mcdropout"]:
        model_class = BaselineDNN
    elif model_type.lower() == "ensemble":
        model_class = EnsembleDNN
    elif model_type.lower() in ["evidential", "emc"]:
        model_class = EvidentialDNN
    elif model_type.lower() == "eoe":
        model_class = EoEDNN
    else:
        raise ValueError(f"Model type {model_type} not recognized")
    return model_class


def get_predict_fn(model_type: str, num_mc_samples=100):
    if model_type.lower() == "mcdropout":
        predict_fn = mc_predict
        predict_kwargs = {"num_mc_samples": num_mc_samples}
    elif model_type.lower() in ["ensemble", "baseline"]:
        predict_fn = predict
        predict_kwargs = {}
    elif model_type.lower() in ["evidential", "eoe"]:
        predict_fn = ev_predict
        predict_kwargs = {}
    elif model_type.lower() == "emc":
        predict_fn = emc_predict
        predict_kwargs = {"num_mc_samples": num_mc_samples}
    else:
        raise ValueError(f"Model type {model_type} not recognized")
    return predict_fn, predict_kwargs


def get_preds(model, dataloaders, model_type, subset="test", num_mc_samples=100):
    predict_fn, predict_kwargs = get_predict_fn(
        model_type, num_mc_samples=num_mc_samples
    )
    preds_res = predict_fn(model, dataloaders[subset], device=DEVICE, **predict_kwargs)
    if model_type in ["evidential", "eoe", "emc"]:
        preds, labels, alea_vars, epi_vars = preds_res
    else:
        preds, labels, alea_vars = preds_res
        epi_vars = None
    if model_type in ["eoe", "emc"]:
        preds, alea_vars, epi_vars = calculate_means(preds, alea_vars, epi_vars)
    return preds, labels, alea_vars, epi_vars


def pkl_preds_export(preds, labels, alea_vars, epi_vars, outpath):
    # preds, labels = predict(model, dataloaders["test"], return_targets=True)
    y_true, y_pred, y_eps, y_err, y_alea = process_preds(
        preds, labels, alea_vars, epi_vars, None
    )
    df = create_df_preds(
        y_true,
        y_pred,
        y_alea,  # y_std,
        y_err,
        y_eps,  # y_alea,
        export=False,
        logger=logger,
    )

    df.to_pickle(os.path.join(outpath, "preds.pkl"))
    return df


def reassess_metrics(
    runs_df,
    figs_out_path,
    csv_out_path,
    project_out_name,
    logger,
):
    # Shuffle rows of the runs_df
    runs_df = runs_df.sample(frac=1).reset_index(drop=True)

    # Reassessing metrics and recalibration of the pretrained models
    for index, row in runs_df.iterrows():
        model_path = row["model_path"]
        model_name = row["model_name"]
        # activity_type = row["activity_type"]
        run_name = row["run_name"]
        if run_name.startswith("vivid"):
            pass
        # print(type(model_path))
        rowkwargs = row.to_dict()
        # popping the model_type
        model_type = rowkwargs.pop("model_type")
        activity_type = rowkwargs.pop("activity_type")

        if model_path:
            model_fig_out_path = os.path.join(figs_out_path, model_name)
            if os.path.exists(model_fig_out_path):
                print(f"Model {model_name} already reassessed")
                continue
            # make the model_fig_out_path dir
            os.makedirs(model_fig_out_path, exist_ok=True)

            config = get_model_config(
                model_type=model_type, activity_type=activity_type, **rowkwargs
            )
            num_mc_samples = config.get("num_mc_samples", 100)
            model_class = get_model_class(model_type)
            prefix = "models." if model_type == "eoe" else ""
            model = load_model(
                model_class, model_path, prefix_to_state_keys=prefix, config=config
            ).to(DEVICE)

            # Getting DataLoaders
            dataloaders = get_dataloader(config, device=DEVICE, logger=logger)

            # RePredict and Evaluate preds
            preds, labels, alea_vars, epi_vars = get_preds(
                model,
                dataloaders,
                model_type,
                subset="test",
                num_mc_samples=num_mc_samples,
            )
            df = pkl_preds_export(
                preds, labels, alea_vars, epi_vars, model_fig_out_path
            )

            # Calculate the metrics
            metrics, plots, uct_logger = evaluate_predictions(
                config,
                preds,
                labels,
                alea_vars,
                model_type,
                logger,
                epi_vars=epi_vars,
                wandb_push=False,
                run_name=config["run_name"],
                project_name=project_out_name,  # for the csv file
                figpath=model_fig_out_path,
                export_preds=False,
                verbose=False,
                csv_path=csv_out_path,
            )

            # Recalibrate model
            preds_val, labels_val, alea_vars_val, epi_vars_val = get_preds(
                model,
                dataloaders,
                model_type,
                subset="val",
                num_mc_samples=num_mc_samples,
            )

            iso_recal_model = recalibrate_model(
                preds_val,
                labels_val,
                alea_vars_val,
                preds,
                labels,
                alea_vars,
                config=config,
                epi_val=epi_vars_val,
                epi_test=epi_vars,
                uct_logger=uct_logger,
                figpath=model_fig_out_path,
            )

            # Log the metrics to the CSV file
            uct_logger.csv_log()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process input parameters for the script."
    )
    parser.add_argument(
        "--activity_type",
        type=str,
        required=True,
        help="The type of activity (e.g., 'kx', 'xc50').",
    )
    parser.add_argument(
        "--runs_file_name",
        type=str,
        required=True,
        help="The name of the runs file (without path).",
    )
    parser.add_argument(
        "--project_name", type=str, required=True, help="The name of the project."
    )

    args = parser.parse_args()

    activity_type = args.activity_type
    project_name = args.project_name
    runs_file_name = args.runs_file_name
    ### Testing
    # activity_type = "xc50"
    # project_name = "runs_evidential_xc50"
    # project_name = "runs_ensemble_mcdp_xc50"
    # runs_file_name = "runs_evidential_xc50.csv"
    # runs_file_name = "runs_ensemble_mcdp_xc50.csv"

    data_name = "papyrus"
    type_n_targets = "all"
    project_out_name = f"reassess-{project_name}"
    data_specific_path = f"{data_name}/{activity_type}/{type_n_targets}"
    # DESCRIPTORS
    descriptor_protein = "ankh-large"
    descriptor_chemical = "ecfp2048"
    prot_input_dim = 1536
    chem_input_dim = 2048

    # PATHS
    preds_dirpath = (
        f"/users/home/bkhalil/Repos/uqdd/uqdd/data/predictions/{data_specific_path}/"
    )
    # models_dir = (
    #     f"/users/home/bkhalil/Repos/uqdd/uqdd/models/saved_models/{data_specific_path}/"
    # )
    models_dir = f"/projects/system/bkhalil/BACKUPS_DO_NOT_DELETE/uqdd/models/saved_models/{data_specific_path}/"
    runs_path = f"/users/home/bkhalil/Repos/uqdd/uqdd/data/runs/{runs_file_name}"

    # runs_path = (
    #     f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/{data_specific_path}/kx-all.csv"
    # )

    # FIGS OUT PATH
    figs_out_path = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/{data_specific_path}/{project_out_name}/"

    csv_out_path = figs_out_path + "metrics.csv"
    # csv_out_path = figs_out_path + "evidential-metrics-reassess-ensemble-time.csv"
    # check if the runs_path exists
    if not os.path.exists(runs_path):
        raise FileNotFoundError(f"File {runs_path} not found")

    # create figs_out_path if it does not exist
    os.makedirs(figs_out_path, exist_ok=True)

    logger = create_logger(
        name="reassess",
        file_level="debug",
        stream_level="debug",
    )

    # Load runs_df and preprocess it
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
    #
    # # Reassessing metrics and recalibration of the pretrained models
    # for index, row in runs_df.iterrows():
    #     model_path = row["model_path"]
    #     model_name = row["model_name"]
    #
    #     # print(type(model_path))
    #     rowkwargs = row.to_dict()
    #     # popping the model_type
    #     model_type = rowkwargs.pop("model_type")
    #
    #     if model_path:
    #         model_fig_out_path = os.path.join(figs_out_path, model_name)
    #         config = get_model_config(model_type=model_type, **rowkwargs)
    #         model_class = get_model_class(model_type)
    #         model = load_model(model_class, model_path, config=config).to(DEVICE)
    #
    #         # Getting DataLoaders
    #         dataloaders = get_dataloader(config, device=DEVICE, logger=logger)
    #
    #         # RePredict and Evaluate preds
    #         preds, labels, alea_vars, epi_vars = get_preds(
    #             model, dataloaders, model_type, subset="test"
    #         )
    #
    #         # Calculate the metrics
    #         metrics, plots, uct_logger = evaluate_predictions(
    #             config,
    #             preds,
    #             labels,
    #             alea_vars,
    #             model_type,
    #             logger,
    #             epi_vars=epi_vars,
    #             wandb_push=False,
    #             run_name=config["run_name"],
    #             project_name=project_out_name,  # for the csv file
    #             figpath=model_fig_out_path,
    #             export_preds=False,
    #             verbose=False,
    #             csv_path=csv_out_path,
    #         )
    #
    #         # Recalibrate model
    #         preds_val, labels_val, alea_vars_val, epi_vars_val = get_preds(
    #             model, dataloaders, model_type, subset="val"
    #         )
    #
    #         iso_recal_model = recalibrate_model(
    #             preds_val,
    #             labels_val,
    #             alea_vars_val,
    #             preds,
    #             labels,
    #             alea_vars,
    #             config=config,
    #             epi_val=epi_vars_val,
    #             epi_test=epi_vars,
    #             uct_logger=uct_logger,
    #             figpath=model_fig_out_path,
    #         )
    #
    #         # Log the metrics to the CSV file
    #         uct_logger.csv_log()
    #         # break
    # activity_type = "xc50"
    # activity_type = "kx"

    # # project_name = "2024-06-25-all-models-100"
    # # project_name = "evidential-models"
    # # project_name = "kx-all"
    # # activity_type = "kx"
    # # project_name = "2024-07-22-all-models-100-kx"
    # project_name = f""

    # runs_path = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/{data_specific_path}/{project_name}/runs.csv"
    # runs_path = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/{data_specific_path}/evidential-runs.csv"
    # runs_path = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/{data_specific_path}/emc-eoe-runs.csv"
    # runs_path = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/{data_specific_path}/ensemble-time-runs.csv"
