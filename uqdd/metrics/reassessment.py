import os
import glob
import ast
import pandas as pd

from uqdd import DEVICE, MODELS_DIR
from uqdd.models.emc import emc_predict, emc_nll
from uqdd.models.ensemble import EnsembleDNN
from uqdd.models.eoe import EoEDNN
from uqdd.models.evidential import EvidentialDNN, ev_predict, ev_nll
from uqdd.models.mcdropout import mc_predict
from uqdd.models.pnn import PNN
from uqdd.models.utils_metrics import process_preds, create_df_preds
from uqdd.models.utils_models import load_model, get_model_config
from uqdd.models.utils_train import predict, evaluate_predictions, recalibrate_model, get_dataloader


def nll_evidentials(evidential_model, test_dataloader, model_type="evidential", num_mc_samples=100, device=DEVICE):
    if model_type in ["evidential", "eoe"]:
        return ev_nll(evidential_model, test_dataloader, device=device)
    elif model_type == "emc":
        return emc_nll(evidential_model, test_dataloader, num_mc_samples=num_mc_samples, device=device)
    else:
        return None


def convert_to_list(val):
    if isinstance(val, str):
        try:
            parsed_val = ast.literal_eval(val)
            if isinstance(parsed_val, list):
                return parsed_val
            else:
                return []
        except (SyntaxError, ValueError):
            print(f"Warning: Unable to parse value {val}, returning empty list.")
            return []
    return val


def preprocess_runs(runs_path, models_dir=MODELS_DIR, data_name="papyrus", activity_type="xc50", descriptor_protein="ankh-large", descriptor_chemical="ecfp2048", data_specific_path="papyrus/xc50/all", prot_input_dim=1536, chem_input_dim=2048):
    runs_df = pd.read_csv(runs_path, converters={"chem_layers": convert_to_list, "prot_layers": convert_to_list, "regressor_layers": convert_to_list})
    runs_df.rename(columns={"Name": "run_name"}, inplace=True)
    i = 1
    for index, row in runs_df.iterrows():
        model_name = row["model_name"] if not pd.isna(row["model_name"]) else row["run_name"]
        model_file_pattern = os.path.join(models_dir, f"*{model_name}.pt")
        model_files = glob.glob(model_file_pattern)
        if model_files:
            model_file_path = model_files[0]
            model_name = os.path.basename(model_file_path).replace(".pt", "")
            runs_df.at[index, "model_name"] = model_name
            runs_df.at[index, "model_path"] = model_file_path
        else:
            print(f"{i} Model file(s) not found for {model_name} \n with pattern {model_file_pattern}")
            runs_df.at[index, "model_path"] = ""
            i += 1
    runs_df["data_name"] = data_name
    runs_df["activity_type"] = activity_type
    runs_df["descriptor_protein"] = descriptor_protein
    runs_df["descriptor_chemical"] = descriptor_chemical
    runs_df["chem_input_dim"] = chem_input_dim
    runs_df["prot_input_dim"] = prot_input_dim
    runs_df["data_specific_path"] = data_specific_path
    runs_df["MT"] = runs_df["n_targets"].apply(lambda x: True if x > 1 else False)
    return runs_df


def get_model_class(model_type: str):
    if model_type.lower() in ["pnn", "mcdropout"]:
        return PNN
    elif model_type.lower() == "ensemble":
        return EnsembleDNN
    elif model_type.lower() in ["evidential", "emc"]:
        return EvidentialDNN
    elif model_type.lower() == "eoe":
        return EoEDNN
    else:
        raise ValueError(f"Model type {model_type} not recognized")


def get_predict_fn(model_type: str, num_mc_samples=100):
    if model_type.lower() == "mcdropout":
        return mc_predict, {"num_mc_samples": num_mc_samples}
    elif model_type.lower() in ["ensemble", "pnn"]:
        return predict, {}
    elif model_type.lower() in ["evidential", "eoe"]:
        return ev_predict, {}
    elif model_type.lower() == "emc":
        return emc_predict, {"num_mc_samples": num_mc_samples}
    else:
        raise ValueError(f"Model type {model_type} not recognized")


def get_preds(model, dataloaders, model_type, subset="test", num_mc_samples=100):
    predict_fn, predict_kwargs = get_predict_fn(model_type, num_mc_samples=num_mc_samples)
    preds_res = predict_fn(model, dataloaders[subset], device=DEVICE, **predict_kwargs)
    if model_type in ["evidential", "eoe", "emc"]:
        preds, labels, alea_vars, epi_vars = preds_res
    else:
        preds, labels, alea_vars = preds_res
        epi_vars = None
    return preds, labels, alea_vars, epi_vars


def pkl_preds_export(preds, labels, alea_vars, epi_vars, outpath, model_type, logger=None):
    y_true, y_pred, y_err, y_alea, y_eps = process_preds(preds, labels, alea_vars, epi_vars, None, model_type)
    df = create_df_preds(y_true=y_true, y_pred=y_pred, y_err=y_err, y_alea=y_alea, y_eps=y_eps, export=False, logger=logger)
    df.to_pickle(os.path.join(outpath, "preds.pkl"))
    return df


def csv_nll_post_processing(csv_path):
    df = pd.read_csv(csv_path)
    df["NLL"] = df.groupby("model name")["NLL"].transform("first")
    df.to_csv(csv_path, index=False)


def reassess_metrics(runs_df, figs_out_path, csv_out_path, project_out_name, logger):
    runs_df = runs_df.sample(frac=1).reset_index(drop=True)
    for index, row in runs_df.iterrows():
        model_path = row["model_path"]
        model_name = row["model_name"]
        run_name = row["run_name"]
        rowkwargs = row.to_dict()
        model_type = rowkwargs.pop("model_type")
        activity_type = rowkwargs.pop("activity_type")
        if model_path:
            model_fig_out_path = os.path.join(figs_out_path, model_name)
            if os.path.exists(model_fig_out_path):
                print(f"Model {model_name} already reassessed")
                continue
            os.makedirs(model_fig_out_path, exist_ok=True)
            config = get_model_config(model_type=model_type, activity_type=activity_type, **rowkwargs)
            num_mc_samples = config.get("num_mc_samples", 100)
            model_class = get_model_class(model_type)
            prefix = "models." if model_type == "eoe" else ""
            model = load_model(model_class, model_path, prefix_to_state_keys=prefix, config=config).to(DEVICE)
            dataloaders = get_dataloader(config, device=DEVICE, logger=logger)
            preds, labels, alea_vars, epi_vars = get_preds(model, dataloaders, model_type, subset="test", num_mc_samples=num_mc_samples)
            nll = nll_evidentials(model, dataloaders["test"], model_type=model_type, num_mc_samples=num_mc_samples, device=DEVICE)
            df = pkl_preds_export(preds, labels, alea_vars, epi_vars, model_fig_out_path, model_type, logger=logger)
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
                project_name=project_out_name,
                figpath=model_fig_out_path,
                export_preds=False,
                verbose=False,
                csv_path=csv_out_path,
                nll=nll,
            )
            preds_val, labels_val, alea_vars_val, epi_vars_val = get_preds(model, dataloaders, model_type, subset="val", num_mc_samples=num_mc_samples)
            nll = nll_evidentials(model, dataloaders["val"], model_type=model_type, num_mc_samples=num_mc_samples, device=DEVICE)
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
                nll=nll,
            )
            uct_logger.csv_log()

