import argparse
import wandb

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from uqdd import DEVICE
from uqdd.models.baseline import BaselineDNN
from uqdd.utils import create_logger, parse_list

from uqdd.models.utils_train import (
    train_model_e2e,
    evaluate_predictions,
    recalibrate_model
)

from uqdd.models.utils_models import (
    get_model_config,
    get_sweep_config,
)


def ev_predict(
        model,
        test_loader,
        device=DEVICE
):
    model.eval()
    outputs_all, targets_all = [], []
    alea_all, epistemic_all = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, total=len(test_loader), desc="Evidential prediction"):
            inputs = tuple(x.to(device) for x in inputs)
            outputs, alea_vars = model(inputs)
            mu, v, alpha, beta = outputs #(d.squeeze() for d in outputs)
            # alea_vars = beta / (alpha - 1)  # aleatoric
            epist_var = torch.sqrt(beta / (v * (alpha - 1)))  # epistemic
            outputs = mu

            outputs_all.append(outputs)
            targets_all.append(targets)
            alea_all.append(alea_vars)
            epistemic_all.append(epist_var)

    outputs_all = torch.cat(outputs_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    alea_all = torch.cat(alea_all, dim=0)
    epistemic_all = torch.cat(epistemic_all, dim=0)

    return outputs_all, targets_all, alea_all, epistemic_all


# Adopted from https://github.com/teddykoker/evidential-learning-pytorch
class NormalInvGamma(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        out = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        # mu_c, logv_c, logalpha_c, logbeta_c = torch.chunk(out, 4, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        alea_var = beta / (alpha - 1)
        return (mu, v, alpha, beta), alea_var


class Dirichlet(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units)
        self.out_units = int(out_units)

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        out = self.dense(x)
        alpha = self.evidence(out) + 1
        return alpha

        # TRANSLATED FROM TF
        # output = self.dense(x)
        # evidence_ = torch.exp(output)
        # alpha = evidence_ + 1
        # prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        # # return torch.cat([alpha, prob], dim=-1)
        # return prob, alpha


class EvidentialDNN(BaselineDNN):
    def __init__(
            self,
            config=None,
            logger=None,
            **kwargs,
    ):
        super(EvidentialDNN, self).__init__(
            config,
            logger,
            **kwargs,
        )

        if self.task_type == "regression":
            self.output_layer = NormalInvGamma(
                self.config["regressor_layers"][-1], self.output_dim
            )
            # self.regressor_or_classifier[-1] = NormalInvGamma(
            #     self.config["regressor_layers"][-1], self.output_dim
            # )
        elif self.task_type == "classification":
            self.output_layer = Dirichlet(
                self.config["regressor_layers"][-1], self.output_dim
            )
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        self.apply(
            self.init_wt
        )
        # reinitiating wts to include the new layers in the model initialization.

    def forward(self, inputs):
        prot_input, chem_input = inputs
        chem_features = self.chem_feature_extractor(chem_input)
        if not self.MT:
            prot_features = self.prot_feature_extractor(prot_input)
            combined_features = torch.cat((chem_features, prot_features), dim=1)
        else:
            combined_features = chem_features
        _output = self.regressor_or_classifier(combined_features)
        output, var_ = self.output_layer(_output)

        return output, var_


def run_evidential(config=None):
    if config is None:
        config = get_model_config("evidential")

    task_type = config.get("task_type", "regression")
    loss = config.get("loss", "evidential_regression")
    # Enforce loss type :
    if task_type == "regression" and loss != "evidential_regression":
        raise ValueError(f"Evidential regression loss should be evidence regression")
    elif task_type == "classification" and loss != "evidential_classification":
        raise ValueError(f"Evidential classification loss should be evidence classification")

    # Temporary turning off aleatoric for training
    # aleat_ = config.get("aleatoric", False)
    # config["aleatoric"] = False

    best_model, dataloaders, config, logger = train_model_e2e(
        config,
        model=EvidentialDNN,
        model_type="evidential",
        logger=LOGGER,
    )

    preds, labels, alea_vars, epi_vars = ev_predict(
        best_model,
        dataloaders["test"],
        device=DEVICE
    )

    # Then comes the predict metrics part
    metrics, plots, uct_logger = evaluate_predictions(
        config,
        preds,
        labels,
        alea_vars,  # TODO dealing with epistemic_vars instead of std Shapes are important here
        "evidential",
        logger,
        epi_vars
    )

    # RECALIBRATION
    preds_val, labels_val, alea_vars_val, epi_vars_vals = ev_predict(
        best_model,
        dataloaders["val"],
        device=DEVICE
    )
    recal_model = recalibrate_model(preds_val, labels_val, preds, labels, config, epi_val=epi_vars_vals, epi_test=epi_vars)

    return best_model, recal_model, metrics, plots

    #
    # # TODO:
    # # here we should specify the loss function according to the task_type,
    # # we dont need the config in this one we need to force it.
    # if
    #
    # raise NotImplementedError("This function is not implemented yet.")


def run_evidential_wrapper(**kwargs):
    global LOGGER
    LOGGER = create_logger(name="evidential", file_level="debug", stream_level="info")
    config = get_model_config(model_name="evidential", **kwargs)

    return run_evidential(config)


def main():

    parser = argparse.ArgumentParser(description="Run Ensemble Model")
    parser.add_argument(
        "--aleatoric",
        type=bool,
        default=True,
        help="Aleatoric inference"
    )
    # parser.add_argument(
    #     "--ensemble_size",
    #     type=int,
    #     default=100,
    #     help="Size of the ensemble",
    # )
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
        "--descriptor_protein",
        type=str,
        default="ankh-base",
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
    parser.add_argument(
        "--median_scaling",
        action="store_true",
        help="Use median scaling",
    )
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
    parser.add_argument(
        "--task_type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type argument",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="ensemble-test",
        help="Wandb project name argument",
    )
    parser.add_argument(
        "--sweep-count",
        type=int,
        default=None,
        help="Sweep count argument",
    )
    # take chem layers as list input
    parser.add_argument(
        "--chem_layers",
        type=parse_list,
        default=None,
        help="Chem layers sizes",
    )
    parser.add_argument(
        "--prot_layers", type=parse_list, default=None, help="Prot layers sizes"
    )
    parser.add_argument(
        "--regressor_layers",
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
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=None, help="Weight decay rate"
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default=None, help="LR scheduler type"
    )
    parser.add_argument(
        "--lr_scheduler_patience", type=int, default=None, help="LR scheduler patience"
    )
    parser.add_argument(
        "--lr_scheduler_factor", type=float, default=None, help="LR scheduler factor"
    )
    parser.add_argument(
        "--max_norm", type=float, default=None, help="Max norm for gradient clipping"
    )

    args = parser.parse_args()
    # Construct kwargs, excluding arguments that were not provided
    kwargs = {k: v for k, v in vars(args).items() if v is not None}

    sweep_count = args.sweep_count # TODO no sweep for now
    # if sweep_count is not None and sweep_count > 0:
    run_evidential_wrapper(
        **kwargs,
    )
    # else:
    #     run_ensemble_wrapper(
    #         **kwargs,
    #     )

    # raise NotImplementedError


if __name__ == "__main__":
    main()
    # run_evidential_wrapper(
    #     data_name="papyrus",
    #     n_targets=-1,
    #     task_type="regression",
    #     activity_type="xc50",
    #     split_type="random",
    #     descriptor_protein="ankh-base",
    #     descriptor_chemical="ecfp2048",
    #     median_scaling=False,
    #     ext="pkl",
    #     wandb_project_name="test-evidential",
    #     # sweep_count = 0  # 250
    #     # aleatoric = True
    #     # epochs=1
    # )
    # main()
