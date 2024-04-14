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
    predict, recalibrate_model
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
            outputs = model(inputs)
            mu, v, alpha, beta = (d.squeeze() for d in outputs)
            alea_vars = beta / (alpha - 1)  # aleatoric
            epist_var = torch.sqrt(beta / (v * (alpha - 1))) # epistemic
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
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        # alea_var = beta / (alpha - 1)
        return mu, v, alpha, beta


class Dirichlet(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units)
        self.out_units = out_units

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        out = self.dense(x)
        alpha = self.evidence(out) + 1
        return alpha


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
            self.regressor_or_classifier[-1] = NormalInvGamma(
                self.config["regressor_layers"][-1], self.output_dim
            )
        elif self.task_type == "classification":
            self.regressor_or_classifier[-1] = Dirichlet(
                self.config["regressor_layers"][-1], self.output_dim
            )
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        self.apply(
            self.init_weights
        )  # reinitiating wts to include the new layers in the model initialization.


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
    config["aleatoric"] = False

    best_model, dataloaders, config, logger = train_model_e2e(
        config,
        model=EvidentialDNN,
        model_type="evidential",
        logger=LOGGER,
    )

    preds, labels, alea_vars, epistemic_vars = ev_predict(
        best_model,
        dataloaders["test"],
        device=DEVICE
    )

    # Then comes the predict metrics part
    metrics, plots = evaluate_predictions(
        config,
        preds,
        labels,
        alea_vars, # TODO dealing with epistemic_vars instead of std Shapes are important here
        "evidential",
        logger
    )

    # RECALIBRATION
    preds_val, labels_val, alea_vars_val, epistemic_vars = ev_predict(
        best_model,
        dataloaders["val"],
        device=DEVICE
    )
    recal_model = recalibrate_model(preds_val, labels_val, preds, labels, config)

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
    raise NotImplementedError


if __name__ == "__main__":
    # run_evidential_wrapper()
    main()