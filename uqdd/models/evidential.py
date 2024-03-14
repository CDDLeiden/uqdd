import torch
from torch import nn
import torch.nn.functional as F

from uqdd.models.baseline import BaselineDNN


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
        chem_input_dim=None,
        prot_input_dim=None,
        task_type="regression",
        n_targets=-1,
        logger=None,
        **kwargs,
    ):
        super().__init__(
            config,
            chem_input_dim,
            prot_input_dim,
            task_type,
            n_targets,
            logger,
            **kwargs,
        )

        if task_type == "regression":
            self.regressor_or_classifier[-1] = NormalInvGamma(
                self.config["regressor_layers"][-1], self.output_dim
            )
        elif task_type == "classification":
            self.regressor_or_classifier[-1] = Dirichlet(
                self.config["regressor_layers"][-1], self.output_dim
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        self.apply(
            self.init_weights
        )  # reinitiating wts to include the new layers in the model initialization.


def run_evidential():

    # TODO:
    # here we should specify the loss function according to the task_type, we dont need the config in this one we need to force it.

    raise NotImplementedError("This function is not implemented yet.")
