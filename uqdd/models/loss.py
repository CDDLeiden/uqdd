import torch
import torch.nn as nn
import torch.nn.functional as F

from uqdd.models.utils_metrics import calc_nanaware_metrics


# Normal Inverse Gamma Negative Log-Likelihood
# from https://arxiv.org/abs/1910.02600:
# > we denote the loss, L^NLL_i as the negative logarithm of model
# > evidence ...
def nig_nll(gamma, v, alpha, beta, y):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()


# Normal Inverse Gamma regularization
# from https://arxiv.org/abs/1910.02600:
# > we formulate a novel evidence regularizer, L^R_i
# > scaled on the error of the i-th prediction
def nig_reg(gamma, v, alpha, _beta, y):
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()


# KL divergence of predicted parameters from uniform Dirichlet distribution
# from https://arxiv.org/abs/1806.01768
# code based on:
# https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
def dirichlet_reg(alpha, y):
    # dirichlet parameters after removal of non-misleading evidence (from the label)
    alpha = y + (1 - y) * alpha

    # uniform dirichlet distribution
    beta = torch.ones_like(alpha)

    sum_alpha = alpha.sum(-1)
    sum_beta = beta.sum(-1)

    t1 = sum_alpha.lgamma() - sum_beta.lgamma()
    t2 = (alpha.lgamma() - beta.lgamma()).sum(-1)
    t3 = alpha - beta
    t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

    kl = t1 - t2 + (t3 * t4).sum(-1)
    return kl.mean()


# Eq. (5) from https://arxiv.org/abs/1806.01768:
# Sum of squares loss
def dirichlet_mse(alpha, y):
    sum_alpha = alpha.sum(-1, keepdims=True)
    p = alpha / sum_alpha
    t1 = (y - p).pow(2).sum(-1)
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1)
    mse = t1 + t2
    return mse.mean()


#
# def evidential_classification(alpha, y, lamb=1.0):
#     """
#     Evidential Classification Loss Function
#     Parameters
#     ----------
#     alpha : torch.Tensor
#         Dirichlet parameters
#     y : torch.Tensor
#         Target values
#     lamb: float
#         Regularization parameter (default: 1.0)
#
#     Returns
#     -------
#     torch.Tensor
#         Evidential Classification Loss
#     """
#     num_classes = alpha.shape[-1]
#     y = F.one_hot(y, num_classes)
#     return dirichlet_mse(alpha, y) + lamb * dirichlet_reg(alpha, y)
#
#
# def evidential_regression(dist_params, y, lamb=1.0):
#     return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y)


def calc_loss_notnan(outputs, targets, nan_mask, loss_fn):
    valid_targets = torch.where(
        ~nan_mask, targets, torch.tensor(0.0, device=targets.device)
    )
    valid_outputs = torch.where(
        ~nan_mask, outputs, torch.tensor(0.0, device=outputs.device)
    )

    loss_per_task = loss_fn(valid_outputs, valid_targets, reduction="none")
    loss = calc_nanaware_metrics(loss_per_task, nan_mask, all_tasks_agg="sum")

    return loss


### Custom Loss Functions ###
class MultiTaskLoss(nn.Module):
    def __init__(self, loss_type="huber", reduction="mean", lamb=1e-2):
        super(MultiTaskLoss, self).__init__()
        self.loss_type = loss_type
        self.loss_fn = build_loss(loss_type, reduction=reduction, lamb=lamb)

    def forward(self, outputs, targets):
        nan_mask = torch.isnan(targets)
        # loss
        loss = calc_loss_notnan(outputs, targets, nan_mask, self.loss_fn)
        return loss


class EvidentialClassLoss(nn.Module):
    def __init__(self, lamb=1.0):
        """
        Evidential Classification Loss Function
        Parameters
        ----------
        lamb: float
            Regularization parameter (default: 1.0)
        """
        super(EvidentialClassLoss, self).__init__()
        self.lamb = lamb

    def forward(self, alpha, y):
        """
        Forward pass of the loss function for evidential classification model
        Parameters
        ----------
        alpha : torch.Tensor
            Dirichlet parameters
        y : torch.Tensor
            Target values

        Returns
        -------
        torch.Tensor
            Evidential Classification Loss
        """
        num_classes = alpha.shape[-1]
        y = F.one_hot(y, num_classes)
        loss = dirichlet_mse(alpha, y) + self.lamb * dirichlet_reg(alpha, y)
        return loss


class EvidenceRegressionLoss(nn.Module):
    def __init__(self, lamb=1.0):
        """
        Evidential Regression Loss Function
        Parameters
        ----------
        lamb: float
            Regularization parameter (default: 1.0)
        """
        super(EvidenceRegressionLoss, self).__init__()
        self.lamb = lamb

    def forward(self, dist_params, y):
        """
        Forward pass of the loss function for evidential regression model
        Parameters
        ----------
        dist_params : tuple
            Tuple of parameters of the Normal Inverse Gamma distribution
        y: torch.Tensor
            Target values for the regression task

        Returns
        -------
        torch.Tensor
            Evidential Regression Loss
        """
        loss = nig_nll(*dist_params, y) + self.lamb * nig_reg(*dist_params, y)
        return loss


def build_loss(loss, reduction="none", lamb=1e-2, mt=False, **kwargs):
    if mt:
        return MultiTaskLoss(loss_type=loss, reduction=reduction, lamb=lamb)

    if loss.lower() in ["mse", "l2"]:
        loss_fn = nn.MSELoss(reduction=reduction, **kwargs)
    elif loss.lower() in ["mae", "l1"]:
        loss_fn = nn.L1Loss(reduction=reduction, **kwargs)
    elif loss.lower() in ["huber", "smoothl1"]:
        loss_fn = nn.SmoothL1Loss(reduction=reduction, **kwargs)
    elif loss.lower() == "evidential_regression":
        # TODO how to deal with reduction here on this one?
        loss_fn = EvidenceRegressionLoss(lamb=lamb)
    elif loss.lower() == "evidential_classification":
        loss_fn = EvidentialClassLoss(lamb=lamb)
    else:
        raise ValueError("Unknown loss: {}".format(loss))
    return loss_fn
