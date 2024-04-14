import logging

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


def calc_loss_notnan(outputs, targets, alea_vars, loss_fn, reduction='mean'):
    """
    Calculates the loss for non-NaN values between outputs and targets using a given loss function,
    and aggregates it per task.

    Parameters
    ----------
    outputs : torch.Tensor
        Predicted outputs from the model, shape [batch_size, num_tasks, num_models]
    targets : torch.Tensor
        True target values, shape [batch_size, num_tasks, 1] after unsqueezing
    alea_vars : torch.Tensor or None
        Predicted aleatoric variances from the model, same shape as outputs.
    loss_fn : Callable
        A loss function compatible with torch.Tensors and supports 'none' reduction.
    reduction : str
        The method for reducing the loss across the batch ('mean', 'sum', or 'none').

    Returns
    -------
    torch.Tensor
        The aggregated loss value excluding NaNs, shape [num_tasks, 1] or scalar if reduction='mean'.
    """
    assert outputs.dim() == 3, "Outputs should be [batch_size, num_tasks, num_models]"
    # if targets.dim() < outputs.dim():
    #     targets =
    # assert targets.dim() == 3, "Targets should be [batch_size, num_tasks, 1] after unsqueezing"

    # Mask out NaN values in targets
    nan_mask = torch.isnan(targets)
    valid_mask = ~nan_mask.squeeze(-1)  # Reduce to two dimensions if not [batch_size, num_tasks]

    # Initialize containers for task-wise losses
    task_losses = torch.zeros(targets.size(1), device=outputs.device)  # [num_tasks]

    # Process each model's predictions separately
    for i in range(outputs.shape[2]):  # Loop through each model in the ensemble
        model_outputs = outputs[:, :, i]
        model_vars = alea_vars[:, :, i] if alea_vars is not None else None

        # Calculate the loss only on valid (non-NaN) entries
        for task_index in range(outputs.size(1)):  # Iterate over each task
            task_valid_mask = valid_mask[:, task_index]
            if task_valid_mask.any():  # Only compute where there are valid targets
                task_outputs = model_outputs[:, task_index][task_valid_mask]
                task_targets = targets[:, task_index][task_valid_mask]
                task_vars = model_vars[:, task_index][task_valid_mask] if model_vars is not None else None

                if task_vars is not None:
                    task_loss = loss_fn(task_outputs, task_targets, task_vars)
                else:
                    task_loss = loss_fn(task_outputs, task_targets)

                # Sum up or mean loss for this task across all valid samples
                if reduction == 'mean':
                    task_losses[task_index] += task_loss.mean()  # Average the loss for this task
                else:  # sum or none
                    task_losses[task_index] += task_loss.sum()

    # Normalize by the number of models in the ensemble
    task_losses /= outputs.shape[2]

    if reduction == 'mean':
        return task_losses.mean()  # Return mean loss across tasks
    elif reduction == 'sum':
        return task_losses.sum()  # Return sum of losses across tasks
    return task_losses  # Return losses per task as a tensor


# def _calc_loss_notnan(outputs, targets, alea_vars, loss_fn, reduction='mean'):
#     """
#     Calculates the loss for non-NaN values between outputs and targets using a given loss function.
#
#     Parameters
#     ----------
#     outputs : torch.Tensor
#         Predicted outputs from the model.
#     targets : torch.Tensor
#         True target values.
#     loss_fn : Callable or function
#         A loss function compatible with torch.Tensors and supports 'none' reduction.
#     reduction : str
#         The method for reducing the loss across the batch ('mean', 'sum', or 'none').
#
#     Returns
#     -------
#     torch.Tensor
#         The aggregated loss value excluding NaNs.
#     """
#     # Shapes fix
#     if outputs.dim() > targets.dim():
#         targets = targets.unsqueeze(-1)
#
#     nan_mask = torch.isnan(targets)
#     valid_targets = torch.where(
#         ~nan_mask, targets, torch.tensor(0.0, device=targets.device)
#     )
#     valid_outputs = torch.where(
#         ~nan_mask, outputs, torch.tensor(0.0, device=outputs.device)
#     )
#
#     if alea_vars is not None:
#         valid_alea_vars = torch.where(
#             ~nan_mask, alea_vars, torch.tensor(0.0, device=alea_vars.device)
#         )
#         loss_per_task = loss_fn(valid_outputs, valid_targets, valid_alea_vars)
#     else:
#         loss_per_task = loss_fn(valid_outputs, valid_targets)
#     loss = calc_nanaware_metrics(loss_per_task, nan_mask, all_tasks_agg="sum")
#
#     return loss


### Custom Loss Functions ###
class MultiTaskLoss(nn.Module):
    def __init__(self, loss_type="huber", reduction="mean", lamb=1e-2, **kwargs):
        super(MultiTaskLoss, self).__init__()
        self.loss_type = loss_type
        self.loss_fn = build_loss(
            loss_type, reduction="none", lamb=lamb, **kwargs
        )
        self.reduction = reduction

    def forward(self, outputs, targets, alea_vars):
        # loss
        loss = calc_loss_notnan(outputs, targets, alea_vars, self.loss_fn, self.reduction) # "mean"
        # loss = loss.mean()
        return loss

    # def prepare_mt_args(self, outputs, targets, alea_vars):
    #     if outputs.dim() > targets.dim():


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


def build_loss(
    loss,
    reduction="none",
    lamb=1e-2,
    mt=False,
    **kwargs,
):
    if mt:
        # if reduction != "none":
        #     logger.warning(
        #         f"reduction should only be none with multitask learning to be able to calculate loss per each task. {reduction=} is provided instead"
        #     )
        #     reduction = "none"
        return MultiTaskLoss(loss_type=loss, reduction=reduction, lamb=lamb, **kwargs)

    if loss.lower() in ["mse", "l2"]:
        loss_fn = nn.MSELoss(reduction=reduction, **kwargs)
    elif loss.lower() in ["mae", "l1"]:
        loss_fn = nn.L1Loss(reduction=reduction, **kwargs)
    elif loss.lower() == "huber":
        loss_fn = nn.HuberLoss(reduction=reduction, **kwargs)
    elif loss.lower() == "smoothl1":
        loss_fn = nn.SmoothL1Loss(reduction=reduction, **kwargs)
    elif loss.lower() == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss(reduction=reduction, **kwargs)
    elif loss.lower() == "nll":
        loss_fn = nn.NLLLoss(reduction=reduction, **kwargs)
    elif loss.lower() == "gaussnll":
        loss_fn = nn.GaussianNLLLoss(reduction=reduction, **kwargs)
    elif loss.lower() == "evidential_regression":
        # TODO how to deal with reduction here on this one?
        loss_fn = EvidenceRegressionLoss(lamb=lamb)
    elif loss.lower() == "evidential_classification":
        loss_fn = EvidentialClassLoss(lamb=lamb)
    else:
        raise ValueError("Unknown loss: {}".format(loss))
    return loss_fn
