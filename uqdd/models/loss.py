import logging
from typing import Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Normal Inverse Gamma Negative Log-Likelihood
# Adopted and Modified from https://arxiv.org/abs/1910.02600:
# > we denote the loss, L^NLL_i as the negative logarithm of model
# > evidence ...
def nig_nll(
    gamma: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Negative Log-Likelihood (NLL) loss for a Normal Inverse Gamma distribution.

    Parameters
    ----------
    gamma : torch.Tensor
        Mean prediction.
    v : torch.Tensor
        Variance scaling factor.
    alpha : torch.Tensor
        Shape parameter for inverse gamma distribution.
    beta : torch.Tensor
        Scale parameter for inverse gamma distribution.
    y : torch.Tensor
        Target values.

    Returns
    -------
    torch.Tensor
        Mean negative log-likelihood loss.
    """
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
def nig_reg(
    gamma: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Computes the regularization term for evidential regression.

    Parameters
    ----------
    gamma : torch.Tensor
        Mean prediction.
    v : torch.Tensor
        Variance scaling factor.
    alpha : torch.Tensor
        Shape parameter for inverse gamma distribution.
    beta : torch.Tensor
        Scale parameter for inverse gamma distribution.
    y : torch.Tensor
        Target values.

    Returns
    -------
    torch.Tensor
        Mean evidential regularization loss.
    """
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()


# KL divergence of predicted parameters from uniform Dirichlet distribution
# from https://arxiv.org/abs/1806.01768
# code based on:
# https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
def dirichlet_reg(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL divergence from a uniform Dirichlet distribution.

    Parameters
    ----------
    alpha : torch.Tensor
        Dirichlet concentration parameters.
    y : torch.Tensor
        Target labels.

    Returns
    -------
    torch.Tensor
        KL divergence regularization term.
    """
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

def dirichlet_mse(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the sum-of-squares loss for evidential classification.

    Parameters
    ----------
    alpha : torch.Tensor
        Dirichlet concentration parameters.
    y : torch.Tensor
        Target labels.

    Returns
    -------
    torch.Tensor
        Mean squared error loss.
    """
    sum_alpha = alpha.sum(-1, keepdims=True)
    p = alpha / sum_alpha
    t1 = (y - p).pow(2).sum(-1)
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1)
    mse = t1 + t2
    return mse.mean()



def calc_loss_notnan(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alea_vars: Optional[torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Calculates the loss for non-NaN values between outputs and targets using a given loss function,
    aggregating per task.

    Parameters
    ----------
    outputs : torch.Tensor
        Model predictions with shape [batch_size, num_tasks, num_models].
    targets : torch.Tensor
        Ground truth labels with shape [batch_size, num_tasks, 1].
    alea_vars : torch.Tensor, optional
        Aleatoric variances, same shape as outputs.
    loss_fn : Callable
        Loss function to compute the loss.
    reduction : str, optional
        Reduction method ('mean', 'sum', or 'none'), by default 'mean'.

    Returns
    -------
    torch.Tensor
        Aggregated loss value.
    """
    assert outputs.dim() == 3, "Outputs should be [batch_size, num_tasks, num_models]"

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


class MultiTaskLoss(nn.Module):
    """
    Multi-task learning loss function that computes losses across multiple tasks.

    Parameters
    ----------
    loss_type : str, optional
        Type of loss function to use, by default "huber".
    reduction : str, optional
        Reduction method ('mean', 'sum', or 'none'), by default 'mean'.
    lamb : float, optional
        Regularization weight, by default 1e-2.
    """

    def __init__(
            self, loss_type: str = "huber", reduction: str = "mean", lamb: float = 1e-2, **kwargs
    ) -> None:
        super(MultiTaskLoss, self).__init__()
        self.loss_type = loss_type
        self.loss_fn = build_loss(
            loss_type, reduction="none", lamb=lamb, **kwargs
        )
        self.reduction = reduction

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, alea_vars: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the loss across tasks.

        Parameters
        ----------
        outputs : torch.Tensor
            Model predictions.
        targets : torch.Tensor
            Ground truth labels.
        alea_vars : torch.Tensor, optional
            Aleatoric variances.

        Returns
        -------
        torch.Tensor
            Computed loss.
        """
        loss = calc_loss_notnan(outputs, targets, alea_vars, self.loss_fn, self.reduction)
        return loss


class EvidentialClassLoss(nn.Module):
    """
    Evidential classification loss function.

    Parameters
    ----------
    lamb : float, optional
        Regularization weight, by default 1.0.
    """
    def __init__(self, lamb: float = 1.0) -> None:
        super(EvidentialClassLoss, self).__init__()
        self.lamb = lamb

    def forward(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the evidential classification loss.

        Parameters
        ----------
        alpha : torch.Tensor
            Dirichlet distribution parameters.
        y : torch.Tensor
            Target values.

        Returns
        -------
        torch.Tensor
            Computed loss.
        """
        num_classes = alpha.shape[-1]
        y = F.one_hot(y, num_classes)
        loss = dirichlet_mse(alpha, y) + self.lamb * dirichlet_reg(alpha, y)
        return loss


class EvidenceRegressionLoss(nn.Module):
    """
    Evidential regression loss function.

    Parameters
    ----------
    lamb : float, optional
        Regularization weight, by default 1.0.
    """
    def __init__(self, lamb: float = 1.0) -> None:
        super(EvidenceRegressionLoss, self).__init__()
        self.lamb = lamb

    def forward(self, dist_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """
        Computes the evidential regression loss.

        Parameters
        ----------
        dist_params : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Normal Inverse Gamma distribution parameters.
        y : torch.Tensor
            Target values.

        Returns
        -------
        torch.Tensor
            Computed loss.
        """
        loss = nig_nll(*dist_params, y) + self.lamb * nig_reg(*dist_params, y)
        return loss


def build_loss(
    loss: str,
    reduction: str = "none",
    lamb: float = 1e-2,
    mt: bool = False,
    **kwargs
) -> nn.Module:
    """
    Constructs and returns the specified loss function.

    Parameters
    ----------
    loss : str
        Type of loss function.
    reduction : str, optional
        Reduction method ('none', 'mean', 'sum'), by default 'none'.
    lamb : float, optional
        Regularization weight, by default 1e-2.
    mt : bool, optional
        Whether to use multi-task loss, by default False.

    Returns
    -------
    nn.Module
        Instantiated loss function.
    """
    if mt:
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
        loss_fn = EvidenceRegressionLoss(lamb=lamb)
    elif loss.lower() == "evidential_classification":
        loss_fn = EvidentialClassLoss(lamb=lamb)
    else:
        raise ValueError("Unknown loss: {}".format(loss))
    return loss_fn
