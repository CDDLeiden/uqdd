from typing import List
import torch.nn as nn

# from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import spectral_norm

import gpytorch
from gpytorch.models import ExactGP, VariationalGP, ApproximateGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import (
    RBFKernel,
    ScaleKernel,
    Kernel,
    MultitaskKernel,
    SpectralDeltaKernel,
    SpectralMixtureKernel,
)
from uqdd.models.gp_kernels_gpytorch import TanimotoKernel
from uqdd.models.baseline import MTBaselineDNN


class GPLayer(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        # self.covar_module = ScaleKernel(TanimotoKernel())
        # You can add more kernel components here if needed

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskGP(ExactGP):
    def __init__(
        self, train_x, train_y, likelihood, num_tasks, rank=1
    ):  # TODO check rank argument
        super(MultitaskGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# TODO using SVGP? --> it is stochastic - with batches
#  https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html
class MultitaskSVGP(ApproximateGP):
    def __init__(
        self, inducing_points, num_tasks, mean_kernel=None, covar_kernel=None, rank=1
    ):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(MultitaskSVGP, self).__init__(variational_strategy)

        mean_kernel = (
            gpytorch.means.ConstantMean() if mean_kernel is None else mean_kernel
        )
        covar_kernel = (
            gpytorch.kernels.RBFKernel() if covar_kernel is None else covar_kernel
        )
        self.mean_module = gpytorch.means.MultitaskMean(
            mean_kernel, num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            covar_kernel, num_tasks=num_tasks, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BaselineDNNwithGP(nn.Module):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        mean_kernel: Kernel = None,
        covar_kernel: Kernel = None,
        input_dim: int = 2048,
        hidden_dims: List[int] = [2048, 256, 256],
        num_tasks: int = 20,
        use_spectral_norm: bool = False,
    ):
        super(BaselineDNNwithGP, self).__init__()
        # Define the DNN part
        baseline_dnn = MTBaselineDNN(
            input_dim=input_dim,
            hidden_dim_1=hidden_dims[0],
            hidden_dim_2=hidden_dims[1],
            hidden_dim_3=hidden_dims[2],
        )
        self.feature_extractor = (
            baseline_dnn.feature_extractor
        )  # existing DNN architecture without the last layer
        # Define the GP part
        self.gp_layer = MultitaskSVGP(
            inducing_points,
            num_tasks,
            mean_kernel=mean_kernel,
            covar_kernel=covar_kernel,
        )

        if use_spectral_norm:
            # self.feature_extractor = spectral_norm(self.feature_extractor)
            self.gp_layer = spectral_norm(self.gp_layer)

    def forward(self, x):
        features = self.feature_extractor(x)
        # If using spectral normalization, apply it to features here
        gp_output = self.gp_layer(features)
        return gp_output


# Train the model
def train_dnn_with_gp(model, train_loader, likelihood, optimizer, mll):
    model.train()
    likelihood.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = -mll(output, target)
        loss.backward()
        optimizer.step()


# Later in your existing code
if __name__ == "__main__":
    # ... your existing setup code ...
    use_spectral_normalization = True  # Set this based on your requirements
    model = BaselineDNNwithGP(
        input_dim, hidden_dims, output_dim, use_spectral_normalization
    )
    likelihood = GaussianLikelihood()
    mll = ExactMarginalLogLikelihood(likelihood, model.gp_layer)
    # ... rest of your training code ...
