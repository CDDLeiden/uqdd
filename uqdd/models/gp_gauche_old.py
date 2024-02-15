import gpytorch
from gpytorch.models import ExactGP, VariationalGP, ApproximateGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import RBFKernel, ScaleKernel, Kernel, MultitaskKernel, SpectralDeltaKernel, SpectralMixtureKernel
from uqdd.models.gp_kernels_gpytorch import TanimotoKernel
from uqdd.models.baselines import BaselineDNN

from botorch import fit_gpytorch_model
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

from gauche.dataloader import MolPropLoader
from gauche.dataloader.data_utils import transform_data


class TanimotoGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(TanimotoGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialise GP likelihood, model and
# marginal log likelihood objective
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = TanimotoGP(X_train, y_train, likelihood)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# fit GP with BoTorch in order to use
# the LBFGS-B optimiser (recommended)
fit_gpytorch_model(mll)

# use the trained GP to get predictions and
# uncertainty estimates for new molecules
model.eval()
likelihood.eval()
preds = model(X_test)
pred_means, pred_vars = preds.mean, preds.variance