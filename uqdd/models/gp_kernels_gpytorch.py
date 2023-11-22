import torch
import gpytorch
# from gpytorch.constraints import Positive


class TanimotoKernel(gpytorch.kernels.Kernel):
    def forward(self, x1, x2, diag=False, **params):
        # Implement the Tanimoto distance calculation here
        # This is a simplified placeholder implementation
        intersect = torch.mm(x1, x2.t())
        union = x1.pow(2).sum(1, keepdim=True) + x2.pow(2).sum(1, keepdim=True).t() - intersect
        tanimoto_similarity = intersect / union

        # Convert similarity to distance
        tanimoto_distance = 1 - tanimoto_similarity

        return tanimoto_distance if diag else tanimoto_distance.diag()

#
# class _TanimotoKernel(gpytorch.kernels.Kernel):
#     has_lengthscale = False
#
#     def __init__(self, variance_prior=None, variance_constraint=None, **kwargs):
#         super(_TanimotoKernel, self).__init__(**kwargs)
#
#         if variance_constraint is None:
#             variance_constraint = Positive()
#
#         self.register_parameter(
#             name="raw_variance",
#             parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
#         )
#         if variance_prior is not None:
#             self.register_prior(
#                 "variance_prior",
#                 variance_prior,
#                 lambda: self.variance,
#                 lambda v: self._set_variance(v),
#             )
#         self.register_constraint("raw_variance", variance_constraint)
#
#     @property
#     def variance(self):
#         return self.raw_variance_constraint.transform(self.raw_variance)
#
#     @variance.setter
#     def variance(self, value):
#         return self._set_variance(value)
#
#     def _set_variance(self, value):
#         if not torch.is_tensor(value):
#             value = torch.as_tensor(value).to(self.raw_variance)
#         self.initialize(raw_variance=self.raw_variance_constraint.inverse_transform(value))
#
#     def forward(self, x1, x2, diag=False, **params):
#         if diag:
#             return torch.fill((x1.size(0),), self.variance.squeeze())
#
#         x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
#         x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
#
#         cross_product = x1.matmul(x2.transpose(-2, -1))
#
#         denominator = -cross_product + x1_norm + x2_norm.transpose(-2, -1)
#
#         return self.variance * cross_product / denominator
#
#
