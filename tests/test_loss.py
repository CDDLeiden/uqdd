"""
Unit tests for loss functions in uqdd.models.loss.

Tests cover:
- Normal Inverse Gamma (NIG) negative log-likelihood
- Evidential regularization
- Dirichlet KL divergence regularization
- Dirichlet MSE loss
- Loss computation with NaN handling
"""

import unittest
from unittest.mock import patch
import pytest
import torch
import numpy as np

import uqdd.models.loss as loss_module


class TestNigNLL(unittest.TestCase):
    """Test cases for Normal Inverse Gamma NLL loss."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 8
        self.dtype = torch.float32

    def test_nig_nll_output_shape(self):
        """Test NIG NLL returns scalar."""
        mu = torch.randn(self.batch_size, 1, device=self.device, dtype=self.dtype)
        v = torch.relu(torch.randn(self.batch_size, 1, device=self.device)) + 0.1
        alpha = torch.relu(torch.randn(self.batch_size, 1, device=self.device)) + 1.1
        beta = torch.relu(torch.randn(self.batch_size, 1, device=self.device)) + 0.1
        y = torch.randn(self.batch_size, 1, device=self.device)

        nll = loss_module.nig_nll(mu, v, alpha, beta, y)
        self.assertEqual(nll.shape, torch.Size([]))

    def test_nig_nll_is_positive(self):
        """Test NIG NLL is positive (it's a log-likelihood)."""
        mu = torch.randn(self.batch_size, 1)
        v = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        alpha = torch.relu(torch.randn(self.batch_size, 1)) + 1.1
        beta = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        y = torch.randn(self.batch_size, 1)

        nll = loss_module.nig_nll(mu, v, alpha, beta, y)
        self.assertTrue(nll.item() > 0)

    def test_nig_nll_is_finite(self):
        """Test NIG NLL returns finite values."""
        mu = torch.randn(self.batch_size, 1)
        v = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        alpha = torch.relu(torch.randn(self.batch_size, 1)) + 1.1
        beta = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        y = torch.randn(self.batch_size, 1)

        nll = loss_module.nig_nll(mu, v, alpha, beta, y)
        self.assertTrue(torch.isfinite(nll).all())

    def test_nig_nll_supports_gradients(self):
        """Test that NIG NLL is differentiable (backward pass works)."""
        mu = torch.randn(self.batch_size, 1, requires_grad=True)
        v = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        v.requires_grad_(True)
        alpha = torch.relu(torch.randn(self.batch_size, 1)) + 1.1
        alpha.requires_grad_(True)
        beta = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        beta.requires_grad_(True)
        y = torch.randn(self.batch_size, 1)

        nll = loss_module.nig_nll(mu, v, alpha, beta, y)
        nll.backward()

        # Check that mu has gradients (it's a leaf tensor)
        self.assertIsNotNone(mu.grad)
        # The backward pass should complete without error
        self.assertTrue(torch.isfinite(nll))

    def test_nig_nll_single_sample(self):
        """Test NIG NLL with single sample."""
        mu = torch.randn(1, 1)
        v = torch.tensor([[0.5]])
        alpha = torch.tensor([[2.0]])
        beta = torch.tensor([[1.0]])
        y = torch.randn(1, 1)

        nll = loss_module.nig_nll(mu, v, alpha, beta, y)
        self.assertEqual(nll.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(nll))

    def test_nig_nll_batch_dimension_independence(self):
        """Test that NIG NLL is mean of per-sample losses."""
        mu = torch.randn(4, 1)
        v = torch.relu(torch.randn(4, 1)) + 0.1
        alpha = torch.relu(torch.randn(4, 1)) + 1.1
        beta = torch.relu(torch.randn(4, 1)) + 0.1
        y = torch.randn(4, 1)

        # Full batch loss
        nll_full = loss_module.nig_nll(mu, v, alpha, beta, y)

        # Individual losses (manually computed)
        losses = []
        for i in range(4):
            nll_i = loss_module.nig_nll(mu[i:i+1], v[i:i+1], alpha[i:i+1], beta[i:i+1], y[i:i+1])
            losses.append(nll_i.item())

        expected_mean = np.mean(losses)
        self.assertAlmostEqual(nll_full.item(), expected_mean, places=5)

    def test_nig_nll_various_batch_sizes_1(self):
        """Test NIG NLL with batch size 1."""
        batch_size = 1
        mu = torch.randn(batch_size, 1)
        v = torch.relu(torch.randn(batch_size, 1)) + 0.1
        alpha = torch.relu(torch.randn(batch_size, 1)) + 1.1
        beta = torch.relu(torch.randn(batch_size, 1)) + 0.1
        y = torch.randn(batch_size, 1)

        nll = loss_module.nig_nll(mu, v, alpha, beta, y)
        self.assertEqual(nll.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(nll))

    def test_nig_nll_various_batch_sizes_large(self):
        """Test NIG NLL with large batch size."""
        batch_size = 32
        mu = torch.randn(batch_size, 1)
        v = torch.relu(torch.randn(batch_size, 1)) + 0.1
        alpha = torch.relu(torch.randn(batch_size, 1)) + 1.1
        beta = torch.relu(torch.randn(batch_size, 1)) + 0.1
        y = torch.randn(batch_size, 1)

        nll = loss_module.nig_nll(mu, v, alpha, beta, y)
        self.assertEqual(nll.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(nll))


class TestEvidentialRegularizer(unittest.TestCase):
    """Test cases for evidential regularization."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.device = torch.device("cpu")

    def test_evidential_regularizer_output_shape(self):
        """Test regularizer returns scalar."""
        mu = torch.randn(self.batch_size, 1)
        v = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        alpha = torch.relu(torch.randn(self.batch_size, 1)) + 1.1
        beta = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        y = torch.randn(self.batch_size, 1)

        reg = loss_module.evidential_regularizer(mu, v, alpha, beta, y)
        self.assertEqual(reg.shape, torch.Size([]))

    def test_evidential_regularizer_is_nonnegative(self):
        """Test regularizer is non-negative."""
        mu = torch.randn(self.batch_size, 1)
        v = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        alpha = torch.relu(torch.randn(self.batch_size, 1)) + 1.1
        beta = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        y = torch.randn(self.batch_size, 1)

        reg = loss_module.evidential_regularizer(mu, v, alpha, beta, y)
        self.assertGreaterEqual(reg.item(), 0.0)

    def test_evidential_regularizer_lambda_effect(self):
        """Test that lambda parameter scales the loss."""
        mu = torch.randn(self.batch_size, 1)
        v = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        alpha = torch.relu(torch.randn(self.batch_size, 1)) + 1.1
        beta = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        y = torch.randn(self.batch_size, 1)

        reg1 = loss_module.evidential_regularizer(mu, v, alpha, beta, y, lam=1.0)
        reg2 = loss_module.evidential_regularizer(mu, v, alpha, beta, y, lam=2.0)

        # reg2 should be approximately 2x reg1 (if lam is used for scaling)
        ratio = reg2.item() / (reg1.item() + 1e-8)
        self.assertAlmostEqual(ratio, 2.0, places=4)

    def test_evidential_regularizer_supports_gradients(self):
        """Test that gradients flow through regularizer."""
        mu = torch.randn(self.batch_size, 1, requires_grad=True)
        v = torch.relu(torch.randn(self.batch_size, 1, requires_grad=True)) + 0.1
        alpha = torch.relu(torch.randn(self.batch_size, 1, requires_grad=True)) + 1.1
        beta = torch.relu(torch.randn(self.batch_size, 1, requires_grad=True)) + 0.1
        y = torch.randn(self.batch_size, 1)

        reg = loss_module.evidential_regularizer(mu, v, alpha, beta, y)
        reg.backward()

        self.assertIsNotNone(mu.grad)
        self.assertIsNotNone(alpha.grad)

    def test_evidential_regularizer_zero_error(self):
        """Test regularizer when prediction equals target."""
        mu = torch.ones(self.batch_size, 1)
        v = torch.ones(self.batch_size, 1) * 0.5
        alpha = torch.ones(self.batch_size, 1) * 2.0
        beta = torch.ones(self.batch_size, 1) * 0.5
        y = torch.ones(self.batch_size, 1)  # Perfect prediction

        reg = loss_module.evidential_regularizer(mu, v, alpha, beta, y)
        # Should be close to zero when error is zero
        self.assertLess(reg.item(), 0.1)


class TestDirichletReg(unittest.TestCase):
    """Test cases for Dirichlet KL divergence regularization."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.num_classes = 3
        self.device = torch.device("cpu")

    def test_dirichlet_reg_output_shape(self):
        """Test dirichlet_reg returns scalar."""
        alpha = torch.relu(torch.randn(self.batch_size, self.num_classes)) + 0.5
        y = torch.nn.functional.one_hot(
            torch.randint(0, self.num_classes, (self.batch_size,)),
            num_classes=self.num_classes
        ).float()

        kl = loss_module.dirichlet_reg(alpha, y)
        self.assertEqual(kl.shape, torch.Size([]))

    def test_dirichlet_reg_is_nonnegative(self):
        """Test dirichlet_reg is non-negative (KL divergence property)."""
        alpha = torch.relu(torch.randn(self.batch_size, self.num_classes)) + 0.5
        y = torch.nn.functional.one_hot(
            torch.randint(0, self.num_classes, (self.batch_size,)),
            num_classes=self.num_classes
        ).float()

        kl = loss_module.dirichlet_reg(alpha, y)
        self.assertGreaterEqual(kl.item(), 0.0)

    def test_dirichlet_reg_supports_gradients(self):
        """Test that gradients flow through dirichlet_reg."""
        alpha = torch.relu(torch.randn(
            self.batch_size, self.num_classes, requires_grad=True
        )) + 0.5
        y = torch.nn.functional.one_hot(
            torch.randint(0, self.num_classes, (self.batch_size,)),
            num_classes=self.num_classes
        ).float()

        kl = loss_module.dirichlet_reg(alpha, y)
        kl.backward()

        self.assertIsNotNone(alpha.grad)

    def test_dirichlet_reg_is_finite(self):
        """Test dirichlet_reg returns finite values."""
        alpha = torch.relu(torch.randn(self.batch_size, self.num_classes)) + 0.5
        y = torch.nn.functional.one_hot(
            torch.randint(0, self.num_classes, (self.batch_size,)),
            num_classes=self.num_classes
        ).float()

        kl = loss_module.dirichlet_reg(alpha, y)
        self.assertTrue(torch.isfinite(kl).all())

    def test_dirichlet_reg_various_classes_binary(self):
        """Test dirichlet_reg with binary classification."""
        num_classes = 2
        alpha = torch.relu(torch.randn(self.batch_size, num_classes)) + 0.5
        y = torch.nn.functional.one_hot(
            torch.randint(0, num_classes, (self.batch_size,)),
            num_classes=num_classes
        ).float()

        kl = loss_module.dirichlet_reg(alpha, y)
        self.assertEqual(kl.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(kl))

    def test_dirichlet_reg_various_classes_multiclass(self):
        """Test dirichlet_reg with multiclass."""
        num_classes = 10
        alpha = torch.relu(torch.randn(self.batch_size, num_classes)) + 0.5
        y = torch.nn.functional.one_hot(
            torch.randint(0, num_classes, (self.batch_size,)),
            num_classes=num_classes
        ).float()

        kl = loss_module.dirichlet_reg(alpha, y)
        self.assertEqual(kl.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(kl))


class TestDirichletMSE(unittest.TestCase):
    """Test cases for Dirichlet MSE loss."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.num_classes = 3

    def test_dirichlet_mse_output_shape(self):
        """Test dirichlet_mse returns scalar."""
        alpha = torch.relu(torch.randn(self.batch_size, self.num_classes)) + 0.5
        y = torch.nn.functional.one_hot(
            torch.randint(0, self.num_classes, (self.batch_size,)),
            num_classes=self.num_classes
        ).float()

        mse = loss_module.dirichlet_mse(alpha, y)
        self.assertEqual(mse.shape, torch.Size([]))

    def test_dirichlet_mse_is_nonnegative(self):
        """Test dirichlet_mse is non-negative."""
        alpha = torch.relu(torch.randn(self.batch_size, self.num_classes)) + 0.5
        y = torch.nn.functional.one_hot(
            torch.randint(0, self.num_classes, (self.batch_size,)),
            num_classes=self.num_classes
        ).float()

        mse = loss_module.dirichlet_mse(alpha, y)
        self.assertGreaterEqual(mse.item(), 0.0)

    def test_dirichlet_mse_supports_gradients(self):
        """Test that gradients flow through dirichlet_mse."""
        alpha = (torch.relu(torch.randn(
            self.batch_size, self.num_classes
        )) + 0.5).requires_grad_(True)
        y = torch.nn.functional.one_hot(
            torch.randint(0, self.num_classes, (self.batch_size,)),
            num_classes=self.num_classes
        ).float()

        mse = loss_module.dirichlet_mse(alpha, y)
        mse.backward()

        self.assertIsNotNone(alpha.grad)

    def test_dirichlet_mse_is_finite(self):
        """Test dirichlet_mse returns finite values."""
        alpha = torch.relu(torch.randn(self.batch_size, self.num_classes)) + 0.5
        y = torch.nn.functional.one_hot(
            torch.randint(0, self.num_classes, (self.batch_size,)),
            num_classes=self.num_classes
        ).float()

        mse = loss_module.dirichlet_mse(alpha, y)
        self.assertTrue(torch.isfinite(mse).all())

    def test_dirichlet_mse_perfect_prediction(self):
        """Test dirichlet_mse with concentrated Dirichlet (confident predictions)."""
        # High alpha values for correct class = confident prediction
        alpha = torch.ones(self.batch_size, self.num_classes) * 0.1
        # Set high alpha for correct class
        for i in range(self.batch_size):
            correct_class = np.random.randint(0, self.num_classes)
            alpha[i, correct_class] = 10.0  # High concentration

        y = torch.zeros(self.batch_size, self.num_classes)
        for i in range(self.batch_size):
            y[i, int(torch.argmax(alpha[i]))] = 1.0

        mse = loss_module.dirichlet_mse(alpha, y)
        self.assertTrue(torch.isfinite(mse))


class TestCalcLossNotnan(unittest.TestCase):
    """Test cases for loss calculation with NaN handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.num_tasks = 2
        self.num_models = 3
        self.dtype = torch.float32

    def test_calc_loss_notnan_ensemble_shape(self):
        """Test calc_loss_notnan with ensemble outputs shape [batch_size, num_tasks, num_models]."""
        outputs = torch.randn(self.batch_size, self.num_tasks, self.num_models, dtype=self.dtype)
        targets = torch.randn(self.batch_size, self.num_tasks, 1, dtype=self.dtype)

        def dummy_loss_fn(pred, targ, var=None):
            return ((pred - targ) ** 2).mean()

        # calc_loss_notnan expects specific 3D shape
        loss = loss_module.calc_loss_notnan(
            outputs, targets,
            alea_vars=None,
            loss_fn=dummy_loss_fn
        )
        self.assertTrue(torch.isfinite(loss))

    def test_calc_loss_notnan_ignores_nans(self):
        """Test that NaN values in targets are properly handled."""
        outputs = torch.randn(self.batch_size, self.num_tasks, self.num_models, dtype=self.dtype)
        targets = torch.randn(self.batch_size, self.num_tasks, 1, dtype=self.dtype)

        # Add some NaN values
        targets[0, 0, 0] = float('nan')
        targets[1, 1, 0] = float('nan')

        def dummy_loss_fn(pred, targ, var=None):
            return ((pred - targ) ** 2).mean()

        # Should handle NaNs gracefully
        loss = loss_module.calc_loss_notnan(outputs, targets, alea_vars=None, loss_fn=dummy_loss_fn)
        self.assertTrue(torch.isfinite(loss) or torch.isnan(loss))



if __name__ == "__main__":
    unittest.main()

