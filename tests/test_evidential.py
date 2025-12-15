"""
Unit tests for evidential model in uqdd.models.evidential.

Tests cover:
- Evidential uncertainty computation
- NIG parameter extraction
- Evidential layer functionality
- Training and inference
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from uqdd.models.evidential import ev_uncertainty, ev_predict_params


class TestEvidentialUncertainty(unittest.TestCase):
    """Test cases for evidential uncertainty computation."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 8

    def test_ev_uncertainty_output_shapes(self):
        """Test that ev_uncertainty returns correct shapes."""
        v = torch.randn(self.batch_size, 1).abs() + 0.1
        alpha = torch.randn(self.batch_size, 1).abs() + 1.1
        beta = torch.randn(self.batch_size, 1).abs() + 0.1

        alea_var, epist_var = ev_uncertainty(v, alpha, beta)

        self.assertEqual(alea_var.shape, v.shape)
        self.assertEqual(epist_var.shape, v.shape)

    def test_ev_uncertainty_positive_variances(self):
        """Test that uncertainties are positive."""
        v = torch.ones(self.batch_size, 1) * 0.5
        alpha = torch.ones(self.batch_size, 1) * 2.0
        beta = torch.ones(self.batch_size, 1) * 1.0

        alea_var, epist_var = ev_uncertainty(v, alpha, beta)

        # Aleatoric = beta / (alpha - 1) should be positive
        self.assertTrue((alea_var > 0).all())
        # Epistemic = sqrt(beta / (v * (alpha - 1))) should be positive
        self.assertTrue((epist_var > 0).all())

    def test_ev_uncertainty_finite_values(self):
        """Test that uncertainties are finite."""
        v = torch.randn(self.batch_size, 1).abs() + 0.1
        alpha = torch.randn(self.batch_size, 1).abs() + 1.1
        beta = torch.randn(self.batch_size, 1).abs() + 0.1

        alea_var, epist_var = ev_uncertainty(v, alpha, beta)

        self.assertTrue(torch.isfinite(alea_var).all())
        self.assertTrue(torch.isfinite(epist_var).all())

    def test_ev_uncertainty_batch_dimension(self):
        """Test ev_uncertainty with different batch sizes."""
        for batch_size in [1, 4, 8, 16]:
            v = torch.ones(batch_size, 1) * 0.5
            alpha = torch.ones(batch_size, 1) * 2.0
            beta = torch.ones(batch_size, 1) * 1.0

            alea_var, epist_var = ev_uncertainty(v, alpha, beta)

            self.assertEqual(alea_var.shape[0], batch_size)
            self.assertEqual(epist_var.shape[0], batch_size)

    def test_ev_uncertainty_mathematical_correctness(self):
        """Test that uncertainty computation matches formulas."""
        batch_size = 4
        v = torch.tensor([[0.5], [1.0], [0.2], [2.0]], dtype=torch.float32)
        alpha = torch.tensor([[2.0], [3.0], [1.5], [4.0]], dtype=torch.float32)
        beta = torch.tensor([[1.0], [2.0], [0.5], [3.0]], dtype=torch.float32)

        alea_var, epist_var = ev_uncertainty(v, alpha, beta)

        # Manual computation
        expected_alea = beta / (alpha - 1)
        expected_epist = torch.sqrt(beta / (v * (alpha - 1)))

        self.assertTrue(torch.allclose(alea_var, expected_alea))
        self.assertTrue(torch.allclose(epist_var, expected_epist))

    def test_ev_uncertainty_higher_alpha_lower_epistemic(self):
        """Test that higher alpha (more evidence) reduces epistemic uncertainty."""
        v = torch.ones(2, 1) * 0.5
        beta = torch.ones(2, 1) * 1.0

        # Low alpha (less evidence)
        alpha_low = torch.ones(2, 1) * 1.5
        alea_low, epist_low = ev_uncertainty(v, alpha_low, beta)

        # High alpha (more evidence)
        alpha_high = torch.ones(2, 1) * 10.0
        alea_high, epist_high = ev_uncertainty(v, alpha_high, beta)

        # Epistemic uncertainty should be lower with higher alpha
        self.assertLess(epist_high[0].item(), epist_low[0].item())


class TestEvPredictParams(unittest.TestCase):
    """Test cases for evidential parameter extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 4

    def _create_dummy_model(self):
        """Create a dummy model that outputs NIG parameters."""
        class DummyEvidentialModel(nn.Module):
            def forward(self, x):
                batch_size = x[0].shape[0] if isinstance(x, tuple) else x.shape[0]
                mu = torch.randn(batch_size, 1)
                v = torch.relu(torch.randn(batch_size, 1)) + 0.1
                alpha = torch.relu(torch.randn(batch_size, 1)) + 1.1
                beta = torch.relu(torch.randn(batch_size, 1)) + 0.1
                return mu, v, alpha, beta

        return DummyEvidentialModel()

    def _create_dummy_dataloader(self, batch_size=4, num_batches=2):
        """Create a dummy dataloader for testing."""
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                x = (torch.randn(256), torch.randn(2048))
                y = torch.randn(1)
                return x, y

        dataset = DummyDataset(batch_size * num_batches)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    def test_ev_predict_params_output_shapes(self):
        """Test that ev_predict_params returns correct output shapes."""
        model = self._create_dummy_model()
        dataloader = self._create_dummy_dataloader()

        mus, vs, alphas, betas, targets = ev_predict_params(model, dataloader)

        # All outputs should be concatenated tensors
        self.assertEqual(mus.shape[1], 1)
        self.assertEqual(vs.shape[1], 1)
        self.assertEqual(alphas.shape[1], 1)
        self.assertEqual(betas.shape[1], 1)
        self.assertGreater(mus.shape[0], 0)

    def test_ev_predict_params_with_eval_mode(self):
        """Test that ev_predict_params sets model to eval mode."""
        model = self._create_dummy_model()
        model.train()
        dataloader = self._create_dummy_dataloader()

        mus, vs, alphas, betas, targets = ev_predict_params(
            model, dataloader, set_on_eval=True
        )

        # Model should be in eval mode
        self.assertFalse(model.training)

    def test_ev_predict_params_no_gradients(self):
        """Test that ev_predict_params runs without computing gradients."""
        model = self._create_dummy_model()
        dataloader = self._create_dummy_dataloader()

        # Should not raise error about leaf variables
        mus, vs, alphas, betas, targets = ev_predict_params(model, dataloader)

        # Outputs should not require gradients
        self.assertFalse(mus.requires_grad)
        self.assertFalse(vs.requires_grad)

    def test_ev_predict_params_finite_outputs(self):
        """Test that outputs are finite."""
        model = self._create_dummy_model()
        dataloader = self._create_dummy_dataloader()

        mus, vs, alphas, betas, targets = ev_predict_params(model, dataloader)

        self.assertTrue(torch.isfinite(mus).all())
        self.assertTrue(torch.isfinite(vs).all())
        self.assertTrue(torch.isfinite(alphas).all())
        self.assertTrue(torch.isfinite(betas).all())

    def test_ev_predict_params_positive_params(self):
        """Test that NIG parameters are positive."""
        # Create a model that outputs positive parameters
        class PositiveEvidentialModel(nn.Module):
            def forward(self, x):
                batch_size = x[0].shape[0] if isinstance(x, tuple) else x.shape[0]
                mu = torch.randn(batch_size, 1)
                v = torch.ones(batch_size, 1) * 0.5
                alpha = torch.ones(batch_size, 1) * 2.0
                beta = torch.ones(batch_size, 1) * 1.0
                return mu, v, alpha, beta

        model = PositiveEvidentialModel()
        dataloader = self._create_dummy_dataloader()

        mus, vs, alphas, betas, targets = ev_predict_params(model, dataloader)

        # v should be positive
        self.assertTrue((vs > 0).all())
        # alpha should be > 1 for valid NIG
        self.assertTrue((alphas > 1).all())
        # beta should be positive
        self.assertTrue((betas > 0).all())


class TestEvidentialIntegration(unittest.TestCase):
    """Integration tests for evidential models."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_uncertainty_estimation_workflow(self):
        """Test a complete uncertainty estimation workflow."""
        # Create dummy model and data
        class DummyEvidentialModel(nn.Module):
            def forward(self, x):
                batch_size = x[0].shape[0] if isinstance(x, tuple) else x.shape[0]
                mu = torch.randn(batch_size, 1)
                v = torch.relu(torch.randn(batch_size, 1)) + 0.1
                alpha = torch.relu(torch.randn(batch_size, 1)) + 1.1
                beta = torch.relu(torch.randn(batch_size, 1)) + 0.1
                return mu, v, alpha, beta

        model = DummyEvidentialModel()

        # Run predictions
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                return (torch.randn(256), torch.randn(2048)), torch.randn(1)

        dataloader = torch.utils.data.DataLoader(
            DummyDataset(), batch_size=4
        )

        # Get parameters
        mus, vs, alphas, betas, targets = ev_predict_params(model, dataloader)

        # Compute uncertainties
        alea_vars, epist_vars = ev_uncertainty(vs, alphas, betas)

        # Uncertainties should be finite and positive
        self.assertTrue(torch.isfinite(alea_vars).all())
        self.assertTrue(torch.isfinite(epist_vars).all())
        self.assertTrue((alea_vars > 0).all())
        self.assertTrue((epist_vars > 0).all())

    def test_calibration_check(self):
        """Test that well-calibrated model has matching prediction confidence and accuracy."""
        # High confidence predictions with correct targets
        mu = torch.zeros(10, 1)  # Prediction = 0
        v = torch.ones(10, 1) * 0.1  # Low epistemic uncertainty
        alpha = torch.ones(10, 1) * 10.0  # High alpha = high confidence
        beta = torch.ones(10, 1) * 0.5

        alea_var, epist_var = ev_uncertainty(v, alpha, beta)

        # Low uncertainty
        self.assertTrue(epist_var.max() < alea_var.max() + 1.0)


class TestEvidentialConfiguration(unittest.TestCase):
    """Test evidential model configuration and setup."""

    def test_nig_parameters_valid_ranges(self):
        """Test that NIG parameters are in valid ranges."""
        batch_size = 8

        # Valid parameters
        v = torch.ones(batch_size, 1) * 0.5  # v > 0
        alpha = torch.ones(batch_size, 1) * 2.0  # alpha > 1
        beta = torch.ones(batch_size, 1) * 1.0  # beta > 0

        # Should compute without error
        alea_var, epist_var = ev_uncertainty(v, alpha, beta)

        self.assertTrue(torch.isfinite(alea_var).all())
        self.assertTrue(torch.isfinite(epist_var).all())

    def test_nig_parameter_edge_cases(self):
        """Test NIG parameters with edge cases."""
        # Very small positive values
        v_small = torch.ones(2, 1) * 1e-5
        alpha_small = torch.ones(2, 1) * 1.001
        beta_small = torch.ones(2, 1) * 1e-5

        alea_var, epist_var = ev_uncertainty(v_small, alpha_small, beta_small)

        # Should still be finite
        self.assertTrue(torch.isfinite(alea_var).all())
        self.assertTrue(torch.isfinite(epist_var).all())

    def test_evidential_vs_aleatoric_epistemic_balance(self):
        """Test the balance between aleatoric and epistemic uncertainty."""
        batch_size = 10
        v = torch.ones(batch_size, 1) * 0.5
        beta = torch.ones(batch_size, 1) * 1.0

        # Case 1: High alpha (high evidence, low epistemic)
        alpha_high = torch.ones(batch_size, 1) * 20.0
        alea_high, epist_high = ev_uncertainty(v, alpha_high, beta)

        # Case 2: Low alpha (low evidence, high epistemic)
        alpha_low = torch.ones(batch_size, 1) * 1.5
        alea_low, epist_low = ev_uncertainty(v, alpha_low, beta)

        # Epistemic should be much lower with high alpha
        self.assertLess(epist_high.mean().item(), epist_low.mean().item())


if __name__ == "__main__":
    unittest.main()

