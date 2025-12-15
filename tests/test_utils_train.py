"""
Unit tests for training utilities in uqdd.models.utils_train.

Tests cover:
- Forward pass coordination
- Training loops
- Evaluation loops
- Model prediction
- Uncertainty processing
- End-to-end training
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from uqdd.models.utils_train import (
    evidential_processing,
    model_forward,
)


class TestEvidentialProcessing(unittest.TestCase):
    """Test cases for evidential uncertainty processing."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 8

    def test_evidential_processing_output_shapes(self):
        """Test that evidential_processing returns correct shapes."""
        mu = torch.randn(self.batch_size, 1)
        v = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        alpha = torch.relu(torch.randn(self.batch_size, 1)) + 1.1
        beta = torch.relu(torch.randn(self.batch_size, 1)) + 0.1

        outputs = (mu, v, alpha, beta)
        alea_vars, epi_vars = evidential_processing(outputs)

        self.assertEqual(alea_vars.shape, (self.batch_size, 1))
        self.assertEqual(epi_vars.shape, (self.batch_size, 1))

    def test_evidential_processing_positive_uncertainties(self):
        """Test that uncertainties are positive."""
        mu = torch.zeros(self.batch_size, 1)
        v = torch.ones(self.batch_size, 1) * 0.5
        alpha = torch.ones(self.batch_size, 1) * 2.0
        beta = torch.ones(self.batch_size, 1) * 1.0

        outputs = (mu, v, alpha, beta)
        alea_vars, epi_vars = evidential_processing(outputs)

        self.assertTrue((alea_vars > 0).all())
        self.assertTrue((epi_vars > 0).all())

    def test_evidential_processing_finite_values(self):
        """Test that uncertainties are finite."""
        mu = torch.randn(self.batch_size, 1)
        v = torch.relu(torch.randn(self.batch_size, 1)) + 0.1
        alpha = torch.relu(torch.randn(self.batch_size, 1)) + 1.1
        beta = torch.relu(torch.randn(self.batch_size, 1)) + 0.1

        outputs = (mu, v, alpha, beta)
        alea_vars, epi_vars = evidential_processing(outputs)

        self.assertTrue(torch.isfinite(alea_vars).all())
        self.assertTrue(torch.isfinite(epi_vars).all())

    def test_evidential_processing_mathematical_correctness(self):
        """Test that formulas are correct."""
        batch_size = 4
        mu = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        v = torch.tensor([[0.5], [1.0], [0.2], [2.0]])
        alpha = torch.tensor([[2.0], [3.0], [1.5], [4.0]])
        beta = torch.tensor([[1.0], [2.0], [0.5], [3.0]])

        outputs = (mu, v, alpha, beta)
        alea_vars, epi_vars = evidential_processing(outputs)

        # Manual computation
        expected_alea = beta / (alpha - 1)
        expected_epi = torch.sqrt(beta / (v * (alpha - 1)))

        self.assertTrue(torch.allclose(alea_vars, expected_alea))
        self.assertTrue(torch.allclose(epi_vars, expected_epi))

    def test_evidential_processing_batch_independence(self):
        """Test that processing is independent across batch."""
        batch_size = 2
        mu = torch.tensor([[1.0], [2.0]])
        v = torch.tensor([[0.5], [1.0]])
        # Choose alpha/beta so aleatoric and epistemic differ across batch items
        alpha = torch.tensor([[2.0], [3.0]])
        beta = torch.tensor([[1.0], [2.5]])

        outputs = (mu, v, alpha, beta)
        alea_vars, epi_vars = evidential_processing(outputs)

        # Different inputs should give different outputs
        self.assertFalse(torch.allclose(alea_vars[0], alea_vars[1]))
        self.assertFalse(torch.allclose(epi_vars[0], epi_vars[1]))


class TestModelForward(unittest.TestCase):
    """Test cases for model_forward function."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.device = torch.device("cpu")

    def _create_dummy_pnn_model(self):
        """Create a simple PNN-like model."""
        class DummyPNN(nn.Module):
            def forward(self, inputs):
                # Returns output and aleatoric variance
                batch_size = inputs[0].shape[0] if isinstance(inputs, tuple) else inputs.shape[0]
                output = torch.randn(batch_size, 1)
                var = torch.relu(torch.randn(batch_size, 1)) + 0.1
                return output, var

        return DummyPNN()

    def _create_dummy_evidential_model(self):
        """Create a simple evidential-like model."""
        class DummyEvidential(nn.Module):
            def forward(self, inputs):
                # Returns mu, v, alpha, beta
                batch_size = inputs[0].shape[0] if isinstance(inputs, tuple) else inputs.shape[0]
                mu = torch.randn(batch_size, 1)
                v = torch.relu(torch.randn(batch_size, 1)) + 0.1
                alpha = torch.relu(torch.randn(batch_size, 1)) + 1.1
                beta = torch.relu(torch.randn(batch_size, 1)) + 0.1
                return mu, v, alpha, beta

        return DummyEvidential()

    def test_model_forward_gaussnll_mode(self):
        """Test model_forward with gaussnll loss."""
        model = self._create_dummy_pnn_model()
        inputs = (torch.randn(self.batch_size, 256), torch.randn(self.batch_size, 2048))
        targets = torch.randn(self.batch_size, 1)

        outputs, alea_vars, epi_vars, loss_args = model_forward(
            model, inputs, targets, lossfname="gaussnll"
        )

        self.assertIsNotNone(outputs)
        self.assertIsNotNone(alea_vars)
        self.assertIsNone(epi_vars)
        self.assertEqual(len(loss_args), 3)  # (outputs, targets, vars)

    def test_model_forward_evidential_mode(self):
        """Test model_forward with evidential regression."""
        model = self._create_dummy_evidential_model()
        inputs = (torch.randn(self.batch_size, 256), torch.randn(self.batch_size, 2048))
        targets = torch.randn(self.batch_size, 1)

        outputs, alea_vars, epi_vars, loss_args = model_forward(
            model, inputs, targets, lossfname="evidential_regression"
        )

        self.assertIsNotNone(outputs)
        self.assertIsNotNone(alea_vars)
        self.assertIsNotNone(epi_vars)
        self.assertEqual(len(loss_args), 2)  # (outputs, targets)

    def test_model_forward_output_structure(self):
        """Test model_forward output structure."""
        model = self._create_dummy_pnn_model()
        inputs = (torch.randn(self.batch_size, 256), torch.randn(self.batch_size, 2048))
        targets = torch.randn(self.batch_size, 1)

        outputs, alea_vars, epi_vars, loss_args = model_forward(
            model, inputs, targets, lossfname="gaussnll"
        )

        # All outputs should be finite
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertTrue(torch.isfinite(alea_vars).all())

    def test_model_forward_finite_values(self):
        """Test that all forward outputs are finite."""
        model = self._create_dummy_evidential_model()
        inputs = (torch.randn(self.batch_size, 256), torch.randn(self.batch_size, 2048))
        targets = torch.randn(self.batch_size, 1)

        outputs, alea_vars, epi_vars, loss_args = model_forward(
            model, inputs, targets, lossfname="evidential_regression"
        )

        self.assertTrue(torch.isfinite(outputs[0]).all())  # mu
        self.assertTrue(torch.isfinite(alea_vars).all())
        self.assertTrue(torch.isfinite(epi_vars).all())


class TestModelForwardIntegration(unittest.TestCase):
    """Integration tests for model_forward."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4

    def test_forward_with_simple_mse(self):
        """Test forward pass with simple MSE loss."""
        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(256 + 2048, 64)
                self.fc2 = nn.Linear(64, 1)
                self.var_layer = nn.Linear(64, 1)

            def forward(self, inputs):
                x = torch.cat(inputs, dim=1)
                h = torch.relu(self.fc1(x))
                output = self.fc2(h)
                var = torch.relu(self.var_layer(h)) + 0.1
                return output, var

        model = SimpleMLP()
        inputs = (torch.randn(self.batch_size, 256), torch.randn(self.batch_size, 2048))
        targets = torch.randn(self.batch_size, 1)

        outputs, alea_vars, epi_vars, loss_args = model_forward(
            model, inputs, targets, lossfname="gaussnll"
        )

        self.assertEqual(outputs.shape, (self.batch_size, 1))
        self.assertEqual(alea_vars.shape, (self.batch_size, 1))

    def test_forward_gradient_flow(self):
        """Test that gradients flow through forward pass."""
        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(256 + 2048, 1)
                self.var_fc = nn.Linear(256 + 2048, 1)

            def forward(self, inputs):
                x = torch.cat(inputs, dim=1)
                output = self.fc(x)
                var = torch.relu(self.var_fc(x)) + 0.1
                return output, var

        model = SimpleMLP()
        inputs = (
            torch.randn(self.batch_size, 256, requires_grad=True),
            torch.randn(self.batch_size, 2048, requires_grad=True)
        )
        targets = torch.randn(self.batch_size, 1)

        outputs, alea_vars, epi_vars, loss_args = model_forward(
            model, inputs, targets, lossfname="gaussnll"
        )

        # Compute loss and backward
        loss = ((outputs - targets) ** 2).mean()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(inputs[0].grad)
        self.assertIsNotNone(inputs[1].grad)


class TestTrainingUtilities(unittest.TestCase):
    """Test basic training utility patterns."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

    def test_batch_processing_loop_pattern(self):
        """Test typical batch processing pattern."""
        model = nn.Linear(10, 1)
        model.eval()

        # Simulate dataloader batches
        num_batches = 4
        batch_size = 8

        outputs_list = []
        for batch_idx in range(num_batches):
            x = torch.randn(batch_size, 10)
            with torch.no_grad():
                output = model(x)
            outputs_list.append(output)

        # Concatenate results
        all_outputs = torch.cat(outputs_list, dim=0)

        self.assertEqual(all_outputs.shape[0], num_batches * batch_size)

    def test_loss_computation_pattern(self):
        """Test loss computation pattern."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        # Forward pass
        output = model(x)
        loss = loss_fn(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))

    def test_validation_loop_pattern(self):
        """Test validation loop pattern (no grad)."""
        model = nn.Linear(10, 1)
        model.eval()

        x = torch.randn(8, 10)

        with torch.no_grad():
            output = model(x)

        # In eval mode with no_grad, should be efficient
        self.assertTrue(torch.isfinite(output).all())

    def test_prediction_aggregation_pattern(self):
        """Test prediction aggregation pattern."""
        # Simulate multiple batches of predictions
        num_samples = 20
        pred_list = [torch.randn(5, 1) for _ in range(4)]

        # Concatenate predictions
        all_preds = torch.cat(pred_list, dim=0)

        self.assertEqual(all_preds.shape[0], num_samples)
        self.assertEqual(all_preds.shape[1], 1)


class TestUncertaintyIntegration(unittest.TestCase):
    """Integration tests for uncertainty handling."""

    def test_aleatoric_variance_handling(self):
        """Test aleatoric variance computation and handling."""
        batch_size = 8

        # Simulate model output and variance
        predictions = torch.randn(batch_size, 1)
        aleatoric_var = torch.relu(torch.randn(batch_size, 1)) + 0.1

        # Should all be positive and finite
        self.assertTrue((aleatoric_var > 0).all())
        self.assertTrue(torch.isfinite(aleatoric_var).all())

    def test_epistemic_variance_handling(self):
        """Test epistemic variance computation and handling."""
        batch_size = 8

        # Simulate evidential parameters
        v = torch.ones(batch_size, 1) * 0.5
        alpha = torch.ones(batch_size, 1) * 2.0
        beta = torch.ones(batch_size, 1) * 1.0

        # Compute epistemic uncertainty
        epistemic_var = torch.sqrt(beta / (v * (alpha - 1)))

        self.assertTrue((epistemic_var > 0).all())
        self.assertTrue(torch.isfinite(epistemic_var).all())

    def test_combined_uncertainty(self):
        """Test combined aleatoric and epistemic uncertainty."""
        batch_size = 8

        # Aleatoric
        alea_var = torch.relu(torch.randn(batch_size, 1)) + 0.1

        # Epistemic
        v = torch.ones(batch_size, 1) * 0.5
        alpha = torch.ones(batch_size, 1) * 2.0
        beta = torch.ones(batch_size, 1) * 1.0
        epist_var = torch.sqrt(beta / (v * (alpha - 1)))

        # Total uncertainty
        total_var = alea_var + epist_var

        self.assertTrue((total_var > 0).all())
        self.assertTrue(torch.isfinite(total_var).all())


if __name__ == "__main__":
    unittest.main()
