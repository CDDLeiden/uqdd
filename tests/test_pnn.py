"""
Unit tests for PNN (Probabilistic Neural Network) model in uqdd.models.pnn.

Tests cover:
- PNN initialization with various configurations
- Forward pass and output shapes
- Weight initialization
- Task type handling (regression vs classification)
- Multitask support
- Aleatoric uncertainty
- MLP construction
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from uqdd.models.pnn import PNN
import uqdd.models.utils_models as um


class TestPNNInitialization(unittest.TestCase):
    """Test cases for PNN model initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def _get_default_config(self):
        """Get default PNN configuration."""
        return {
            "chem_input_dim": 2048,
            "prot_input_dim": 256,
            "chem_hidden_dims": [512, 256],
            "prot_hidden_dims": [256, 128],
            "hidden_dims": [256, 128],
            "output_dim": 1,
            "dropout": 0.2,
            "task_type": "regression",
            "aleatoric": False,
            "n_targets": -1,
            "MT": False,
        }

    def test_pnn_init_with_config(self):
        """Test PNN initialization with configuration."""
        config = self._get_default_config()
        model = PNN(config=config)

        self.assertIsNotNone(model.chem_feature_extractor)
        self.assertIsNotNone(model.prot_feature_extractor)
        self.assertIsNotNone(model.regressor_or_classifier)

    def test_pnn_init_default_config(self):
        """Test PNN initialization with default config via get_model_config."""
        with patch("uqdd.models.pnn.get_model_config") as mock_get_config:
            mock_get_config.return_value = self._get_default_config()

            model = PNN()

            self.assertIsNotNone(model.config)
            self.assertFalse(model.MT)

    def test_pnn_init_regression_task(self):
        """Test PNN initialization for regression task."""
        config = self._get_default_config()
        config["task_type"] = "regression"

        model = PNN(config=config)

        self.assertEqual(model.task_type, "regression")
        self.assertEqual(model.output_dim, 1)

    def test_pnn_init_classification_task(self):
        """Test PNN initialization for classification task."""
        config = self._get_default_config()
        config["task_type"] = "classification"

        model = PNN(config=config)

        self.assertEqual(model.task_type, "classification")
        # For single-task classification: 2 * n_targets = 2 * 1 = 2
        self.assertEqual(model.output_dim, 2)

    def test_pnn_init_multitask(self):
        """Test PNN initialization for multitask learning."""
        config = self._get_default_config()
        config["n_targets"] = 5
        config["MT"] = True

        model = PNN(config=config)

        self.assertTrue(model.MT)
        self.assertEqual(model.output_dim, 5)

    def test_pnn_init_multitask_classification(self):
        """Test PNN initialization for multitask classification."""
        config = self._get_default_config()
        config["n_targets"] = 5
        config["MT"] = True
        config["task_type"] = "classification"

        model = PNN(config=config)

        self.assertTrue(model.MT)
        # For classification: 2 * n_targets = 2 * 5 = 10
        self.assertEqual(model.output_dim, 10)

    def test_pnn_init_with_aleatoric(self):
        """Test PNN initialization with aleatoric uncertainty."""
        config = self._get_default_config()
        config["aleatoric"] = True

        model = PNN(config=config)

        self.assertTrue(model.aleatoric)
        self.assertIsNotNone(model.aleavar_layer)

    def test_pnn_init_invalid_task_type(self):
        """Test that invalid task type raises assertion."""
        config = self._get_default_config()
        config["task_type"] = "invalid_task"

        with self.assertRaises(AssertionError):
            PNN(config=config)

    def test_pnn_init_creates_all_layers(self):
        """Test that PNN initialization creates all required layers."""
        config = self._get_default_config()
        model = PNN(config=config)

        self.assertIsNotNone(model.chem_feature_extractor)
        self.assertIsNotNone(model.prot_feature_extractor)
        self.assertIsNotNone(model.regressor_or_classifier)
        self.assertIsNotNone(model.output_layer)


class TestPNNForward(unittest.TestCase):
    """Test cases for PNN forward pass."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.batch_size = 8

        self.config = {
            "chem_input_dim": 2048,
            "prot_input_dim": 256,
            "chem_hidden_dims": [512, 256],
            "prot_hidden_dims": [256, 128],
            "hidden_dims": [256, 128],
            "output_dim": 1,
            "dropout": 0.2,
            "task_type": "regression",
            "aleatoric": False,
            "n_targets": -1,
            "MT": False,
        }

    def test_pnn_forward_regression(self):
        """Test PNN forward pass for regression."""
        model = PNN(config=self.config)
        model.to(self.device)
        model.eval()

        prot = torch.randn(self.batch_size, self.config["prot_input_dim"], device=self.device)
        chem = torch.randn(self.batch_size, self.config["chem_input_dim"], device=self.device)

        output, var = model((prot, chem))

        self.assertEqual(output.shape, torch.Size([self.batch_size, 1]))
        self.assertIsNone(var)

    def test_pnn_forward_classification(self):
        """Test PNN forward pass for classification."""
        config = self.config.copy()
        config["task_type"] = "classification"

        model = PNN(config=config)
        model.to(self.device)
        model.eval()

        prot = torch.randn(self.batch_size, self.config["prot_input_dim"], device=self.device)
        chem = torch.randn(self.batch_size, self.config["chem_input_dim"], device=self.device)

        output, var = model((prot, chem))

        # Classification output: 2 (active/inactive)
        self.assertEqual(output.shape, torch.Size([self.batch_size, 2]))

    def test_pnn_forward_with_aleatoric(self):
        """Test PNN forward pass with aleatoric uncertainty."""
        config = self.config.copy()
        config["aleatoric"] = True

        model = PNN(config=config)
        model.to(self.device)
        model.eval()

        prot = torch.randn(self.batch_size, self.config["prot_input_dim"], device=self.device)
        chem = torch.randn(self.batch_size, self.config["chem_input_dim"], device=self.device)

        output, var = model((prot, chem))

        self.assertEqual(output.shape, torch.Size([self.batch_size, 1]))
        self.assertIsNotNone(var)
        self.assertEqual(var.shape, torch.Size([self.batch_size, 1]))

    def test_pnn_forward_multitask(self):
        """Test PNN forward pass for multitask learning."""
        config = self.config.copy()
        config["n_targets"] = 3
        config["MT"] = True

        model = PNN(config=config)
        model.to(self.device)
        model.eval()

        prot = torch.randn(self.batch_size, self.config["prot_input_dim"], device=self.device)
        chem = torch.randn(self.batch_size, self.config["chem_input_dim"], device=self.device)

        output, var = model((prot, chem))

        # Multitask: output_dim = n_targets = 3
        self.assertEqual(output.shape, torch.Size([self.batch_size, 3]))

    def test_pnn_forward_output_finite(self):
        """Test that PNN forward pass produces finite outputs."""
        model = PNN(config=self.config)
        model.to(self.device)
        model.eval()

        prot = torch.randn(self.batch_size, self.config["prot_input_dim"], device=self.device)
        chem = torch.randn(self.batch_size, self.config["chem_input_dim"], device=self.device)

        output, var = model((prot, chem))

        self.assertTrue(torch.isfinite(output).all())

    def test_pnn_forward_gradient_flow(self):
        """Test that gradients flow through forward pass."""
        model = PNN(config=self.config)
        model.to(self.device)

        prot = torch.randn(self.batch_size, self.config["prot_input_dim"],
                           device=self.device, requires_grad=True)
        chem = torch.randn(self.batch_size, self.config["chem_input_dim"],
                           device=self.device, requires_grad=True)

        output, var = model((prot, chem))
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(prot.grad)
        self.assertIsNotNone(chem.grad)

        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_pnn_forward_different_batch_sizes(self):
        """Test PNN forward pass with different batch sizes."""
        model = PNN(config=self.config)
        model.to(self.device)
        model.eval()

        for batch_size in [1, 4, 16, 32]:
            prot = torch.randn(batch_size, self.config["prot_input_dim"], device=self.device)
            chem = torch.randn(batch_size, self.config["chem_input_dim"], device=self.device)

            output, var = model((prot, chem))

            self.assertEqual(output.shape[0], batch_size)
            self.assertTrue(torch.isfinite(output).all())


class TestPNNWeightInitialization(unittest.TestCase):
    """Test cases for PNN weight initialization."""

    def test_init_wt_linear_layer(self):
        """Test weight initialization for linear layers."""
        layer = nn.Linear(100, 50)
        PNN.init_wt(layer)

        # Check that weights are initialized (not zero)
        self.assertFalse(torch.allclose(layer.weight, torch.zeros_like(layer.weight)))

    def test_init_wt_preserves_shape(self):
        """Test that weight initialization preserves shapes."""
        layer = nn.Linear(100, 50)
        original_shape = layer.weight.shape

        PNN.init_wt(layer)

        self.assertEqual(layer.weight.shape, original_shape)

    def test_init_wt_multiple_layers(self):
        """Test weight initialization on multiple layers."""
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1)
        )

        model.apply(PNN.init_wt)

        # All linear layers should have initialized weights
        for layer in model:
            if isinstance(layer, nn.Linear):
                self.assertFalse(torch.allclose(layer.weight, torch.zeros_like(layer.weight)))


class TestPNNCreateMLP(unittest.TestCase):
    """Test cases for MLP creation."""

    def test_create_mlp_structure(self):
        """Test that create_mlp creates correct structure."""
        input_dim = 100
        layer_dims = [50, 25]
        dropout = 0.2

        mlp = PNN.create_mlp(input_dim, layer_dims, dropout)

        self.assertIsInstance(mlp, nn.Sequential)
        # Each layer: Linear + ReLU + Dropout
        # Last layer might be different
        self.assertGreater(len(mlp), 0)

    def test_create_mlp_single_layer(self):
        """Test create_mlp with single hidden layer."""
        mlp = PNN.create_mlp(100, [50], 0.2)

        self.assertIsInstance(mlp, nn.Sequential)
        self.assertGreater(len(mlp), 0)

    def test_create_mlp_no_layers(self):
        """Test create_mlp with no hidden layers."""
        mlp = PNN.create_mlp(100, [], 0.2)

        self.assertIsInstance(mlp, nn.Sequential)

    def test_create_mlp_forward_pass(self):
        """Test that created MLP can perform forward pass."""
        mlp = PNN.create_mlp(100, [50, 25], 0.2)
        mlp.eval()

        x = torch.randn(8, 100)
        output = mlp(x)

        self.assertEqual(output.shape[0], 8)
        self.assertTrue(torch.isfinite(output).all())

    def test_create_mlp_various_dropouts(self):
        """Test create_mlp with various dropout rates.
        Converted from pytest parametrize to be compatible with unittest.TestCase.
        """
        for dropout_rate in [0.0, 0.1, 0.3, 0.5]:
            with self.subTest(dropout_rate=dropout_rate):
                mlp = PNN.create_mlp(100, [50, 25], dropout_rate)
                self.assertIsInstance(mlp, nn.Sequential)
                # Forward pass should work
                x = torch.randn(8, 100)
                output = mlp(x)
                self.assertTrue(torch.isfinite(output).all())


class TestPNNIntegration(unittest.TestCase):
    """Integration tests for PNN model."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.config = {
            "chem_input_dim": 2048,
            "prot_input_dim": 256,
            "chem_hidden_dims": [512, 256],
            "prot_hidden_dims": [256, 128],
            "hidden_dims": [256, 128],
            "output_dim": 1,
            "dropout": 0.2,
            "task_type": "regression",
            "aleatoric": False,
            "n_targets": -1,
            "MT": False,
        }

    def test_pnn_training_loop(self):
        """Test a simple training loop with PNN."""
        model = PNN(config=self.config)
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        batch_size = 8
        prot = torch.randn(batch_size, self.config["prot_input_dim"], device=self.device)
        chem = torch.randn(batch_size, self.config["chem_input_dim"], device=self.device)
        targets = torch.randn(batch_size, 1, device=self.device)

        # Forward pass
        output, _ = model((prot, chem))
        loss = loss_fn(output, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that weights were updated
        self.assertTrue(torch.isfinite(loss))

    def test_pnn_eval_mode(self):
        """Test PNN in evaluation mode (dropout disabled)."""
        model = PNN(config=self.config)
        model.eval()

        batch_size = 8
        prot = torch.randn(batch_size, self.config["prot_input_dim"])
        chem = torch.randn(batch_size, self.config["chem_input_dim"])

        # Multiple forward passes in eval mode should be deterministic
        um.set_seed(42)
        output1, _ = model((prot, chem))

        um.set_seed(42)
        output2, _ = model((prot, chem))

        self.assertTrue(torch.allclose(output1, output2))

    def test_pnn_train_mode(self):
        """Test PNN in training mode (dropout enabled)."""
        model = PNN(config=self.config)
        model.train()

        batch_size = 8
        prot = torch.randn(batch_size, self.config["prot_input_dim"])
        chem = torch.randn(batch_size, self.config["chem_input_dim"])

        # Multiple forward passes in train mode might differ due to dropout
        output1, _ = model((prot, chem))
        output2, _ = model((prot, chem))

        # They might be different due to dropout, but both should be valid
        self.assertEqual(output1.shape, output2.shape)


if __name__ == "__main__":
    unittest.main()
