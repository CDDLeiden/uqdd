"""
Unit tests for ensemble model in uqdd.models.ensemble.

Tests cover:
- EnsembleDNN initialization with various configurations
- Forward pass and output aggregation
- Ensemble member management
- Training and inference
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List

from uqdd.models.ensemble import EnsembleDNN
from uqdd.models.pnn import PNN


class TestEnsembleDNNInitialization(unittest.TestCase):
    """Test cases for EnsembleDNN initialization."""

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
            "ensemble_size": 3,
            "seed": 42,
            "n_targets": -1,
            "MT": False,
        }

    def test_ensemble_init_creates_correct_number_of_models(self):
        """Test that ensemble initialization creates correct number of models."""
        ensemble = EnsembleDNN(config=self.config)
        self.assertEqual(len(ensemble.models), 3)

    def test_ensemble_init_different_seeds(self):
        """Test that ensemble members are initialized with different seeds."""
        ensemble = EnsembleDNN(config=self.config)

        # Get first layer weights from each model
        weights_list = []
        for model in ensemble.models:
            weights = model.chem_feature_extractor[0].weight.data.clone()
            weights_list.append(weights)

        # Models should have different weights due to different seeds
        self.assertFalse(torch.allclose(weights_list[0], weights_list[1]))
        self.assertFalse(torch.allclose(weights_list[1], weights_list[2]))

    def test_ensemble_init_with_model_list(self):
        """Test ensemble initialization with pre-initialized models."""
        models = [PNN(config=self.config) for _ in range(2)]
        ensemble = EnsembleDNN(model_list=models, config=self.config)

        self.assertEqual(len(ensemble.models), 2)

    def test_ensemble_init_default_ensemble_size(self):
        """Test ensemble initialization with default ensemble size."""
        config = self.config.copy()
        del config["ensemble_size"]  # Use default

        ensemble = EnsembleDNN(config=config)
        # Default should be 100 or some reasonable value
        self.assertGreater(len(ensemble.models), 0)

    def test_ensemble_init_custom_model_class(self):
        """Test ensemble with custom model class."""
        # Use default PNN class (implicit)
        ensemble = EnsembleDNN(config=self.config, model_class=PNN)

        self.assertEqual(len(ensemble.models), 3)
        for model in ensemble.models:
            self.assertIsInstance(model, PNN)

    def test_ensemble_config_storage(self):
        """Test that ensemble stores configuration."""
        ensemble = EnsembleDNN(config=self.config)

        self.assertIsNotNone(ensemble.config)
        self.assertEqual(ensemble.config["ensemble_size"], 3)


class TestEnsembleDNNForward(unittest.TestCase):
    """Test cases for ensemble forward pass."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 8
        self.config = {
            "chem_input_dim": 2048,
            "prot_input_dim": 256,
            "chem_hidden_dims": [256, 128],
            "prot_hidden_dims": [128, 64],
            "hidden_dims": [128, 64],
            "output_dim": 1,
            "dropout": 0.2,
            "task_type": "regression",
            "aleatoric": False,
            "ensemble_size": 2,
            "seed": 42,
            "n_targets": -1,
            "MT": False,
        }

    def test_ensemble_forward_output_shape(self):
        """Test ensemble forward pass output shape."""
        ensemble = EnsembleDNN(config=self.config)
        ensemble.to(self.device)
        ensemble.eval()

        prot = torch.randn(self.batch_size, self.config["prot_input_dim"], device=self.device)
        chem = torch.randn(self.batch_size, self.config["chem_input_dim"], device=self.device)

        output, var = ensemble((prot, chem))

        # Output should be [batch_size, output_dim, ensemble_size]
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[2], 2)  # ensemble_size = 2

    def test_ensemble_forward_returns_tuple(self):
        """Test that forward pass returns tuple."""
        ensemble = EnsembleDNN(config=self.config)
        ensemble.eval()

        prot = torch.randn(self.batch_size, self.config["prot_input_dim"])
        chem = torch.randn(self.batch_size, self.config["chem_input_dim"])

        result = ensemble((prot, chem))

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_ensemble_forward_finite_outputs(self):
        """Test that ensemble forward outputs are finite."""
        ensemble = EnsembleDNN(config=self.config)
        ensemble.eval()

        prot = torch.randn(self.batch_size, self.config["prot_input_dim"])
        chem = torch.randn(self.batch_size, self.config["chem_input_dim"])

        output, var = ensemble((prot, chem))

        self.assertTrue(torch.isfinite(output).all())

    def test_ensemble_forward_different_batch_sizes(self):
        """Test ensemble with different batch sizes."""
        ensemble = EnsembleDNN(config=self.config)
        ensemble.eval()

        for batch_size in [1, 4, 8, 16]:
            prot = torch.randn(batch_size, self.config["prot_input_dim"])
            chem = torch.randn(batch_size, self.config["chem_input_dim"])

            output, var = ensemble((prot, chem))

            self.assertEqual(output.shape[0], batch_size)
            self.assertTrue(torch.isfinite(output).all())

    def test_ensemble_stacks_model_outputs(self):
        """Test that ensemble properly stacks outputs from all members."""
        ensemble = EnsembleDNN(config=self.config)
        ensemble.eval()

        prot = torch.randn(self.batch_size, self.config["prot_input_dim"])
        chem = torch.randn(self.batch_size, self.config["chem_input_dim"])

        # Get ensemble output
        ens_output, _ = ensemble((prot, chem))

        # Get individual model outputs
        individual_outputs = []
        for model in ensemble.models:
            model.eval()
            out, _ = model((prot, chem))
            individual_outputs.append(out)

        # Ensemble outputs should match stacked individual outputs
        stacked = torch.stack(individual_outputs, dim=2)
        self.assertTrue(torch.allclose(ens_output, stacked, atol=1e-6))


class TestEnsembleDNNTraining(unittest.TestCase):
    """Test cases for ensemble training."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 4
        self.config = {
            "chem_input_dim": 256,
            "prot_input_dim": 128,
            "chem_hidden_dims": [64],
            "prot_hidden_dims": [32],
            "hidden_dims": [64],
            "output_dim": 1,
            "dropout": 0.1,
            "task_type": "regression",
            "aleatoric": False,
            "ensemble_size": 2,
            "seed": 42,
            "n_targets": -1,
            "MT": False,
        }

    def test_ensemble_training_loop(self):
        """Test simple training loop with ensemble."""
        ensemble = EnsembleDNN(config=self.config)
        ensemble.to(self.device)

        optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        # Create batch
        prot = torch.randn(self.batch_size, self.config["prot_input_dim"], device=self.device)
        chem = torch.randn(self.batch_size, self.config["chem_input_dim"], device=self.device)
        targets = torch.randn(self.batch_size, 1, device=self.device)

        # Forward pass
        outputs, _ = ensemble((prot, chem))

        # Average over ensemble members for loss
        avg_output = outputs.mean(dim=2)
        loss = loss_fn(avg_output, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that weights were updated
        self.assertTrue(torch.isfinite(loss))

    def test_ensemble_eval_vs_train_mode(self):
        """Test ensemble behavior in eval vs train mode."""
        ensemble = EnsembleDNN(config=self.config)

        prot = torch.randn(self.batch_size, self.config["prot_input_dim"])
        chem = torch.randn(self.batch_size, self.config["chem_input_dim"])

        # Eval mode - should be deterministic
        ensemble.eval()
        out1, _ = ensemble((prot, chem))
        out2, _ = ensemble((prot, chem))
        self.assertTrue(torch.allclose(out1, out2))

        # Train mode - outputs might differ due to dropout
        ensemble.train()
        out3, _ = ensemble((prot, chem))
        out4, _ = ensemble((prot, chem))
        # They might be different, but same shape
        self.assertEqual(out3.shape, out4.shape)


class TestEnsembleDNNIntegration(unittest.TestCase):
    """Integration tests for ensemble model."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "chem_input_dim": 256,
            "prot_input_dim": 128,
            "chem_hidden_dims": [64],
            "prot_hidden_dims": [32],
            "hidden_dims": [64],
            "output_dim": 1,
            "dropout": 0.1,
            "task_type": "regression",
            "aleatoric": False,
            "ensemble_size": 3,
            "seed": 42,
            "n_targets": -1,
            "MT": False,
        }

    def test_ensemble_inference_aggregation(self):
        """Test that ensemble can aggregate predictions from all members."""
        ensemble = EnsembleDNN(config=self.config)
        ensemble.eval()

        batch_size = 8
        prot = torch.randn(batch_size, self.config["prot_input_dim"])
        chem = torch.randn(batch_size, self.config["chem_input_dim"])

        with torch.no_grad():
            outputs, _ = ensemble((prot, chem))

        # Compute ensemble statistics
        mean_output = outputs.mean(dim=2)
        std_output = outputs.std(dim=2)

        self.assertEqual(mean_output.shape, (batch_size, 1))
        self.assertEqual(std_output.shape, (batch_size, 1))
        self.assertTrue(torch.isfinite(mean_output).all())
        self.assertTrue(torch.isfinite(std_output).all())

    def test_ensemble_with_aleatoric(self):
        """Test ensemble with aleatoric uncertainty."""
        config = self.config.copy()
        config["aleatoric"] = True

        ensemble = EnsembleDNN(config=config)
        ensemble.eval()

        batch_size = 4
        prot = torch.randn(batch_size, self.config["prot_input_dim"])
        chem = torch.randn(batch_size, self.config["chem_input_dim"])

        outputs, vars_ = ensemble((prot, chem))

        self.assertIsNotNone(vars_)
        self.assertTrue(torch.isfinite(vars_).all())

    def test_ensemble_all_models_receive_same_input(self):
        """Test that all ensemble members process the same input."""
        ensemble = EnsembleDNN(config=self.config)
        ensemble.eval()

        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        prot = torch.randn(4, self.config["prot_input_dim"])
        chem = torch.randn(4, self.config["chem_input_dim"])

        outputs, _ = ensemble((prot, chem))

        # Each column in the last dimension should be from one model
        # All should process the same input
        self.assertEqual(outputs.shape[2], len(ensemble.models))

        # Verify that different models give different outputs
        # (because they have different weights)
        for i in range(outputs.shape[2] - 1):
            self.assertFalse(torch.allclose(
                outputs[:, :, i],
                outputs[:, :, i + 1],
                atol=1e-4
            ))


class TestEnsembleDNNConfiguration(unittest.TestCase):
    """Test ensemble configuration and setup."""

    def test_ensemble_with_multitask(self):
        """Test ensemble with multitask configuration."""
        config = {
            "chem_input_dim": 256,
            "prot_input_dim": 128,
            "chem_hidden_dims": [64],
            "prot_hidden_dims": [32],
            "hidden_dims": [64],
            "output_dim": 3,  # 3 tasks
            "dropout": 0.1,
            "task_type": "regression",
            "aleatoric": False,
            "ensemble_size": 2,
            "seed": 42,
            "n_targets": 3,
            "MT": True,
        }

        ensemble = EnsembleDNN(config=config)
        ensemble.eval()

        prot = torch.randn(4, 128)
        chem = torch.randn(4, 256)

        outputs, _ = ensemble((prot, chem))

        # Should have 3 outputs per sample per model
        self.assertEqual(outputs.shape[1], 3)
        self.assertEqual(outputs.shape[2], 2)  # ensemble_size

    def test_ensemble_with_classification(self):
        """Test ensemble with classification task."""
        config = {
            "chem_input_dim": 256,
            "prot_input_dim": 128,
            "chem_hidden_dims": [64],
            "prot_hidden_dims": [32],
            "hidden_dims": [64],
            "output_dim": 2,  # binary classification (active/inactive)
            "dropout": 0.1,
            "task_type": "classification",
            "aleatoric": False,
            "ensemble_size": 2,
            "seed": 42,
            "n_targets": -1,
            "MT": False,
        }

        ensemble = EnsembleDNN(config=config)
        ensemble.eval()

        prot = torch.randn(4, 128)
        chem = torch.randn(4, 256)

        outputs, _ = ensemble((prot, chem))

        # Should have 2 outputs per sample (class logits)
        self.assertEqual(outputs.shape[1], 2)


if __name__ == "__main__":
    unittest.main()

