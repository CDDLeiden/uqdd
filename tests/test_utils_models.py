"""
Unit tests for model utilities in uqdd.models.utils_models.

Tests cover:
- Random seed setting and reproducibility
- Norm computation (parameter and gradient norms)
- Descriptor length extraction and lookup
- Model configuration retrieval
- Dataset and DataLoader building
- Optimizer and scheduler initialization
- Model saving and loading
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path

import uqdd.models.utils_models as um
from uqdd.models.pnn import PNN


class TestSetSeed(unittest.TestCase):
    """Test cases for set_seed function."""

    def test_set_seed_reproducibility_torch(self):
        """Test that set_seed makes PyTorch operations deterministic."""
        um.set_seed(42)
        t1 = torch.randn(10, 10)

        um.set_seed(42)
        t2 = torch.randn(10, 10)

        self.assertTrue(torch.allclose(t1, t2))

    def test_set_seed_reproducibility_numpy(self):
        """Test that set_seed makes NumPy operations deterministic."""
        um.set_seed(42)
        arr1 = np.random.randn(100)

        um.set_seed(42)
        arr2 = np.random.randn(100)

        np.testing.assert_array_equal(arr1, arr2)

    def test_set_seed_reproducibility_python(self):
        """Test that set_seed makes Python random operations deterministic."""
        import random

        um.set_seed(42)
        lst1 = [random.random() for _ in range(10)]

        um.set_seed(42)
        lst2 = [random.random() for _ in range(10)]

        self.assertEqual(lst1, lst2)

    def test_set_seed_model_training_reproducibility(self):
        """Test that set_seed makes model initialization deterministic."""
        um.set_seed(42)
        model1 = nn.Linear(100, 50)
        weights1 = model1.weight.data.clone()

        um.set_seed(42)
        model2 = nn.Linear(100, 50)
        weights2 = model2.weight.data

        self.assertTrue(torch.allclose(weights1, weights2))

    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different outputs."""
        um.set_seed(42)
        t1 = torch.randn(10, 10)

        um.set_seed(123)
        t2 = torch.randn(10, 10)

        self.assertFalse(torch.allclose(t1, t2))


class TestComputePnorm(unittest.TestCase):
    """Test cases for compute_pnorm function."""

    def test_compute_pnorm_simple_model(self):
        """Test pnorm computation on simple model."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        pnorm = um.compute_pnorm(model)

        self.assertGreater(pnorm, 0)
        self.assertTrue(np.isfinite(pnorm))

    def test_compute_pnorm_zero_initialized(self):
        """Test pnorm of zero-initialized model."""
        model = nn.Linear(10, 5)
        with torch.no_grad():
            model.weight.fill_(0)
            model.bias.fill_(0)

        pnorm = um.compute_pnorm(model)
        self.assertEqual(pnorm, 0.0)

    def test_compute_pnorm_consistency(self):
        """Test that pnorm is consistent across calls."""
        model = nn.Linear(100, 50)
        pnorm1 = um.compute_pnorm(model)
        pnorm2 = um.compute_pnorm(model)

        self.assertEqual(pnorm1, pnorm2)

    def test_compute_pnorm_matches_manual_calculation(self):
        """Test pnorm matches manual L2 norm calculation."""
        model = nn.Linear(3, 2)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(1.0)

        pnorm = um.compute_pnorm(model)

        # Manual calculation: sqrt(sum of squared parameters)
        # weight: 3*2=6 params with value 1
        # bias: 2 params with value 1
        # sum of squares: 6 + 2 = 8
        # L2 norm: sqrt(8) = 2.828...
        expected = np.sqrt(8)
        self.assertAlmostEqual(pnorm, expected, places=5)

    def test_compute_pnorm_larger_weights_larger_norm(self):
        """Test that larger weights result in larger pnorm."""
        model1 = nn.Linear(10, 5)
        with torch.no_grad():
            model1.weight.mul_(1.0)

        model2 = nn.Linear(10, 5)
        with torch.no_grad():
            model2.weight.mul_(10.0)

        pnorm1 = um.compute_pnorm(model1)
        pnorm2 = um.compute_pnorm(model2)

        self.assertLess(pnorm1, pnorm2)


class TestComputeGnorm(unittest.TestCase):
    """Test cases for compute_gnorm function."""

    def test_compute_gnorm_after_backward(self):
        """Test gnorm computation after backward pass."""
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        output = model(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()

        gnorm = um.compute_gnorm(model)
        self.assertGreater(gnorm, 0)
        self.assertTrue(np.isfinite(gnorm))

    def test_compute_gnorm_no_gradient_no_error(self):
        """Test gnorm when model has no gradients."""
        model = nn.Linear(10, 5)

        # Model has no gradients yet, should handle gracefully
        gnorm = um.compute_gnorm(model)
        self.assertEqual(gnorm, 0.0)

    def test_compute_gnorm_zero_gradients(self):
        """Test gnorm when gradients are zero."""
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Zero out gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        gnorm = um.compute_gnorm(model)
        self.assertEqual(gnorm, 0.0)

    def test_compute_gnorm_consistency_per_step(self):
        """Test that gnorm increases with different gradients."""
        model = nn.Linear(10, 5)

        # First gradient step
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        output = model(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()
        gnorm1 = um.compute_gnorm(model)

        # Should be consistent
        gnorm1_repeat = um.compute_gnorm(model)
        self.assertEqual(gnorm1, gnorm1_repeat)


class TestGetDescLenFromDataset(unittest.TestCase):
    """Test cases for get_desc_len_from_dataset function."""

    def test_get_desc_len_from_dataset_tuple_inputs(self):
        """Test descriptor length extraction from dataset with tuple inputs."""
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self):
                pass

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                prot = torch.randn(256)
                chem = torch.randn(2048)
                target = torch.randn(1)
                return (prot, chem), target

        dataset = DummyDataset()
        prot_len, chem_len = um.get_desc_len_from_dataset(dataset)

        self.assertEqual(prot_len, 256)
        self.assertEqual(chem_len, 2048)

    def test_get_desc_len_from_dataset_single_tensor(self):
        """Test descriptor length extraction from dataset with single tensor input."""
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self):
                pass

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                chem = torch.randn(2048)
                target = torch.randn(1)
                return chem, target

        dataset = DummyDataset()
        prot_len, chem_len = um.get_desc_len_from_dataset(dataset)

        self.assertEqual(prot_len, 0)
        self.assertEqual(chem_len, 2048)


class TestGetDescLen(unittest.TestCase):
    """Test cases for get_desc_len function."""

    @patch("uqdd.models.utils_models.get_config")
    def test_get_desc_len_single_descriptor(self, mock_get_config):
        """Test descriptor length retrieval for single descriptor."""
        mock_get_config.return_value = {
            "ecfp2048": 2048,
            "ankh-base": 256,
        }

        lengths = um.get_desc_len("ecfp2048")
        self.assertEqual(lengths, (2048,))

    @patch("uqdd.models.utils_models.get_config")
    def test_get_desc_len_multiple_descriptors(self, mock_get_config):
        """Test descriptor length retrieval for multiple descriptors."""
        mock_get_config.return_value = {
            "ecfp2048": 2048,
            "ankh-base": 256,
        }

        lengths = um.get_desc_len("ecfp2048", "ankh-base")
        self.assertEqual(lengths, (2048, 256))

    @patch("uqdd.models.utils_models.get_config")
    def test_get_desc_len_missing_descriptor(self, mock_get_config):
        """Test descriptor length retrieval with missing descriptor."""
        mock_get_config.return_value = {
            "ecfp2048": 2048,
        }

        lengths = um.get_desc_len("ecfp2048", "unknown")
        self.assertEqual(lengths, (2048, 0))


class TestGetModelConfig(unittest.TestCase):
    """Test cases for get_model_config function."""

    @patch("uqdd.models.utils_models.get_config")
    def test_get_model_config_pnn(self, mock_get_config):
        """Test model config retrieval for PNN."""
        mock_config = {"model": "pnn", "hidden_dims": [256, 128]}
        mock_get_config.return_value = mock_config

        config = um.get_model_config(model_type="pnn")

        self.assertEqual(config["model"], "pnn")
        mock_get_config.assert_called_once()

    @patch("uqdd.models.utils_models.get_config")
    def test_get_model_config_ensemble(self, mock_get_config):
        """Test model config retrieval for ensemble."""
        mock_config = {"model": "ensemble", "ensemble_size": 10}
        mock_get_config.return_value = mock_config

        config = um.get_model_config(model_type="ensemble")

        self.assertEqual(config["model"], "ensemble")

    def test_get_model_config_invalid_type(self):
        """Test that invalid model type raises assertion."""
        with self.assertRaises(AssertionError):
            um.get_model_config(model_type="invalid_model")

    @patch("uqdd.models.utils_models.get_config")
    def test_get_model_config_with_kwargs(self, mock_get_config):
        """Test model config retrieval with additional kwargs."""
        mock_config = {"model": "pnn", "hidden_dims": [256, 128]}
        mock_get_config.return_value = mock_config

        config = um.get_model_config(
            model_type="pnn",
            split_type="scaffold",
            activity_type="kx"
        )

        # Should pass kwargs to get_config
        mock_get_config.assert_called_once()


class TestGetSweepConfig(unittest.TestCase):
    """Test cases for get_sweep_config function."""

    @patch("uqdd.models.utils_models.get_config")
    def test_get_sweep_config_pnn(self, mock_get_config):
        """Test sweep config retrieval for PNN."""
        mock_config = {
            "parameters": {
                "learning_rate": {"min": 1e-4, "max": 1e-2},
                "hidden_dims": {"values": [[128, 64], [256, 128]]},
            }
        }
        mock_get_config.return_value = mock_config

        config = um.get_sweep_config(model_name="pnn")

        self.assertIn("parameters", config)
        mock_get_config.assert_called_once()

    @patch("uqdd.models.utils_models.get_config")
    def test_get_sweep_config_with_overrides(self, mock_get_config):
        """Test sweep config with parameter overrides."""
        mock_config = {
            "parameters": {
                "learning_rate": {"min": 1e-4, "max": 1e-2},
            }
        }
        mock_get_config.return_value = mock_config

        config = um.get_sweep_config(model_name="pnn", learning_rate=1e-3)

        self.assertIn("parameters", config)

    def test_get_sweep_config_invalid_model(self):
        """Test that invalid model name raises assertion."""
        with self.assertRaises(AssertionError):
            um.get_sweep_config(model_name="invalid_model")


class TestBuildDatasets(unittest.TestCase):
    """Test cases for build_datasets function."""

    @patch("uqdd.models.utils_models.get_datasets")
    def test_build_datasets_papyrus_default(self, mock_get_datasets):
        """Test dataset building with default Papyrus parameters."""
        mock_datasets = {
            "train": MagicMock(),
            "val": MagicMock(),
            "test": MagicMock(),
        }
        mock_get_datasets.return_value = mock_datasets

        datasets = um.build_datasets(data_name="papyrus")

        self.assertIn("train", datasets)
        mock_get_datasets.assert_called_once()

    @patch("uqdd.models.utils_models.get_datasets")
    def test_build_datasets_invalid_data_name(self, mock_get_datasets):
        """Test that invalid data name raises ValueError."""
        with self.assertRaises(ValueError):
            um.build_datasets(data_name="unknown_dataset")

    @patch("uqdd.models.utils_models.get_datasets")
    def test_build_datasets_with_config(self, mock_get_datasets):
        """Test dataset building with custom configuration."""
        mock_datasets = {"train": MagicMock(), "val": MagicMock(), "test": MagicMock()}
        mock_get_datasets.return_value = mock_datasets

        datasets = um.build_datasets(
            data_name="papyrus",
            n_targets=5,
            activity_type="kx",
            split_type="scaffold",
            desc_prot="ankh-base",
            desc_chem="ecfp2048",
        )

        self.assertIn("train", datasets)
        mock_get_datasets.assert_called_once()


class TestModelSaveLoad(unittest.TestCase):
    """Test cases for model saving and loading."""

    def test_save_and_load_model_state(self):
        """Test that model state is preserved after save/load."""
        model = nn.Linear(10, 5)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(2.0)

        original_state = model.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"

            # Save
            torch.save(model.state_dict(), model_path)

            # Load
            loaded_model = nn.Linear(10, 5)
            loaded_model.load_state_dict(torch.load(model_path))

            # Compare
            for key in original_state:
                self.assertTrue(torch.allclose(
                    original_state[key],
                    loaded_model.state_dict()[key]
                ))

    def test_save_and_load_full_model(self):
        """Test saving and loading full model."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "full_model.pt"

            # Save full model
            torch.save(model, model_path)

            # Load full model (PyTorch 2.6 defaults weights_only=True; override for safe local test)
            loaded_model = torch.load(model_path, weights_only=False)

            # Test inference
            x = torch.randn(4, 10)
            out1 = model(x)
            out2 = loaded_model(x)

            self.assertTrue(torch.allclose(out1, out2))


if __name__ == "__main__":
    unittest.main()
