"""
Pytest configuration and shared fixtures for model tests.

This module provides common fixtures and utilities for testing the UQDD models,
following patterns similar to test_data_papyrus.py.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Tuple, Optional

import uqdd.models.utils_models as um
from uqdd.models.pnn import PNN


# ============================================================================
# DEVICE AND DTYPE FIXTURES
# ============================================================================

@pytest.fixture
def device():
    """Return CPU device for testing (avoid GPU issues)."""
    return torch.device("cpu")


@pytest.fixture
def dtype():
    """Return default float dtype for tensors."""
    return torch.float32


# ============================================================================
# TENSOR AND BATCH FIXTURES
# ============================================================================

@pytest.fixture
def sample_tensor_2d(device, dtype):
    """Create a sample 2D tensor (batch_size=8, features=10)."""
    return torch.randn(8, 10, device=device, dtype=dtype)


@pytest.fixture
def sample_tensor_3d(device, dtype):
    """Create a sample 3D tensor (batch_size=4, features=5, time_steps=3)."""
    return torch.randn(4, 5, 3, device=device, dtype=dtype)


@pytest.fixture
def batch_proteins(device, dtype):
    """Create a batch of protein descriptors (batch_size=8, prot_dim=256)."""
    return torch.randn(8, 256, device=device, dtype=dtype)


@pytest.fixture
def batch_chemicals(device, dtype):
    """Create a batch of chemical descriptors (batch_size=8, chem_dim=2048)."""
    return torch.randn(8, 2048, device=device, dtype=dtype)


@pytest.fixture
def batch_targets(device, dtype):
    """Create a batch of regression targets (batch_size=8)."""
    return torch.randn(8, 1, device=device, dtype=dtype)


@pytest.fixture
def batch_labels_binary(device):
    """Create a batch of binary classification labels (batch_size=8)."""
    return torch.randint(0, 2, (8, 1), device=device, dtype=torch.float32)


# ============================================================================
# MODEL CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def minimal_pnn_config():
    """Minimal PNN configuration for testing."""
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


@pytest.fixture
def minimal_ensemble_config():
    """Minimal ensemble configuration for testing."""
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
        "ensemble_size": 3,
        "seed": 42,
        "n_targets": -1,
        "MT": False,
    }


@pytest.fixture
def minimal_evidential_config():
    """Minimal evidential model configuration for testing."""
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


# ============================================================================
# MODEL FIXTURES
# ============================================================================

@pytest.fixture
def pnn_model(minimal_pnn_config, device):
    """Create a simple PNN model for testing."""
    model = PNN(config=minimal_pnn_config)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def pnn_model_with_aleatoric(minimal_pnn_config, device):
    """Create a PNN model with aleatoric uncertainty for testing."""
    config = minimal_pnn_config.copy()
    config["aleatoric"] = True
    model = PNN(config=config)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def simple_mlp(device):
    """Create a simple MLP for testing."""
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(50, 25),
        nn.ReLU(),
        nn.Linear(25, 1),
    )
    model.to(device)
    return model


# ============================================================================
# OPTIMIZER AND SCHEDULER FIXTURES
# ============================================================================

@pytest.fixture
def optimizer(simple_mlp):
    """Create an optimizer for testing."""
    return torch.optim.Adam(simple_mlp.parameters(), lr=1e-3)


@pytest.fixture
def scheduler(optimizer):
    """Create a learning rate scheduler for testing."""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# ============================================================================
# DATASET AND DATALOADER FIXTURES
# ============================================================================

@pytest.fixture
def dummy_dataset(device, dtype):
    """Create a simple dummy dataset for testing."""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=32, prot_dim=256, chem_dim=2048):
            self.size = size
            self.prot_dim = prot_dim
            self.chem_dim = chem_dim

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            prot = torch.randn(self.prot_dim, device=device, dtype=dtype)
            chem = torch.randn(self.chem_dim, device=device, dtype=dtype)
            target = torch.randn(1, device=device, dtype=dtype)
            return (prot, chem), target

    return DummyDataset()


@pytest.fixture
def dummy_dataloader(dummy_dataset):
    """Create a DataLoader from the dummy dataset."""
    return torch.utils.data.DataLoader(dummy_dataset, batch_size=4, shuffle=False)


# ============================================================================
# LOSS FUNCTION FIXTURES
# ============================================================================

@pytest.fixture
def nig_parameters(batch_targets, device, dtype):
    """Create NIG parameters for testing."""
    batch_size = batch_targets.shape[0]
    return {
        "mu": torch.randn(batch_size, 1, device=device, dtype=dtype),
        "v": torch.relu(torch.randn(batch_size, 1, device=device, dtype=dtype)) + 0.1,
        "alpha": torch.relu(torch.randn(batch_size, 1, device=device, dtype=dtype)) + 1.1,
        "beta": torch.relu(torch.randn(batch_size, 1, device=device, dtype=dtype)) + 0.1,
        "y": batch_targets,
    }


@pytest.fixture
def dirichlet_parameters(device, dtype):
    """Create Dirichlet parameters for testing."""
    batch_size = 8
    num_classes = 2
    return {
        "alpha": torch.relu(torch.randn(batch_size, num_classes, device=device, dtype=dtype)) + 0.5,
        "y": torch.nn.functional.one_hot(torch.randint(0, num_classes, (batch_size,)), num_classes=num_classes).float().to(device),
    }


# ============================================================================
# UTILITY FUNCTION FIXTURES
# ============================================================================

@pytest.fixture
def mock_config_dir(tmp_path):
    """Create a temporary directory with mock config files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def mock_model_dir(tmp_path):
    """Create a temporary directory for model artifacts."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


# ============================================================================
# SEED FIXTURES FOR DETERMINISM
# ============================================================================

@pytest.fixture(autouse=True)
def reset_seed():
    """Reset random seed before and after each test."""
    um.set_seed(42)
    yield
    um.set_seed(42)


# ============================================================================
# CONTEXT MANAGERS AND UTILITIES
# ============================================================================

@pytest.fixture
def no_wandb():
    """Context manager to mock wandb during tests."""
    with patch("wandb.log"):
        yield


@pytest.fixture
def mock_device():
    """Mock device operations for testing without GPU."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


# ============================================================================
# CUSTOM MARKERS
# ============================================================================

def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# ============================================================================
# HELPER UTILITIES
# ============================================================================

def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...]):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {tensor.shape}"
    )


def assert_tensor_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype):
    """Assert tensor has expected dtype."""
    assert tensor.dtype == expected_dtype, (
        f"Expected dtype {expected_dtype}, got {tensor.dtype}"
    )


def assert_finite(tensor: torch.Tensor):
    """Assert tensor contains no NaN or Inf values."""
    assert torch.isfinite(tensor).all(), (
        f"Tensor contains NaN or Inf values"
    )


def assert_grad_flow(tensor: torch.Tensor):
    """Assert that gradient exists for backward pass."""
    assert tensor.grad is not None, "No gradient computed"


import pytest

# Provide dropout_rate fixture for parametrized unittest methods
@pytest.fixture(params=[0.0, 0.1, 0.3, 0.5])
def dropout_rate(request):
    return request.param
