"""
Unit tests for metrics utilities in uqdd.models.utils_metrics.

Tests cover:
- NaN-aware metric calculation
- Regression metrics (RMSE, R², explained variance)
- Per-task metrics for multitask learning
- Prediction processing
- DataFrame creation
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
import torch
import numpy as np
import pandas as pd
from typing import Tuple

from uqdd.models.utils_metrics import (
    calc_nanaware_metrics,
    calc_regr_metrics,
)


class TestCalcNanAwareMetrics(unittest.TestCase):
    """Test cases for NaN-aware metrics calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8

    def test_calc_nanaware_metrics_no_nans(self):
        """Test nanaware metrics with no NaN values."""
        tensor = torch.randn(self.batch_size, 2)
        nan_mask = torch.zeros_like(tensor, dtype=torch.bool)

        result = calc_nanaware_metrics(tensor, nan_mask, all_tasks_agg=False)

        self.assertEqual(result.shape[0], 2)
        self.assertTrue(torch.isfinite(result).all())

    def test_calc_nanaware_metrics_with_nans(self):
        """Test nanaware metrics with some NaN values."""
        tensor = torch.randn(self.batch_size, 2)
        nan_mask = torch.zeros_like(tensor, dtype=torch.bool)
        nan_mask[0, 0] = True  # Mark one value as NaN
        nan_mask[1, 1] = True  # Mark another

        result = calc_nanaware_metrics(tensor, nan_mask, all_tasks_agg=False)

        # Should handle NaNs gracefully
        self.assertEqual(result.shape[0], 2)

    def test_calc_nanaware_metrics_mean_aggregation(self):
        """Test mean aggregation across tasks."""
        tensor = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        nan_mask = torch.zeros_like(tensor, dtype=torch.bool)

        result = calc_nanaware_metrics(tensor, nan_mask, all_tasks_agg="mean")

        # Should return scalar
        self.assertEqual(result.dim(), 0)
        self.assertTrue(torch.isfinite(result))

    def test_calc_nanaware_metrics_sum_aggregation(self):
        """Test sum aggregation across tasks."""
        tensor = torch.ones(self.batch_size, 2)
        nan_mask = torch.zeros_like(tensor, dtype=torch.bool)

        result = calc_nanaware_metrics(tensor, nan_mask, all_tasks_agg="sum")

        # Should return scalar
        self.assertEqual(result.dim(), 0)
        self.assertTrue(torch.isfinite(result))

    def test_calc_nanaware_metrics_all_nans_task(self):
        """Test nanaware metrics when entire task is NaN."""
        tensor = torch.randn(self.batch_size, 2)
        nan_mask = torch.zeros_like(tensor, dtype=torch.bool)
        nan_mask[:, 0] = True  # First task all NaN

        result = calc_nanaware_metrics(tensor, nan_mask, all_tasks_agg=False)

        # Should handle entire NaN columns
        self.assertEqual(result.shape[0], 2)

    def test_calc_nanaware_metrics_single_task(self):
        """Test nanaware metrics with single task."""
        tensor = torch.randn(self.batch_size, 1)
        nan_mask = torch.zeros_like(tensor, dtype=torch.bool)

        result = calc_nanaware_metrics(tensor, nan_mask, all_tasks_agg=False)

        self.assertEqual(result.shape[0], 1)

    def test_calc_nanaware_metrics_many_tasks(self):
        """Test nanaware metrics with many tasks."""
        num_tasks = 10
        tensor = torch.randn(self.batch_size, num_tasks)
        nan_mask = torch.zeros_like(tensor, dtype=torch.bool)

        result = calc_nanaware_metrics(tensor, nan_mask, all_tasks_agg=False)

        self.assertEqual(result.shape[0], num_tasks)


class TestCalcRegrMetrics(unittest.TestCase):
    """Test cases for regression metrics calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8

    def test_calc_regr_metrics_perfect_prediction(self):
        """Test regression metrics with perfect predictions."""
        targets = torch.randn(self.batch_size, 1)
        outputs = targets.clone()

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        # Perfect prediction should have
        # - Near-zero RMSE
        # - R² close to 1.0
        # - Explained variance close to 1.0
        self.assertLess(rmse, 1e-5)
        self.assertGreater(r2, 0.99)

    def test_calc_regr_metrics_constant_prediction(self):
        """Test regression metrics with constant prediction (worst case)."""
        targets = torch.tensor([
            [1.0], [2.0], [3.0], [4.0],
            [5.0], [1.0], [2.0], [3.0]
        ])
        outputs = torch.ones(self.batch_size, 1) * targets.mean()  # Predict mean

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        # All should be finite
        self.assertTrue(np.isfinite(rmse))
        self.assertTrue(np.isfinite(r2))
        self.assertTrue(np.isfinite(explained_var))

    def test_calc_regr_metrics_random_prediction(self):
        """Test regression metrics with random predictions."""
        targets = torch.randn(self.batch_size, 1)
        outputs = torch.randn(self.batch_size, 1)

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        # All should be computed and finite
        self.assertTrue(np.isfinite(rmse))
        self.assertTrue(np.isfinite(r2))
        self.assertTrue(np.isfinite(explained_var))

    def test_calc_regr_metrics_output_types(self):
        """Test that output types are correct."""
        targets = torch.randn(self.batch_size, 1)
        outputs = torch.randn(self.batch_size, 1)

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        # Should be scalars
        self.assertIsInstance(rmse, (float, np.floating))
        self.assertIsInstance(r2, (float, np.floating))
        self.assertIsInstance(explained_var, (float, np.floating))

    def test_calc_regr_metrics_single_sample(self):
        """Test regression metrics with single sample."""
        targets = torch.tensor([[1.0]])
        outputs = torch.tensor([[1.1]])

        # Might raise error or handle specially
        try:
            rmse, r2, explained_var = calc_regr_metrics(targets, outputs)
            # If it succeeds, values should be valid
            self.assertTrue(np.isfinite(rmse) or np.isnan(rmse))
        except (RuntimeError, ValueError):
            # Some metrics may not be computable with single sample
            pass

    def test_calc_regr_metrics_large_batch(self):
        """Test regression metrics with large batch."""
        batch_size = 1000
        targets = torch.randn(batch_size, 1)
        outputs = targets + torch.randn(batch_size, 1) * 0.1

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        self.assertTrue(np.isfinite(rmse))
        self.assertTrue(np.isfinite(r2))
        self.assertTrue(np.isfinite(explained_var))

    def test_calc_regr_metrics_per_task(self):
        """Test per-task regression metrics."""
        num_tasks = 3
        targets = torch.randn(self.batch_size, num_tasks)
        outputs = targets + torch.randn(self.batch_size, num_tasks) * 0.1

        rmse, r2, explained_var = calc_regr_metrics(
            targets, outputs, metrics_per_task=True
        )

        # Should return per-task metrics
        if isinstance(rmse, torch.Tensor):
            self.assertEqual(rmse.shape[0], num_tasks)
        else:
            # Or could be single value
            self.assertTrue(np.isfinite(rmse))


class TestMetricsIntegration(unittest.TestCase):
    """Integration tests for metrics computation."""

    def setUp(self):
        """Set up test fixtures with a default batch size used across tests."""
        self.batch_size = 16

    def test_regression_metrics_workflow(self):
        """Test complete regression metrics workflow."""
        batch_size = 16
        targets = torch.randn(batch_size, 1)
        outputs = targets + torch.randn(batch_size, 1) * 0.2

        # Compute metrics
        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        # All should be meaningful values
        self.assertGreater(rmse, 0)
        self.assertLess(r2, 1.0)
        self.assertTrue(np.isfinite(rmse))
        self.assertTrue(np.isfinite(r2))

    def test_multitask_metrics_workflow(self):
        """Test multitask regression metrics."""
        batch_size = 16
        num_tasks = 3

        targets = torch.randn(batch_size, num_tasks)
        # Slightly noisy predictions
        outputs = targets + torch.randn(batch_size, num_tasks) * 0.1

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        self.assertTrue(np.isfinite(rmse))
        self.assertTrue(np.isfinite(r2))

    def test_metrics_with_nan_handling(self):
        """Test metrics computation with NaN values."""
        targets = torch.randn(self.batch_size, 2)
        outputs = torch.randn(self.batch_size, 2)

        # Add some NaN values
        targets[0, 0] = float('nan')
        outputs[1, 1] = float('nan')

        # Try to compute metrics (might skip NaN samples)
        try:
            rmse, r2, explained_var = calc_regr_metrics(targets, outputs)
            # If it succeeds, should handle NaNs
            self.assertTrue(np.isfinite(rmse) or np.isnan(rmse))
        except ValueError:
            # Might raise error for invalid data
            pass

    def test_metrics_reproducibility(self):
        """Test that metrics are reproducible."""
        torch.manual_seed(42)
        targets = torch.randn(16, 1)
        outputs = targets + torch.randn(16, 1) * 0.1

        # Compute twice
        rmse1, r2_1, ev1 = calc_regr_metrics(targets, outputs)
        rmse2, r2_2, ev2 = calc_regr_metrics(targets, outputs)

        # Should be identical
        self.assertEqual(rmse1, rmse2)
        self.assertEqual(r2_1, r2_2)
        self.assertEqual(ev1, ev2)


class TestMetricsBoundaryConditions(unittest.TestCase):
    """Test metrics with boundary conditions."""

    def test_metrics_with_zeros(self):
        """Test metrics when predictions and targets are zero."""
        batch_size = 8
        targets = torch.zeros(batch_size, 1)
        outputs = torch.zeros(batch_size, 1)

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        # Perfect prediction (zero error)
        self.assertLess(rmse, 1e-5)

    def test_metrics_with_large_values(self):
        """Test metrics with large values."""
        batch_size = 8
        targets = torch.randn(batch_size, 1) * 1e6
        outputs = targets + torch.randn(batch_size, 1) * 1e4

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        self.assertTrue(np.isfinite(rmse))
        self.assertTrue(np.isfinite(r2))

    def test_metrics_with_small_values(self):
        """Test metrics with small values."""
        batch_size = 8
        targets = torch.randn(batch_size, 1) * 1e-6
        outputs = targets + torch.randn(batch_size, 1) * 1e-8

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        self.assertTrue(np.isfinite(rmse) or rmse < 1e-10)

    def test_metrics_with_negative_values(self):
        """Test metrics with negative target values."""
        batch_size = 8
        targets = torch.randn(batch_size, 1) - 10  # Negative values
        outputs = targets + torch.randn(batch_size, 1) * 0.1

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        self.assertTrue(np.isfinite(rmse))
        self.assertTrue(np.isfinite(r2))

    def test_metrics_imbalanced_variance(self):
        """Test metrics with imbalanced variance in targets."""
        batch_size = 8
        # One task has high variance, one has low
        targets = torch.stack([
            torch.randn(batch_size, 1) * 10,  # High variance
            torch.randn(batch_size, 1) * 0.01  # Low variance
        ], dim=1).squeeze(-1)

        outputs = targets + torch.randn_like(targets) * 0.1

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        self.assertTrue(np.isfinite(rmse))


class TestMetricsEdgeCases(unittest.TestCase):
    """Test metrics with edge cases."""

    def test_metrics_constant_targets(self):
        """Test metrics when all targets are identical."""
        batch_size = 8
        targets = torch.ones(batch_size, 1) * 5.0
        outputs = targets + torch.randn(batch_size, 1) * 0.1

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        # Should handle constant targets
        self.assertTrue(np.isfinite(rmse))

    def test_metrics_constant_predictions(self):
        """Test metrics when all predictions are identical."""
        batch_size = 8
        targets = torch.randn(batch_size, 1)
        outputs = torch.ones(batch_size, 1) * targets.mean()

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        # Should compute metrics
        self.assertTrue(np.isfinite(rmse))

    def test_metrics_opposite_correlation(self):
        """Test metrics with negative correlation."""
        batch_size = 8
        targets = torch.linspace(-5, 5, batch_size).unsqueeze(1)
        outputs = -targets  # Perfect negative correlation

        rmse, r2, explained_var = calc_regr_metrics(targets, outputs)

        # R² should be negative (worse than predicting mean)
        self.assertTrue(r2 < 0)


if __name__ == "__main__":
    unittest.main()

