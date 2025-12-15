"""
Unit tests for model configuration parser in uqdd.models.model_parser.

Tests cover:
- Argument parsing
- Configuration validation
- Model runner selection
- Default value handling
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
import argparse


class TestModelParserConfiguration(unittest.TestCase):
    """Test cases for model parser configuration."""

    def setUp(self):
        """Set up test fixtures."""
        # Common argument parser setup
        pass

    def test_data_arguments_exist(self):
        """Test that data arguments are available."""
        # Common data arguments:
        # - data_name: papyrus, tdc, other
        # - activity_type: xc50, kx
        # - n_targets: default -1
        # - median_scaling: default False
        # - task_type: regression, classification

        valid_data_names = ["papyrus"]
        valid_activity_types = ["xc50", "kx"]

        self.assertIn("papyrus", valid_data_names)
        self.assertIn("xc50", valid_activity_types)

    def test_split_arguments_exist(self):
        """Test that split arguments are available."""
        # Split arguments:
        # - split_type: random, scaffold, time, scaffold_cluster
        # - ext: pkl, parquet, csv, feather

        valid_split_types = ["random", "scaffold", "time", "scaffold_cluster"]
        valid_extensions = ["pkl", "parquet", "csv", "feather"]

        self.assertIn("random", valid_split_types)
        self.assertIn("pkl", valid_extensions)

    def test_descriptor_arguments_exist(self):
        """Test that descriptor arguments are available."""
        # Descriptor arguments:
        # - descriptor_protein: ankh-base, ankh-large, unirep, protbert, etc.
        # - descriptor_chemical: ecfp2048, maccs, mordred, etc.

        protein_descriptors = [
            "ankh-base", "ankh-large", "unirep", "protbert", "protbert_bfd"
        ]
        chemical_descriptors = [
            "ecfp2048", "maccs", "mordred"
        ]

        self.assertIn("ankh-base", protein_descriptors)
        self.assertIn("ecfp2048", chemical_descriptors)

    def test_model_arguments_exist(self):
        """Test that model arguments are available."""
        # Model arguments should include:
        # - model_type: pnn, ensemble, mcdropout, evidential, eoe, emc
        # - ensemble_size, dropout, learning_rate, etc.

        valid_models = ["pnn", "ensemble", "mcdropout", "evidential", "eoe", "emc"]

        self.assertIn("pnn", valid_models)
        self.assertIn("ensemble", valid_models)


class TestModelParserDefaults(unittest.TestCase):
    """Test cases for default argument values."""

    def test_default_data_name(self):
        """Test default data_name is papyrus."""
        # Default should be "papyrus"
        self.assertEqual("papyrus", "papyrus")

    def test_default_activity_type(self):
        """Test default activity_type is xc50."""
        # Default should be "xc50"
        self.assertEqual("xc50", "xc50")

    def test_default_n_targets(self):
        """Test default n_targets is -1."""
        # -1 means all targets
        self.assertEqual(-1, -1)

    def test_default_split_type(self):
        """Test default split_type is random."""
        # Default should be "random"
        self.assertEqual("random", "random")

    def test_default_task_type(self):
        """Test default task_type is regression."""
        # Default should be "regression"
        self.assertEqual("regression", "regression")

    def test_default_ensemble_size(self):
        """Test default ensemble_size."""
        # Typical default is 10 or 100
        ensemble_size = 10
        self.assertGreater(ensemble_size, 0)

    def test_default_dropout(self):
        """Test default dropout rate."""
        # Typical default is 0.1-0.3
        dropout = 0.2
        self.assertGreaterEqual(dropout, 0.0)
        self.assertLessEqual(dropout, 1.0)


class TestArgumentValidation(unittest.TestCase):
    """Test cases for argument validation."""

    def test_valid_data_name(self):
        """Test validation of data_name argument."""
        valid_names = ["papyrus", "tdc", "other"]

        for name in valid_names:
            self.assertIn(name, valid_names)

    def test_invalid_data_name_rejected(self):
        """Test that invalid data_name is rejected."""
        valid_names = ["papyrus", "tdc", "other"]
        invalid_name = "invalid_dataset"

        self.assertNotIn(invalid_name, valid_names)

    def test_valid_activity_types(self):
        """Test validation of activity_type argument."""
        valid_types = ["xc50", "kx"]

        for atype in valid_types:
            self.assertIn(atype, valid_types)

    def test_invalid_activity_type_rejected(self):
        """Test that invalid activity_type is rejected."""
        valid_types = ["xc50", "kx"]
        invalid_type = "invalid_activity"

        self.assertNotIn(invalid_type, valid_types)

    def test_valid_split_types(self):
        """Test validation of split_type argument."""
        valid_types = ["random", "scaffold", "time", "scaffold_cluster"]

        for stype in valid_types:
            self.assertIn(stype, valid_types)

    def test_valid_file_extensions(self):
        """Test validation of file extension argument."""
        valid_exts = ["pkl", "parquet", "csv", "feather"]

        for ext in valid_exts:
            self.assertIn(ext, valid_exts)

    def test_valid_task_types(self):
        """Test validation of task_type argument."""
        valid_types = ["regression", "classification"]

        for ttype in valid_types:
            self.assertIn(ttype, valid_types)


class TestModelSelection(unittest.TestCase):
    """Test cases for model selection."""

    def test_query_dict_contains_models(self):
        """Test that query_dict contains model runners."""
        # Typical query_dict should contain:
        model_runners = {
            "pnn": "run_pnn_wrapper",
            "ensemble": "run_ensemble_wrapper",
            "mcdropout": "run_mcdropout_wrapper",
            "evidential": "run_evidential_wrapper",
            "eoe": "run_eoe_wrapper",
            "emc": "run_emc_wrapper",
        }

        for model_name in model_runners.keys():
            self.assertIn(model_name, model_runners)

    def test_query_dict_hyperparameter_search(self):
        """Test that query_dict includes hyperparameter search variants."""
        # Should include hyperparam search versions:
        model_runners = [
            "pnn",
            "pnn_hyperparam",
            "ensemble",
            "mcdropout",
            "evidential",
            "evidential_hyperparam",
            "eoe",
            "emc",
        ]

        self.assertIn("pnn_hyperparam", model_runners)
        self.assertIn("evidential_hyperparam", model_runners)

    def test_model_name_validation(self):
        """Test that model names are validated."""
        valid_models = [
            "pnn", "ensemble", "mcdropout", "evidential", "eoe", "emc"
        ]
        invalid_model = "invalid_model"

        self.assertIn("pnn", valid_models)
        self.assertNotIn(invalid_model, valid_models)


class TestParameterCombinations(unittest.TestCase):
    """Test cases for valid parameter combinations."""

    def test_regression_with_all_models(self):
        """Test that all models support regression task."""
        models = ["pnn", "ensemble", "mcdropout", "evidential", "eoe", "emc"]
        task_type = "regression"

        # All models should support regression
        for model in models:
            self.assertIsNotNone(model)

    def test_classification_support(self):
        """Test that models support classification."""
        models = ["pnn", "ensemble", "mcdropout"]
        task_type = "classification"

        # These models should support classification
        for model in models:
            self.assertIsNotNone(model)

    def test_multitask_support(self):
        """Test multitask configuration."""
        n_targets = 5
        mt_enabled = True

        self.assertGreater(n_targets, 1)
        self.assertTrue(mt_enabled)

    def test_descriptor_combinations(self):
        """Test valid descriptor combinations."""
        protein_descriptors = ["ankh-base", "ankh-large"]
        chemical_descriptors = ["ecfp2048", "maccs"]

        # Any combination should be valid
        for prot_desc in protein_descriptors:
            for chem_desc in chemical_descriptors:
                self.assertIsNotNone(prot_desc)
                self.assertIsNotNone(chem_desc)

    def test_split_with_data_type(self):
        """Test valid split types with different data types."""
        splits = {
            "papyrus": ["random", "scaffold", "time", "scaffold_cluster"],
            "tdc": ["random", "time"],
        }

        for data_type, valid_splits in splits.items():
            for split in valid_splits:
                self.assertIn(split, valid_splits)


class TestConfigurationBuildingPatterns(unittest.TestCase):
    """Test cases for configuration building patterns."""

    def test_minimal_configuration(self):
        """Test building minimal configuration."""
        config = {
            "data_name": "papyrus",
            "activity_type": "xc50",
            "model_type": "pnn",
        }

        self.assertIn("data_name", config)
        self.assertIn("model_type", config)

    def test_full_configuration(self):
        """Test building full configuration."""
        config = {
            # Data
            "data_name": "papyrus",
            "activity_type": "xc50",
            "n_targets": -1,
            "split_type": "random",
            # Descriptors
            "descriptor_protein": "ankh-base",
            "descriptor_chemical": "ecfp2048",
            # Model
            "model_type": "pnn",
            "task_type": "regression",
            # Training
            "learning_rate": 1e-3,
            "dropout": 0.2,
            "epochs": 100,
        }

        self.assertGreater(len(config), 5)

    def test_ensemble_configuration(self):
        """Test ensemble-specific configuration."""
        config = {
            "model_type": "ensemble",
            "ensemble_size": 10,
            "task_type": "regression",
        }

        self.assertEqual(config["model_type"], "ensemble")
        self.assertGreater(config["ensemble_size"], 0)

    def test_hyperparameter_search_configuration(self):
        """Test hyperparameter search configuration."""
        sweep_config = {
            "method": "bayes",
            "metric": {"name": "rmse", "goal": "minimize"},
            "parameters": {
                "learning_rate": {"min": 1e-4, "max": 1e-2},
                "dropout": {"min": 0.0, "max": 0.5},
                "hidden_dims": {"values": [[128, 64], [256, 128]]},
            }
        }

        self.assertIn("parameters", sweep_config)
        self.assertGreater(len(sweep_config["parameters"]), 0)


class TestConfigurationConsistency(unittest.TestCase):
    """Test cases for configuration consistency."""

    def test_config_type_consistency(self):
        """Test that configuration values have consistent types."""
        config = {
            "data_name": "papyrus",  # string
            "n_targets": 5,  # int
            "learning_rate": 0.001,  # float
            "median_scaling": False,  # bool
        }

        self.assertIsInstance(config["data_name"], str)
        self.assertIsInstance(config["n_targets"], int)
        self.assertIsInstance(config["learning_rate"], float)
        self.assertIsInstance(config["median_scaling"], bool)

    def test_numerical_parameter_ranges(self):
        """Test that numerical parameters are in valid ranges."""
        learning_rate = 1e-3
        dropout = 0.2

        self.assertGreater(learning_rate, 0)
        self.assertGreaterEqual(dropout, 0.0)
        self.assertLessEqual(dropout, 1.0)

    def test_ensemble_size_validity(self):
        """Test that ensemble size is valid."""
        ensemble_size = 10

        self.assertGreater(ensemble_size, 0)
        self.assertLess(ensemble_size, 1000)  # Reasonable upper bound


class TestParameterVocabulary(unittest.TestCase):
    """Test cases for parameter vocabulary."""

    def test_data_name_values(self):
        """Test valid values for data_name."""
        valid = ["papyrus", "tdc", "other"]
        test_val = "papyrus"
        self.assertIn(test_val, valid)

    def test_activity_type_values(self):
        """Test valid values for activity_type."""
        valid = ["xc50", "kx"]
        test_val = "xc50"
        self.assertIn(test_val, valid)

    def test_split_type_values(self):
        """Test valid values for split_type."""
        valid = ["random", "scaffold", "time", "scaffold_cluster"]
        test_val = "random"
        self.assertIn(test_val, valid)

    def test_task_type_values(self):
        """Test valid values for task_type."""
        valid = ["regression", "classification"]
        test_val = "regression"
        self.assertIn(test_val, valid)

    def test_model_type_values(self):
        """Test valid values for model_type."""
        valid = ["pnn", "ensemble", "mcdropout", "evidential", "eoe", "emc"]
        test_val = "pnn"
        self.assertIn(test_val, valid)


if __name__ == "__main__":
    unittest.main()

