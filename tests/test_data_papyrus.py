import unittest
from unittest import mock
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Module under test
import uqdd.data.data_papyrus as dp


class TestPapyrusParseActivityKey(unittest.TestCase):
    def test_parse_activity_key_valid(self):
        key, types = dp.Papyrus._parse_activity_key("xc50")
        self.assertEqual(key, "xc50")
        self.assertEqual(types, ["IC50", "EC50"])

        key, types = dp.Papyrus._parse_activity_key("kx")
        self.assertEqual(key, "kx")
        self.assertEqual(types, ["Ki", "Kd"])

    def test_parse_activity_key_invalid(self):
        with self.assertRaises(ValueError):
            dp.Papyrus._parse_activity_key("invalid")


class TestPapyrusGetSplitTypes(unittest.TestCase):
    def test_get_split_types_all_single_task(self):
        p = self._make_minimal_papyrus(mt=False)
        types = p._get_split_types("all")
        self.assertListEqual(types, ["random", "scaffold", "time", "scaffold_cluster"])

    def test_get_split_types_all_multitask(self):
        p = self._make_minimal_papyrus(mt=True)
        types = p._get_split_types("all")
        self.assertListEqual(types, ["random", "scaffold", "scaffold_cluster"])

    def test_get_split_types_specific(self):
        p = self._make_minimal_papyrus(mt=False)
        types = p._get_split_types("random")
        self.assertListEqual(types, ["random"])

    def _make_minimal_papyrus(self, mt: bool):
        # Create a Papyrus instance with minimal mocks
        with patch.object(dp.Papyrus, "load_or_process_papyrus", return_value=(pd.DataFrame(), pd.DataFrame())):
            p = dp.Papyrus(activity_type="xc50")
        p.MT = mt
        return p


class TestPapyrusGetColsToInclude(unittest.TestCase):
    def test_get_cols_single_task_with_protein(self):
        cols = dp.Papyrus.get_cols_to_include("ankh-base", "ecfp2048", -1, ["pchembl_value_Mean"])
        self.assertIn("SMILES", cols)
        self.assertIn("connectivity", cols)
        self.assertIn("ecfp2048", cols)
        self.assertIn("target_id", cols)
        self.assertIn("ankh-base", cols)
        self.assertIn("Year", cols)
        self.assertIn("pchembl_value_Mean", cols)

    def test_get_cols_single_task_without_protein(self):
        cols = dp.Papyrus.get_cols_to_include(None, "ecfp2048", -1, ["pchembl_value_Mean"])
        self.assertIn("SMILES", cols)
        self.assertIn("connectivity", cols)
        self.assertIn("ecfp2048", cols)
        self.assertNotIn("target_id", cols)
        self.assertNotIn("Year", cols)

    def test_get_cols_multitask(self):
        cols = dp.Papyrus.get_cols_to_include("ankh-base", "ecfp2048", 10, ["t1", "t2"])
        self.assertIn("SMILES", cols)
        self.assertIn("connectivity", cols)
        self.assertIn("ecfp2048", cols)
        self.assertNotIn("ankh-base", cols)
        self.assertIn("t1", cols)
        self.assertIn("t2", cols)


class TestPapyrusPrepareQueryCol(unittest.TestCase):
    def setUp(self):
        with patch.object(dp.Papyrus, "load_or_process_papyrus", return_value=(pd.DataFrame(), pd.DataFrame())):
            self.p = dp.Papyrus(activity_type="xc50")

    def test_prepare_query_none(self):
        q, df = self.p._prepare_query_col(pd.DataFrame(), None)
        self.assertIsNone(q)

    def test_prepare_query_protein_desc(self):
        df = pd.DataFrame({"target_id": ["P1"], "Sequence": ["AAAA"], "SMILES": ["C"], "connectivity": ["c1"]})
        self.p.papyrus_protein_data = pd.DataFrame({"target_id": ["P1"], "Sequence": ["AAAA"]})
        q, df_out = self.p._prepare_query_col(df.drop(columns=["Sequence"]), "ankh-base")
        self.assertEqual(q, "Sequence")
        self.assertIn("Sequence", df_out.columns)

    def test_prepare_query_unirep(self):
        q, _ = self.p._prepare_query_col(pd.DataFrame({"target_id": ["P1"]}), "unirep")
        self.assertEqual(q, "target_id")

    def test_prepare_query_chem_connectivity(self):
        q, _ = self.p._prepare_query_col(pd.DataFrame({"connectivity": ["c"]}), "mordred")
        self.assertEqual(q, "connectivity")

    def test_prepare_query_chem_smiles(self):
        q, _ = self.p._prepare_query_col(pd.DataFrame({"SMILES": ["C"]}), "ecfp2048")
        self.assertEqual(q, "SMILES")

    def test_prepare_query_unsupported(self):
        with self.assertRaises(ValueError):
            self.p._prepare_query_col(pd.DataFrame(), "unknown_desc")


class TestPapyrusMergeDescriptors(unittest.TestCase):
    def setUp(self):
        with patch.object(dp.Papyrus, "load_or_process_papyrus", return_value=(pd.DataFrame(), pd.DataFrame())):
            self.p = dp.Papyrus(activity_type="xc50")

    @patch("uqdd.utils_chem.get_chem_desc")
    @patch("uqdd.utils_prot.get_embeddings")
    def test_merge_descriptors_calls_helpers(self, mock_get_embeddings, mock_get_chem_desc):
        df = pd.DataFrame({
            "SMILES": ["C"],
            "connectivity": ["c1"],
            "target_id": ["P1"],
            "Sequence": ["AAAA"],
            "pchembl_value_Mean": [6.0],
        })
        mock_get_chem_desc.side_effect = lambda df_in, desc, qc, **kw: df_in.assign(**{desc: [[0.1, 0.2]]})
        mock_get_embeddings.side_effect = lambda df_in, desc, qc, **kw: df_in.assign(**{desc: [[0.3, 0.4]]})

        out = self.p.merge_descriptors(df.copy(), desc_prot="ankh-base", desc_chem="ecfp2048", batch_size=2)
        self.assertIn("ecfp2048", out.columns)
        self.assertIn("ankh-base", out.columns)
        mock_get_chem_desc.assert_called()
        mock_get_embeddings.assert_called()

    def test_merge_desc_skips_when_none(self):
        df = pd.DataFrame({"SMILES": ["C"], "connectivity": ["c1"]})
        out = self.p.merge_descriptors(df.copy(), desc_prot=None, desc_chem=None)
        self.assertTrue(out.equals(df))


class TestPapyrusMergeProteinSequences(unittest.TestCase):
    def setUp(self):
        with patch.object(dp.Papyrus, "load_or_process_papyrus", return_value=(pd.DataFrame(), pd.DataFrame())):
            self.p = dp.Papyrus(activity_type="xc50")
        self.p.papyrus_protein_data = pd.DataFrame({"target_id": ["P1"], "Sequence": ["AAAA"]})

    def test_merge_protein_sequences_adds_sequence(self):
        df = pd.DataFrame({"target_id": ["P1"], "SMILES": ["C"]})
        out = self.p.merge_protein_sequences(df)
        self.assertIn("Sequence", out.columns)
        self.assertEqual(out.loc[0, "Sequence"], "AAAA")

    def test_merge_protein_sequences_noop_if_exists(self):
        df = pd.DataFrame({"target_id": ["P1"], "Sequence": ["BBBB"]})
        out = self.p.merge_protein_sequences(df)
        self.assertEqual(out.loc[0, "Sequence"], "BBBB")


class TestPapyrusCallAssertions(unittest.TestCase):
    def setUp(self):
        with patch.object(dp.Papyrus, "load_or_process_papyrus", return_value=(pd.DataFrame(), pd.DataFrame())):
            self.p = dp.Papyrus(activity_type="xc50")

    def test_assertions_multitask_protein_set_to_none(self):
        adjusted = self.p._call_assertions("ankh-base", "ecfp2048", False, n_targets=5)
        self.assertIsNone(adjusted)
        self.assertTrue(self.p.MT)

    def test_assertions_require_descriptor(self):
        with self.assertRaises(AssertionError):
            self.p._call_assertions(None, None, False, n_targets=-1)


class TestPapyrusDataset(unittest.TestCase):
    @patch("uqdd.data.data_papyrus.load_df")
    def test_dataset_regression_and_classification(self, mock_load_df):
        df = pd.DataFrame({
            "pchembl_value_Mean": [5.0, 7.0, 6.0],
            "ecfp2048": [[0, 1], [1, 0], [0.5, 0.5]],
            "ankh-base": [[0.1], [0.2], [0.3]],
        })
        mock_load_df.return_value = df

        # Regression without protein
        ds_reg = dp.PapyrusDataset(
            file_path=Path("/tmp/fake/train.pkl"),
            desc_prot=None,
            desc_chem="ecfp2048",
            task_type="regression",
            calc_median=True,
            median_scaling=False,
        )
        self.assertEqual(len(ds_reg), 3)
        self.assertEqual(ds_reg.labels.shape, torch.Size([3, 1]))
        self.assertEqual(ds_reg.chem_desc.shape, torch.Size([3, 2]))
        self.assertEqual(ds_reg.prot_desc.shape[1], 1)  # placeholder when None

        # Classification with protein
        ds_cls = dp.PapyrusDataset(
            file_path=Path("/tmp/fake/train.pkl"),
            desc_prot="ankh-base",
            desc_chem="ecfp2048",
            task_type="classification",
            calc_median=True,
            median_scaling=False,
            median_point=6.0,
        )
        # labels should be binarized around median_point
        self.assertTrue(torch.equal(ds_cls.labels.squeeze(), torch.tensor([0., 1., 0.])))
        self.assertEqual(ds_cls.prot_desc.shape, torch.Size([3, 1]))

    @patch("uqdd.data.data_papyrus.load_df")
    def test_dataset_handles_numeric_conversion(self, mock_load_df):
        df = pd.DataFrame({
            "pchembl_value_Mean": [6.0],
            "ecfp2048": [["1", "2"]],
        })
        mock_load_df.return_value = df
        ds = dp.PapyrusDataset(
            file_path=Path("/tmp/fake/train.pkl"),
            desc_prot=None,
            desc_chem="ecfp2048",
        )
        self.assertTrue(torch.equal(ds.chem_desc.squeeze(), torch.tensor([1.0, 2.0])))


class TestGetDatasets(unittest.TestCase):
    @patch("uqdd.data.data_papyrus.Papyrus")
    @patch("uqdd.data.data_papyrus.create_logger")
    def test_get_datasets_path_and_auto_generate(self, mock_logger, mock_papyrus):
        # Simulate files not present first call, present after Papyrus() is invoked
        base = dp.DATASET_DIR / "papyrus" / "xc50" / "all"
        train = base / "random_ecfp2048_train.pkl"
        val = base / "random_ecfp2048_val.pkl"
        test = base / "random_ecfp2048_test.pkl"

        with patch.object(Path, "is_file", side_effect=lambda self: str(self).endswith("train.pkl")):
            # Mock dataset construction to avoid reading actual files
            with patch("uqdd.data.data_papyrus.PapyrusDataset", return_value=MagicMock(median_point=6.0)) as mock_ds:
                ds = dp.get_datasets(
                    n_targets=-1,
                    activity_type="xc50",
                    split_type="random",
                    desc_prot=None,
                    desc_chem="ecfp2048",
                    median_scaling=False,
                    task_type="regression",
                    ext="pkl",
                    device="cpu",
                )
                self.assertIn("train", ds)
                mock_papyrus.assert_called()
                mock_ds.assert_called()

    @patch("uqdd.data.data_papyrus.create_logger")
    def test_get_datasets_filename_prefixes(self, mock_logger):
        # Single task, no protein
        dir_path = dp.DATASET_DIR / "papyrus" / "xc50" / "all"
        paths = [
            dir_path / "random_ecfp2048_train.pkl",
            dir_path / "random_ecfp2048_val.pkl",
            dir_path / "random_ecfp2048_test.pkl",
        ]
        with patch.object(Path, "is_file", return_value=True):
            with patch("uqdd.data.data_papyrus.PapyrusDataset", return_value=MagicMock(median_point=6.0)):
                ds = dp.get_datasets(
                    n_targets=-1,
                    activity_type="xc50",
                    split_type="random",
                    desc_prot=None,
                    desc_chem="ecfp2048",
                )
                self.assertIn("train", ds)

        # Single task, with protein
        with patch.object(Path, "is_file", return_value=True):
            with patch("uqdd.data.data_papyrus.PapyrusDataset", return_value=MagicMock(median_point=6.0)):
                ds = dp.get_datasets(
                    n_targets=-1,
                    activity_type="xc50",
                    split_type="random",
                    desc_prot="ankh-base",
                    desc_chem="ecfp2048",
                )
                self.assertIn("train", ds)

        # Multitask: prefix should not include protein
        with patch.object(Path, "is_file", return_value=True):
            with patch("uqdd.data.data_papyrus.PapyrusDataset", return_value=MagicMock(median_point=6.0)):
                ds = dp.get_datasets(
                    n_targets=5,
                    activity_type="xc50",
                    split_type="random",
                    desc_prot="ankh-base",
                    desc_chem="ecfp2048",
                )
                self.assertIn("train", ds)


if __name__ == "__main__":
    unittest.main()
