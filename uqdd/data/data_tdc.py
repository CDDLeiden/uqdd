import pickle
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tdc.benchmark_group import dti_dg_group
from tdc.multi_pred import DTI
from tdc.chem_utils import MolConvert

from uqdd import DATA_DIR, DATASET_DIR, DEVICE
from uqdd.utils import create_logger, get_config
from uqdd.utils_chem import standardize_df, get_chem_desc
from uqdd.data.utils_data import (
    split_data,
    export_df,
    load_df,
    check_if_processed_file,
    get_tasks,
    export_tasks,
)
from uqdd.utils_prot import get_embeddings


class TDC_DTI:
    def __init__(
        self,
        name="BindingDB_IC50",  # path=DATA_DIR,
        benchmark=False,
        std_smiles=True,
        verbose_files=False,
    ):
        # LOGGING
        self.log = create_logger(name="tdc", file_level="debug", stream_level="info")
        self.config = get_config("tdc")
        self.cols_to_keep = self.config.get("cols_to_keep", None)
        self.benchmark = benchmark
        self.name = name
        self.activity_key = name.split("_")[-1].lower()

        self.out_path, self.tdc_filepath, self.tdc_datapath, self.tdc_file = (
            self._setup_path(self.activity_key, benchmark)
        )
        self.std_smiles, self.verbose_files = std_smiles, verbose_files

        self.df_filtered = self._load_or_process_tdc()

        if self.benchmark:
            self.log.debug(f"Using benchmark data: {self.benchmark}")
            self.process_tdc_benchmark()
        else:
            self.log.debug(f"Processing TDC data: {self.name} from {self.path}")
            self.process_tdc()

    def _load_or_process_tdc(self):
        if self.tdc_file:
            self.log.debug(f"Loading TDC data from {self.tdc_filepath}")
            df = load_df(self.tdc_filepath)
        else:
            self.log.debug(f"Processing TDC data from {self.tdc_datapath}")
            df = self._preprocess_tdc()
        return df

    def _preprocess_tdc(self):
        raise NotImplementedError

    @staticmethod
    def _setup_path(activity_key, benchmark=False):
        # TODO deal with the difference between benchmark and not
        activity_key = activity_key.lower()

        out_path = DATASET_DIR / "tdc" / activity_key
        out_path.mkdir(parents=True, exist_ok=True)

        tdc_filepath = out_path / "tdc_filtered.csv"
        tdc_file = tdc_filepath.is_file()

        tdc_datapath = tdc_filepath if tdc_file else DATA_DIR / "tdc"
        tdc_datapath.mkdir(parents=True, exist_ok=True)

        return out_path, tdc_filepath, tdc_datapath, tdc_file

    def __call__(self, split_type="random"):
        raise NotImplementedError

    def process_tdc(self):
        data = DTI(path=self.path, name=self.name, print_stats=True)
        data.convert_to_log(form="binding")
        data.harmonize_affinities(mode="mean")
        splits = self.data_splitting(data)
        return splits

    def process_tdc_benchmark(self):
        self.log.debug(f"Processing benchmark data: {self.benchmark}")
        group = dti_dg_group(path=self.path)
        splits = self.data_splitting(group)

        return splits

    def data_splitting(self, data_or_group):
        self.log.debug(f"Splitting the data using {self.split_type}")
        splits = dict()
        if self.benchmark:
            # group is time-based
            supported_splits = ["default", "random", "scaffold", "combination", "group"]
            assert (
                self.split_type in supported_splits
            ), f"Split type {self.split_type} not supported"

            benchmark = data_or_group.get("BindingDB_Patent")
            name = benchmark["name"]
            splits["test"] = benchmark["test"]
            # train_val, split["test"] = benchmark["train_val"], benchmark["test"]
            splits["train"], splits["valid"] = data_or_group.get_train_valid_split(
                benchmark=name, split_type=self.split_type, seed=self.seed
            )

        else:
            supported_splits = ["random", "cold_drug", "cold_target", "scaffold"]
            assert (
                self.split_type in supported_splits
            ), f"Split type {self.split_type} not supported"

            if self.split_type in ["random", "cold_drug", "cold_target"]:
                splits = data_or_group.get_split(
                    method=self.split_type, seed=self.seed, frac=[0.7, 0.15, 0.15]
                )

            elif self.split_type == "scaffold":
                # Scaffold-splitting of the data
                df = data_or_group.get_data()  # .rename(columns={"Drug": "smiles"})
                splits["train"], splits["valid"], splits["test"] = scaffold_split(
                    df,
                    smiles_col="Drug",
                    train_frac=0.7,
                    val_frac=0.15,
                    test_frac=0.15,
                    seed=self.seed,
                )

            else:
                raise ValueError(f"Split type {self.split_type} not supported")

        return splits

    def sanitize(self, data_df: pd.DataFrame):
        if self.std_smiles:
            self.log.debug(f"Standardizing SMILES: {self.std_smiles}")
            df_filtered, df_nan, df_dup = standardize_df(
                df=data_df,
                smiles_col="Drug",
                other_dup_col="Target",
                drop=True,
                keep="last",
                logger=self.log,
            )
            if self.verbose_files:
                df_filtered.to_csv(
                    os.path.join(
                        DATA_DIR, f"{today}_tdc_{self.name}_01_standardized.csv"
                    )
                )
                df_dup.to_csv(
                    os.path.join(DATA_DIR, f"{today}_tdc_{self.name}_02_dup_smiles.csv")
                )
                df_nan.to_csv(
                    os.path.join(DATA_DIR, f"{today}_tdc_{self.name}_03_NaN_smiles.csv")
                )

    def get_drug_df(self, df: pd.DataFrame, smiles_col: str = "Drug"):
        return df[[smiles_col]].drop_duplicates(subset=[smiles_col], ignore_index=True)

    def get_protein_df(self, df: pd.DataFrame, target_col: str = "Target"):
        return df[[target_col]].drop_duplicates(subset=[target_col], ignore_index=True)

    def molecular_desc(
        self, df: pd.DataFrame, smiles_col: str, desc_type: str = "ecfp"
    ):
        drugs_df = self.get_drug_df(df, smiles_col=smiles_col)
        desc_type = desc_type.lower()

        if desc_type == "ecfp":
            drugs_df = self.generate_ecfp(drugs_df)
        elif desc_type == "pyg":
            convert = MolConvert(src="SMILES", dst="PyG")
            drugs_df["pyg"] = drugs_df["Drug"].apply(lambda x: convert(x))
        elif desc_type == "dgl":
            convert = MolConvert(src="SMILES", dst="DGL")
            drugs_df["dgl"] = drugs_df["Drug"].apply(lambda x: convert(x))
        elif desc_type == "graph2d":
            convert = MolConvert(src="SMILES", dst="Graph2D")
            drugs_df["graph2d"] = drugs_df["Drug"].apply(lambda x: convert(x))
        else:
            raise ValueError(f"Descriptor type {desc_type} not supported")

        df = pd.merge(df, drugs_df, on=smiles_col, how="left", validate="many_to_many")

        return drugs_df

    def protein_desc(
        self,
        df: pd.DataFrame,
        target_col: str = "Target",
        desc_type: str = "language",
    ):
        # TODO : create language embedder here - ProteinBERT, BioGPT, other protein sequence descriptors
        emb_model = {
            "biobert": "dmis-lab/biobert-v1.1",
            "protbert": "Rostlab/prot_bert",
            "biogpt": "",
        }
        prot_df = self.get_protein_df(df, target_col=target_col)
        desc_type = desc_type.lower()

        if desc_type in emb_model.keys():
            prot_df["biobert"] = prot_df[target_col].apply(
                transformer_featurizer, model_name=emb_model[desc_type]
            )
            prot_
        # BioProt?
        # DescribePROT
        #

        pass

    def generate_ecfp(self, drugs_df):
        self.log.debug("Generating ECFP")
        drugs_df = generate_ecfp(drugs_df, 2, 1024, False, False, smiles_col="Drug")
        drugs_df = generate_ecfp(drugs_df, 4, 2048, False, False, smiles_col="Drug")
        return drugs_df

    def pickle_data(self, output_path=None, name="BindingDB_IC50", **kwargs):
        if output_path is None:
            output_path = os.path.join(DATA_DIR, "dataset", "tdc", name)
            os.makedirs(output_path, exist_ok=True)

        for k, v in kwargs.items():
            with open(os.path.join(output_path, f"{k}.pkl"), "wb") as file:
                self.log.info(f"Pickling {k} to {output_path}")
                pickle.dump(v, file)


def get_datasets():
    raise NotImplementedError


# Dataset Statistics: (# of DTI pairs, # of drugs, # of proteins)

data_kd = DTI(path="uqdd/data/tdc/", name="BindingDB_Kd")  # 52,284/10,665/1,413 for Kd,
data_ki = DTI(
    path="uqdd/data/tdc/", name="BindingDB_Ki"
)  # 375,032/174,662/3,070 for Ki.

data_ic50 = DTI(
    path="uqdd/data/tdc/", name="BindingDB_IC50"
)  # 991,486/549,205/5,078 for IC50, ***
# visualize label distribution
data_ic50.label_distribution()
# data = data_ic50.get_data()

# data.get_data()
data_ic50.harmonize_affinities(mode="mean")
split = data_ic50.get_split(method="random", seed=42, frac=[0.7, 0.15, 0.15])
