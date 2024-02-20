import os
import pickle
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import torch
from papyrus_scripts.download import download_papyrus
from papyrus_scripts.preprocess import (
    keep_quality,
    keep_match,
    keep_type,
    keep_organism,
    consume_chunks
)
from papyrus_scripts.reader import (
    read_papyrus,
    read_protein_set,
    read_molecular_descriptors,
    read_protein_descriptors,
)
from torch.utils.data import Dataset
from uqdd import TODAY, DATA_DIR, DATASET_DIR
from uqdd.utils import create_logger, get_config
from uqdd.utils_chem import standardize_df, generate_ecfp
from uqdd.data.utils_data import split_data


class Papyrus:
    def __init__(
            self,
            accession: Union[None, str, List] = None,
            activity_type: Union[None, str, List] = None,
            std_smiles: bool = True,
            verbose_files: bool = False,
    ):
        # LOGGING
        self.log = create_logger(name="Papyrus", file_level="debug", stream_level="info")
        self.config = get_config("papyrus")
        self.chunksize = self.config.get("chunksize", 1000000)
        self.cols_to_keep = self.config.get("cols_to_keep", None)
        self.dtypes, self.dtypes_protein = self.config.get("dtypes", {}), self.config.get("dtypes_protein", {})
        self.keep_accession = accession
        self.activity_key, self.activity_type = self._parse_activity_key(activity_type)
        self.output_path, self.pap_filepath, self.pap_path, self.pap_file, self.pap_protpath = self._setup_path(
            self.activity_type
        )
        self.std_smiles, self.verbose_files = std_smiles, verbose_files
        self.df_filtered, self.papyrus_protein_data = self._load_or_process_file()
        # if self.pap_file:
        #     self.log.info("PapyrusApi processing input from previously processed files")
        #     self.df_filtered = pd.read_csv(
        #         self.pap_path,
        #         index_col=0,
        #         dtype=dtypes,
        #         low_memory=False
        #     )
        #     self.papyrus_protein_data = pd.read_csv(
        #         self.pap_protpath,
        #         index_col=0,
        #         dtype=dtypes_protein,
        #         low_memory=False
        #     )
        #
        #     # output is the same dir as input file
        #     self.output_path = os.path.dirname(self.pap_path)
        #
        # else:
        #
        #     self.log.info("PapyrusApi processing input from Raw Papyrus database")
        #     papyrus_data, papyrus_protein_data = self._reader()
        #     self.df_filtered = self._filter_raw_file(papyrus_data, papyrus_protein_data)
        #
        #     # Forcing verbose here to avoid reprocessing
        #     self.df_filtered.to_csv(self.pap_filepath)
        #     papyrus_protein_data.to_csv(self.pap_protpath)
        #     self.papyrus_protein_data = papyrus_protein_data
        #
        #     os.makedirs(self.output_path, exist_ok=True)
    
    def __call__(self):
        return self._process_papyrus()

    def merge_descriptors(
            self,
            desc_prot,
            desc_chem,
            df=None,
            target_id_col="target_id",
            connectivity_col="connectivity"
    ):
        # "unirep", "esm", "protbert", "msa", "all"
        # "mold2", "mordred", "cddd", "fingerprint", "moe", "all"
        if df is None:
            df = self.df_filtered

        if desc_prot:
            self._merge_protein_descriptors(
                df,
                target_id_col=target_id_col,
                desc_type=desc_prot
            )

        if desc_chem:
            self._merge_molecular_descriptors(
                df,
                connectivity_id_col=connectivity_col,
                desc_type=desc_chem
            )

        return df

    def merge_protein_sequences(
            self,
            df,
            target_id_col="target_id"
    ):
        return df.merge(
            self.papyrus_protein_data[["target_id", "Sequence"]],
            left_on=target_id_col,
            right_on="target_id",
            how="left",
        )

    @staticmethod
    def _setup_path(activity_type):
        activity_type = activity_type.lower()
        out_path = DATASET_DIR / "papyrus" / activity_type
        out_path.mkdir(parents=True, exist_ok=True)
        pap_filepath = out_path / "papyrus_filtered_raw.csv"
        pap_file = pap_filepath.is_file()
        pap_path = out_path if pap_file else DATA_DIR
        pap_protpath = DATASET_DIR / "papyrus" / "papyrus_proteins_raw.csv"
        #
        # out_path = os.path.join(
        #     DATASET_DIR,
        #     "papyrus",
        #     activity_type,
        # )
        # os.makedirs(out_path, exist_ok=True)
        #
        # pap_filepath = os.path.join(
        #     str(out_path),
        #     "papyrus_filtered_raw.csv"
        # )
        # pap_file = os.path.isfile(pap_filepath)
        # where download will create the folder papyrus if to be downloaded else file path
        # pap_path = pap_filepath if pap_file else DATA_DIR
        # pap_protpath = os.path.join(DATASET_DIR, "papyrus", "papyrus_proteins_raw.csv")

        return out_path, pap_filepath, pap_path, pap_file, pap_protpath

    def _load_or_process_file(self):
        if self.pap_file:
            self.log.info("Loading previously processed Papyrus data ...")
            df_filtered = pd.read_csv(
                self.pap_path,
                index_col=0,
                dtype=self.dtypes,
                low_memory=False
            )
            papyrus_protein_data = pd.read_csv(
                self.pap_protpath,
                index_col=0,
                dtype=self.dtypes_protein,
                low_memory=False
            )
        else:
            self.log.info("PapyrusApi processing input from Raw Papyrus database")
            papyrus_data, papyrus_protein_data = self._reader()
            df_filtered = self._filter_raw_file(papyrus_data, papyrus_protein_data)
            # Forcing verbose here to avoid reprocessing
            df_filtered.to_csv(self.pap_filepath)
            papyrus_protein_data.to_csv(self.pap_protpath)

        return df_filtered, papyrus_protein_data

    def _process_papyrus(self):
        self.log.info("Processing Papyrus data ...")

        df_nan, df_dup = None, None
        # SMILES standardization
        if self.std_smiles:
            self.df_filtered, df_nan, df_dup = self._sanitize_smiles()

        if self.cols_to_keep:
            self.df_filtered = self.df_filtered[self.cols_to_keep]

        # Verbosing files
        if self.verbose_files:
            self._verbosing_files(self.df_filtered, df_nan, df_dup)

        return self.df_filtered

    @staticmethod
    def _parse_activity_key(activity_type):
        activity_type = activity_type.lower()
        if isinstance(activity_type, str) and activity_type in ["xc50", "kx"]:
            act_dict = {
                "xc50": ["IC50", "EC50"],
                "kx": ["Ki", "Kd"]
            }
            activity_key = activity_type
            activity_type = act_dict[activity_type]
        elif isinstance(activity_type, str) and activity_type in ["ic50", "ec50", "kd", "ki"]:
            activity_key = activity_type

        elif isinstance(activity_type, list):
            activity_key = "_".join(activity_type)
        else:
            raise ValueError("activity_type must be a string or a list of strings")
        return activity_key, activity_type

    def _download(self):
        self.log.info("Downloading Papyrus data ...")
        download_papyrus(
            outdir=self.pap_path,
            version="latest",
            nostereo=True,
            stereo=False,
            only_pp=False,
            structures=False,
            descriptors="all",
            progress=True,
            disk_margin=0.01,
        )

        return None

    def _reader(self):
        self._download()
        papyrus_data = read_papyrus(
            is3d=False,
            version="latest",
            plusplus=True,
            chunksize=self.chunksize,
            source_path=self.pap_path,
        )
        papyrus_protein_data = read_protein_set(source_path=self.pap_path, version="latest")
        return papyrus_data, papyrus_protein_data

    def _filter_raw_file(self, papyrus_data, papyrus_protein_data):
        filter_1 = keep_quality(
            data=papyrus_data, min_quality="High"
        )
        filter_2 = keep_match(
            data=filter_1, column="Protein_Type", values="WT"
        )
        filter_3 = keep_type(
            data=filter_2, activity_types=self.activity_type
        )
        filter_4 = keep_organism(
            data=filter_3, protein_data=papyrus_protein_data, organism="Homo sapiens (Human)"  # self.keep_organism
        )
        df_filtered = consume_chunks(filter_4, progress=True, total=int(60000000 / self.chunksize))

        # IMPORTANT - WE HERE HAVE TO SET THE STD_SMILES TO TRUE
        self.std_smiles = True

        return df_filtered

    def _sanitize_smiles(self):
        self.log.info("Sanitizing the SMILES")
        df_filtered, df_nan, df_dup = standardize_df(
            df=self.df_filtered,
            smiles_col="SMILES",
            other_dup_col="accession",
            drop=True,
            sorting_col="Year",
            keep="last",  # keeps latest year datapoints if duplicated
            logger=self.log,
        )

        return df_filtered, df_nan, df_dup

    @staticmethod
    def _merge_cols(row):
        return np.array(row[1:])

    def _get_protein_descriptors(self, df, target_id_col, desc_type="unirep"):
        self.log.info("Getting the protein descriptors")
        desc_type = desc_type.lower()
        target_ids = df[target_id_col].tolist()

        protein_descriptors = read_protein_descriptors(
            desc_type=desc_type,
            version="latest",
            chunksize=100000,
            source_path=DATA_DIR,
            ids=target_ids,
            verbose=True,
        )

        protein_descriptors[desc_type] = protein_descriptors.apply(self._merge_cols, axis=1)

        protein_descriptors = protein_descriptors[["target_id", desc_type]]

        self.log.info("Protein descriptors shape: {}".format(protein_descriptors.shape))

        return protein_descriptors

    def _get_molecular_descriptors(self, df, connectivity_id_col, desc_type="cddd"):
        # "mold2", "mordred", "cddd", "fingerprint", "moe", "all"
        desc_type = desc_type.lower()
        connectivity_ids = df[connectivity_id_col].tolist()
        self.log.info("Getting the molecular descriptors")

        mol_descriptors = read_molecular_descriptors(
            desc_type=desc_type,
            is3d=False,
            version="latest",
            chunksize=100000,
            source_path=DATA_DIR,
            ids=connectivity_ids,
            verbose=True,
        )

        mol_descriptors[desc_type] = mol_descriptors.apply(self._merge_cols, axis=1)

        mol_descriptors = mol_descriptors[["connectivity", desc_type]]

        self.log.info(f"Shape of the molecular descriptors: {mol_descriptors.shape}")

        return mol_descriptors

    def _merge_protein_descriptors(
            self,
            df,
            target_id_col="target_id",
            desc_type="unirep"
    ):
        self.log.info("Merging protein descriptors")
        if isinstance(desc_type, list):
            for desc in desc_type:
                df = self._merge_protein_descriptors(
                    df,
                    target_id_col,
                    desc_type=desc
                )

        elif isinstance(desc_type, str):
            protein_descriptors = self._get_protein_descriptors(df, target_id_col, desc_type=desc_type)
            df = df.merge(
                protein_descriptors,
                left_on=target_id_col,
                right_on="target_id",
                how="left",
            )

        return df

    def _merge_molecular_descriptors(
            self,
            df,
            connectivity_id_col="connectivity",
            desc_type="cddd"
    ):
        self.log.info("Merging molecular descriptors")
        if isinstance(desc_type, list):
            for desc in desc_type:
                df = self._merge_molecular_descriptors(
                    df,
                    connectivity_id_col,
                    desc_type=desc
                )

        elif isinstance(desc_type, str):
            mol_descriptors = self._get_molecular_descriptors(df, connectivity_id_col, desc_type=desc_type)
            df = df.merge(
                mol_descriptors,
                left_on=connectivity_id_col,
                right_on="connectivity",
                how="left",
            )

        return df

    def _verbosing_files(self, processed_df, df_nan, df_dup):
        data_dict = {
            "papyrus_processed.csv": processed_df,
            "papyrus_NaN_smiles.csv": df_nan,
            "papyrus_dup_smiles.csv": df_dup,
        }
        for file, data in data_dict.items():
            if data:
                data.to_csv(os.path.join(self.output_path, f"{file}"))


class PapyrusDataProcessor:
    """
    A class for processing and preparing biochemical papyrus data for analysis.
    """
    def __init__(
            self,
            activity="xc50",
            std_smiles=True,
            add_protein_sequence=True,
            verbose_files=False
    ):
        self.log = create_logger(name="PapyrusDataProcessor", file_level="debug", stream_level="info")
        self.std_smiles = std_smiles
        self.verbose_files = verbose_files
        self.activity = activity
        self.add_protein_sequence = add_protein_sequence

        # Read the data
        # We will only use the desc_prot and desc
        self.papyrus_obj = Papyrus(
            accession=None,
            activity_type=activity,
            std_smiles=self.std_smiles,
            verbose_files=self.verbose_files,
        )
        self.papyrus_df = self.papyrus_obj()

        self.output_path = self.papyrus_obj.output_path

    def __call__(
            self,
            n_top: int = -1,
            # multitask: bool = True,
            split_type: str = "random",
            desc_prot: Union[str, List[str], None] = None,
            desc_chem: Union[str, List[str], None] = None
    ):
        return self._data_preparation(
            n_top,
            # multitask,
            split_type,
            desc_prot,
            desc_chem
        )

    def _data_preparation(
            self,
            n_top=20,
            # multitask=True,
            split_type="random",
            desc_prot=None,
            desc_chem=None
    ):
        label_col = ["pchembl_value_Mean"]
        if n_top > 0:
            df, top_targets = self._get_top_targets(self.papyrus_df, n_top)
            self.output_path = os.path.join(self.output_path, f"top{n_top}")
            # if multitask:
            pivoted = pd.pivot_table(
                df,
                index="SMILES",
                columns="accession",
                values="pchembl_value_Mean",
                aggfunc="first"
            )
            # reset the index to make the "smiles" column a regular column
            pivoted = pivoted.reset_index()
            # replace any missing values with NaN
            df = pivoted.fillna(value=np.nan)
            label_col = list(top_targets.index)
                # if desc_prot:
                #     self.log.warning("Multitask learning with protein descriptors not supported")

            # else:
                # df = df[[
                #     "SMILES",
                #     "connectivity",
                #     "target_id",
                #     "accession",
                #     "pchembl_value_Mean",
                #     "Year"
                # ]]
        else:
            # dataset/papyrus/xc50/all/
            self.output_path = os.path.join(self.output_path, "all")
            df = self.papyrus_df.copy()  # the whole thing

        df = self._get_descriptors(
            df,
            desc_prot,
            desc_chem,
        )

        # Split the data
        all_data = split_data(
            df,
            split_type=split_type,
            smiles_col="SMILES",
            time_col="Year",
            train_frac=0.7,
            val_frac=0.15,
            test_frac=0.15,
            seed=42,
            output_path=self.output_path,
            label_col=label_col
        )

        return all_data

    @staticmethod
    def _get_top_targets(df, n_top=20):
        # step 1: group the dataframe by protein target
        grouped = df.groupby("accession")
        # step 2: count the number of measurements for each protein target
        counts = grouped["accession"].count()
        # step 3: sort the counts in descending order
        sorted_counts = counts.sort_values(ascending=False, by="counts")
        # step 4: select the x protein targets with the highest counts
        top_targets = sorted_counts.head(n_top)
        # step 5: filter the original dataframe to only include rows corresponding to these x protein targets
        filtered_df = df[df["accession"].isin(top_targets.index)]
        # step 6: filter the dataframe to only include rows with a pchembl value mean
        filtered_df = filtered_df[filtered_df["pchembl_value_Mean"].notna()]
        return filtered_df, top_targets

    def _get_descriptors(
            self,
            df,
            desc_prot,
            desc_chem,
            n_top=-1
    ):
        if n_top > 0 and desc_prot:
            self.log.warning("Multitask learning with protein descriptors not supported - desc_prot will be ignored")
            desc_prot = None
        desc_prot_from_papyrus = desc_prot in ["unirep", "prodec"]
        desc_chem_from_papyrus = desc_chem in ["mold2", "mordred", "cddd", "fingerprint", "moe"]

        if desc_prot_from_papyrus and desc_chem_from_papyrus:
            df = self.papyrus_obj.merge_descriptors(
                    desc_prot=desc_prot,
                    desc_chem=desc_chem,
                    df=df,
                    target_id_col="accession",
                    connectivity_col="connectivity"
                )
        elif desc_prot_from_papyrus: # desc_chem_from_papyrus is False
            pass
        elif desc_chem_from_papyrus: # desc_prot_from_papyrus is False

            pass
        else:
            pass



        df = self.papyrus_obj.merge_descriptors(
            desc_prot=desc_prot,
            desc_chem=desc_chem,
            df=df,
            target_id_col="targetIds",
            connectivity_col="connectivity"
        )

        raise NotImplementedError

    def _get_other_prot_desc(self, df, desc_prot):
        raise NotImplementedError

    def _get_other_mol_desc(self, df, desc_chem):
        raise NotImplementedError


class PapyrusDatasetMT(Dataset):
    def __init__(
            self,
            file_path: Union[str, Path] = os.path.join(DATASET_DIR, "xc50", "all", "random", "train.pkl"),
            input_col: str = "ecfp1024",
            target_col: Union[str, List, None] = None,
            device: object = "cuda",
    ) -> None:
        """
        Parameters
        ----------
        file_path: str or Path
        input_col: str
        target_col: str or List or None
        device: object
        """
        folder_path = os.path.dirname(file_path)
        with open(file_path, "rb") as file:
            self.data = pickle.load(file)

        self.input_col = input_col.lower()

        if target_col is None:  # has to be changed if all and not MT
            # Assuming your DataFrame is named "df"
            target_col_path = os.path.join(folder_path, "target_col.pkl")
            with open(target_col_path, "rb") as file:
                target_col = pickle.load(file)

        self.target_col = target_col

        self.input_data = torch.from_numpy(np.stack(self.data[self.input_col].values)).to(torch.float).to(device)
        self.target_data = torch.tensor(self.data[self.target_col].values).to(torch.float).to(device)

        del self.data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_sample = self.input_data[idx]
        y_sample = self.target_data[idx]

        return x_sample, y_sample


class PapyrusDataset(Dataset):
    def __init__(
            self,
            file_path: Union[str, Path] = os.path.join(DATASET_DIR, "all", "xc50", "random", "train.pkl"),
            chem_xcol: str = "ecfp1024",
            prot_xcol: str = "protein_embeddings",
            label_ycol: Union[str, List, None] = "pchembl_value_Mean",
            # input_col: str = "ecfp1024",
            device: object = "cuda",
    ) -> None:
        folder_path = os.path.dirname(file_path)
        with open(file_path, "rb") as file:
            self.data = pickle.load(file)

        self.chem_xcol = chem_xcol.lower()
        self.prot_xcol = prot_xcol.lower()
        # TODO check if label_ycol is None
        # if label_ycol is None: # MT

        self.label_ycol = label_ycol

        if label_ycol is None:  # has to be changed if all and not MT
            # Assuming your DataFrame is named "df"
            target_col_path = os.path.join(folder_path, "target_col.pkl")
            with open(target_col_path, "rb") as file:
                target_col = pickle.load(file)

        self.target_col = target_col

        self.input_data = torch.from_numpy(np.stack(self.data[self.chem_xcol].values)).to(torch.float).to(device)
        self.target_data = torch.tensor(self.data[self.target_col].values).to(torch.float).to(device)

        del self.data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_sample = self.input_data[idx]
        y_sample = self.target_data[idx]

        return x_sample, y_sample


if __name__ == "__main__":
    papyrus_ = Papyrus(
        accession=None,
        activity_type="xc50",
        std_smiles=True,
        verbose_files=True
    )

    pap_df = papyrus_()

    print(pap_df.shape)


