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
from uqdd import TODAY, DATA_DIR
from uqdd.utils import create_logger, get_config
from uqdd.chem_utils import standardize_df, generate_ecfp, random_split, scaffold_split, time_split


class Papyrus:
    def __init__(
            self,
            path: str = DATA_DIR,
            accession: Union[None, str, List] = None,
            activity_type: Union[None, str, List] = None,
            organism: Union[None, str, List] = 'Homo sapiens (Human)',
            protein_class: Union[None, str, List] = None,
            std_smiles: bool = True,
            verbose_files: bool = False,
    ):

        # LOGGING
        self.log = create_logger(name="Papyrus", file_level="debug", stream_level="info")

        self.log.info("Initializing -- PAPYRUS -- Module")
        self.config = get_config("papyrus")
        self.papyrus_path = path
        self.papyrus_file = os.path.isfile(path)

        self.chunksize = self.config["chunksize"]

        self.keep_accession = accession
        activity_type = activity_type.lower()
        if isinstance(activity_type, str) and activity_type in ["xc50", "kx"]:
            act_dict = {
                "xc50": ["IC50", "EC50"],
                "kx": ["Ki", "Kd"]
            }

            self.activity_key = activity_type
            activity_type = act_dict[activity_type]
        elif isinstance(activity_type, str) and activity_type in ["ic50", "ec50", "kd", "ki"]:
            self.activity_key = activity_type

        elif isinstance(activity_type, list):
            self.activity_key = "_".join(activity_type)

        self.keep_type = activity_type
        self.keep_organism = organism
        self.keep_protein_class = protein_class
        self.std_smiles = std_smiles
        self.verbose_files = verbose_files

        self.cols_rename_map = self.config.get("cols_rename_map", None)

        dtypes = self.config.get("dtypes", None)

        if self.papyrus_file:
            self.log.info("PapyrusApi processing input from previously processed file")
            self.df_filtered = pd.read_csv(
                self.papyrus_path,
                index_col=0,
                dtype=dtypes,
                low_memory=False
            )
            self.papyrus_data, self.papyrus_protein_data = None, None
        else:
            self.df_filtered = None
            self.papyrus_data, self.papyrus_protein_data = self.reader()

    def __call__(self):
        return self.process_papyrus()

    def _download(self):
        if not self.papyrus_path:
            self.papyrus_path = "/"
            os.makedirs(self.papyrus_path, exist_ok=True)

        self.log.info("Downloading Papyrus data ...")
        download_papyrus(
            outdir=self.papyrus_path,
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

    def reader(self):
        self._download()
        papyrus_data = read_papyrus(
            is3d=False,
            version="latest",
            plusplus=False,
            chunksize=self.chunksize,
            source_path=self.papyrus_path,
        )
        papyrus_protein_data = read_protein_set(source_path=self.papyrus_path, version="latest")
        return papyrus_data, papyrus_protein_data

    def process_papyrus(self):
        self.log.info("Processing Papyrus data ...")
        if not self.papyrus_file:
            filter_1 = keep_quality(
                data=self.papyrus_data, min_quality="High"
            )
            filter_2 = keep_match(
                data=filter_1, column="Protein_Type", values="WT"
            )
            filter_3 = keep_type(
                data=filter_2, activity_types=self.keep_type
            )
            filter_4 = keep_organism(
                data=filter_3, protein_data=self.papyrus_protein_data, organism=self.keep_organism
            )
            self.df_filtered = consume_chunks(filter_4, progress=True, total=int(60000000 / self.chunksize))

            # IMPORTANT - WE HERE HAVE TO SET THE STD_SMILES TO TRUE
            self.std_smiles = True

        # Renaming before any processing for consistency
        self.df_filtered.rename(columns=self.cols_rename_map, inplace=True)

        # SMILES standardization
        if self.std_smiles:
            self.df_filtered, df_nan, df_dup = standardize_df(
                df=self.df_filtered,
                smiles_col="smiles",
                other_dup_col="accession",
                drop=True,
                sorting_col="year",
                keep="last",  # keeps latest year datapoints if duplicated
                logger=self.log,
            )

            if self.verbose_files:
                self.df_filtered.to_csv(f"data/{TODAY}_papyrus_{self.activity_key}_01_standardized.csv")
                df_nan.to_csv(f"data/{TODAY}_papyrus_{self.activity_key}_02_NaN_smiles.csv")
                df_dup.to_csv(f"data/{TODAY}_papyrus_{self.activity_key}_03_dup_smiles.csv")

        return self.df_filtered

    def molecular_descriptors(self):
        self.log.info("Getting the molecular descriptors")

        mol_descriptors = read_molecular_descriptors(
            desc_type="all",
            is3d=False,
            version="latest",
            chunksize=100000,
            source_path=self.papyrus_path,
            ids=self.df_filtered["connectivity"].tolist(),
            verbose=True,
        )

        self.log.info(f"Shape of the molecular descriptors: {mol_descriptors.shape}")

        if self.verbose_files:
            mol_descriptors.to_csv(f"data/{TODAY}_papyrus_{self.activity_key}_04_mol_desc.csv")

        return mol_descriptors

    def protein_descriptors(self):
        self.log.info("Getting the protein descriptors")
        target_ids = self.papyrus_protein_data["target_id"].tolist()
        protein_descriptors = read_protein_descriptors(
            desc_type="all",
            version="latest",
            chunksize=100000,
            source_path=self.papyrus_path,
            ids=target_ids,
            verbose=True,
        )

        if self.verbose_files:
            protein_descriptors.to_csv(f"data/{TODAY}_papyrus_{self.activity_key}_05_protein_desc.csv")

        self.log.info("Protein descriptors shape: {}".format(protein_descriptors.shape))

        return protein_descriptors


def get_data_info(train_data, val_data, test_data, output_path=None):
    combined_data = pd.concat([train_data, val_data, test_data], keys=['train_data', 'val_data', 'test_data'])
    combined_data.reset_index(inplace=True)
    count_data = combined_data.groupby('level_0').count()
    count_data = count_data.pivot_table(columns='level_0')
    count_data.reset_index(inplace=True)
    if output_path:
        count_data.to_csv(os.path.join(output_path, "data_info.csv"), index=False)
    return count_data


def get_top_targets(df, n_top=20):
    # step 1: group the dataframe by protein target
    grouped = df.groupby('accession')
    # step 2: count the number of measurements for each protein target
    counts = grouped['accession'].count()
    # step 3: sort the counts in descending order
    sorted_counts = counts.sort_values(ascending=False, by='counts')  # TODO check by='counts'?
    # step 4: select the 20 protein targets with the highest counts
    top_targets = sorted_counts.head(n_top)
    # step 5: filter the original dataframe to only include rows corresponding to these 20 protein targets
    filtered_df = df[df['accession'].isin(top_targets.index)]
    # step 6: filter the dataframe to only include rows with a pchembl value mean
    filtered_df = filtered_df[filtered_df['pchemblValueMean'].notna()]

    return filtered_df, top_targets


def data_preparation(
        papyrus_path: str = DATA_DIR,
        activity="xc50",
        organism=None,
        n_top=-1,  # 20,
        multitask=True,
        std_smiles=True,
        split_type='random',
        output_path="data/dataset/",
        verbose_files=False,
):
    assert activity.lower() in ["xc50", "kx"], "activity must be either xc50 or kx"
    assert isinstance(n_top, int), "n_top must be an integer"
    assert isinstance(std_smiles, bool), "std_smiles must be a boolean"
    assert split_type.lower() in ["random", "scaffold", "time"], "split_type must be either random or scaffold or time"
    os.makedirs(output_path, exist_ok=True)

    # Read the data
    papyrus_ = Papyrus(
        path=papyrus_path,
        accession=None,
        activity_type=activity,
        organism=organism,
        protein_class=None,
        std_smiles=std_smiles,
        verbose_files=verbose_files,
    )
    df = papyrus_()

    if n_top > 0:
        df, top_targets = get_top_targets(df, n_top)

        if multitask:
            pivoted = pd.pivot_table(
                df,
                index='smiles',
                columns='accession',
                values='pchemblValueMean',
                aggfunc='first'
            )
            # reset the index to make the 'smiles' column a regular column
            pivoted = pivoted.reset_index()
            # replace any missing values with NaN
            df = pivoted.fillna(value=np.nan)
            target_col = list(top_targets.index)

        else:
            df = df[["smiles", "accession", "pchemblValueMean", "year"]]
            target_col = ["pchemblValueMean"]
        target_col_path = os.path.join(output_path, "target_col.pkl")
        with open(target_col_path, 'wb') as file:
            pickle.dump(target_col, file)

    # step 8: generate the ecfp descriptors
    df = generate_ecfp(df, 2, 1024, False, False)
    df = generate_ecfp(df, 4, 2048, False, False)

    if split_type == 'random':
        train_data, val_data, test_data = random_split(
            df, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42
        )

    elif split_type == 'scaffold':
        # Scaffold-splitting of the data
        train_data, val_data, test_data = scaffold_split(
            df, smiles_col='smiles', train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42
        )
    elif split_type == 'time':
        # Time-splitting of the data -
        train_data, val_data, test_data = time_split(
            df, time_col='year', train_frac=0.7, val_frac=0.15, test_frac=0.15
        )
    else:
        raise ValueError("split_type must be either random or scaffold or time")

    data_info = get_data_info(train_data, val_data, test_data, output_path)
    data = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "all": df,
        "data_info": data_info
    }
    for name, subset in data.items():
        file_path = os.path.join(output_path, f"{name}.pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(subset, file)

    return train_data, val_data, test_data, df


class PapyrusDataset(Dataset):
    def __init__(
            self,
            file_path: Union[str, Path] = "data/dataset/xc50/random/train.pkl",
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
        with open(file_path, 'rb') as file:
            self.data = pickle.load(file)

        self.input_col = input_col.lower()
        if target_col is None:
            # Assuming your DataFrame is named 'df'
            target_col_path = os.path.join(folder_path, "target_col.pkl")
            with open(target_col_path, 'rb') as file:
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
