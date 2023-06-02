__author__ = "Bola Khalil"
__copyright__ = "Copyright 2022, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__license__ = "All rights reserved, Janssen Pharmaceutica NV & Johannes-Kepler Universität Linz"
__version__ = "0.0.1"
__maintainer__ = "Bola Khalil"
__email__ = "bkhalil@its.jnj.com"
__status__ = "Development"

import os
from typing import Union, List, Tuple
import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.model_selection import train_test_split
from papyrus_scripts.download import download_papyrus
from papyrus_scripts.reader import (
    read_papyrus,
    read_protein_set,
    read_molecular_descriptors,
    read_protein_descriptors,
)
from papyrus_scripts.preprocess import (
    keep_accession,
    keep_quality,
    keep_match,
    keep_type,
    keep_organism,
    consume_chunks
)
# from smiles_standardizer import check_std_smiles
from chemutils import standardize_df, ECFP_from_smiles, generate_ecfp, generate_mol_descriptors

import torch
from torch.utils.data import Dataset

# get todays date as yyyy/mm/dd format
from datetime import date
today = date.today()
today = today.strftime("%Y%m%d")


class Papyrus:
    def __init__(
            self,
            path: str = "data/",
            chunksize: int = 1000000,
            accession: Union[None, str, List] = None,
            activity_type: Union[None, str, List] = None,
            protein_class: Union[None, str, List] = None,
            std_smiles: bool = True,
            verbose_files: bool = False,
    ):

        # LOGGING
        log_name = "Papyrus"
        self.log = logging.getLogger(log_name)
        self.log.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s:%(levelname)s:%(name)s:%(message)s:%(relativeCreated)d",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        out_log = os.path.join("logs/", f"{log_name}.log")
        file_handler = logging.FileHandler(out_log, mode="w")
        file_handler.setFormatter(formatter)
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        self.log.addHandler(file_handler)
        # self.log.addHandler(stream_handler)

        self.log.info("Initializing -- PAPYRUS -- Module")

        self.papyrus_path = path
        self.papyrus_file = os.path.isfile(path)

        self.chunksize = chunksize

        self.keep_accession = accession
        activity_type = activity_type.lower()
        if type(activity_type)==str and activity_type in ["xc50", "kx"]:
            act_dict = {
                "xc50": ["IC50", "EC50"],
                "kx": ["Ki", "Kd"]}

            self.activity_key = activity_type
            activity_type = act_dict[activity_type]
            # self.activity_type = activity_type
        elif type(activity_type)==str and activity_type in ["ic50", "ec50", "kd", "ki"]:
            self.activity_key = activity_type

        elif type(activity_type)==list:
            self.activity_key = activity_type.join("_")

        self.keep_type = activity_type
        self.keep_protein_class = protein_class
        self.std_smiles = std_smiles
        self.verbose_files = verbose_files

        self.cols_rename_map = {
            "Activity_ID": "activityIds",
            "Quality": "quality",
            "source": "source",
            "CID": "cIds",
            "SMILES": "smiles",
            "connectivity": "connectivity",
            "InChIKey": "inchiKey",
            "InChI": "inchi",
            "InChI_AuxInfo": "inchiAuxInfo",
            "target_id": "targetIds",
            "TID": "tIds",
            "accession": "accession",
            "Protein_Type": "proteinType",
            "AID": "aIds",
            "doc_id": "docIds",
            "Year": "year",
            "all_doc_ids": "allDocIds",
            "all_years": "allYears",
            "type_IC50": "typeIC50",
            "type_EC50": "typeEC50",
            "type_KD": "typeKd",
            "type_Ki": "typeKi",
            "type_other": "typeOther",
            "Activity_class": "activityClass",
            "relation": "relation",
            "pchembl_value": "pchemblValue",
            "pchembl_value_Mean": "pchemblValueMean",
            "pchembl_value_StdDev": "pchemblValueStdDev",
            "pchembl_value_SEM": "pchemblValueSEM",
            "pchembl_value_N": "pchemblValueN",
            "pchembl_value_Median": "pchemblValueMedian",
            "pchembl_value_MAD": "pchemblValueMAD",
        }

        dtypes = {
            "typeIC50": 'str',  # 'int32',
            "typeEC50": 'str',  # 'int32',
            "typeKi": 'str',  # 'int32',
            "typeKd": 'str',  # 'int32',
            "relation": 'str',
            "activityClass": 'str',
            "pchemblValue": 'str',
            "pchemblValueMean": 'float64',
            "pchemblValueStdDev": 'float64',
            "pchemblValueSEM": 'float64',
            "pchemblValueN": 'int32',
            "pchemblValueMedian": 'float64',
            "pchemblValueMAD": 'float64',
        }

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
            self.papyrus_path = "data/"
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
            self.df_filtered = consume_chunks(filter_3, progress=True, total=int(60000000 / self.chunksize))

            # IMPORTANT - WE HERE HAVE TO SET THE STD_SMILES TO TRUE
            self.std_smiles = True

            # self.df_filtered = consume_chunks(filter_2, progress=True, total=int(60000000 / self.chunksize)) #
            # filter_3 = keep_organism(
            #     data=filter_2, protein_data=self.papyrus_protein_data, organism='Homo sapiens (Human)'
            # )
            # self.df_filtered = consume_chunks(filter_3, progress=True, total=int(60000000 / self.chunksize))

            # if self.keep_type:
            #     filter_4 = keep_type(
            #         data=filter_3, activity_types=self.keep_type
            #     )
            # else:
            #     filter_4 = filter_3

            # filter_4 = keep_type(
            #     data=filter_3, activity_types=self.keep_type
            # )

            # if self.keep_accession:
            #     filter_5 = keep_accession(
            #         data=filter_4, accession=self.keep_accession
            #     )
            # else:
            #     filter_5 = filter_4

            # self.df_filtered = consume_chunks(filter_5, progress=True, total=int(60000000/self.chunksize))

            # if self.verbose_files:
            #     # Renaming before any processing for consistency
            #     # self.df_filtered.rename(columns=self.cols_rename_map, inplace=True)
            #     self.df_filtered.to_csv(f"data/{today}_papyrus_filtered_high_quality_00_preprocessed.csv")

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
                self.df_filtered.to_csv(f"data/{today}_papyrus_{self.activity_key}_01_standardized.csv")
                df_nan.to_csv(f"data/{today}_papyrus_{self.activity_key}_02_NaN_smiles.csv")
                df_dup.to_csv(f"data/{today}_papyrus_{self.activity_key}_03_dup_smiles.csv")

        # # converting pchembl values into list of floats - not needed anymore
        # self.df_filtered["pchemblValue"] = (
        #     self.df_filtered["pchemblValue"]
        #     .str.split(";")
        #     .apply(lambda x: [float(i) for i in x] if type(x) != float else x)
        # )

        # # calculate ECFP fingerprints
        # self.df_filtered = generate_ecfp(self.df_filtered, 2, 1024, False, False)
        # >>>>>>> TO THE DATALOADER
        # # calculate mol descriptors
        # self.df_filtered = generate_mol_descriptors(self.df_filtered, 'smiles', None)
        # TODO Adding Molecular and Protein Descriptors ??? NO NEED will be added in the dataloader
        # mol_descriptors = self.molecular_descriptors()
        # protein_descriptors = self.protein_descriptors()

        return self.df_filtered  # , mol_descriptors, protein_descriptors

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
            mol_descriptors.to_csv(f"data/{today}_papyrus_{self.activity_key}_04_mol_desc.csv")

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
            protein_descriptors.to_csv(f"data/{today}_papyrus_{self.activity_key}_05_protein_desc.csv")

        self.log.info("Protein descriptors shape: {}".format(protein_descriptors.shape))

        return protein_descriptors


def data_preparation(
        papyrus_path: str = "data/",
        activity="xc50",
        n_top=20,
        multitask=True,
        std_smiles=True,
        output_path="data/dataset/",
        verbose_files=False,
):
    assert activity.lower() in ["xc50", "kx"], "activity must be either xc50 or kx"
    assert isinstance(n_top, int), "n_top must be an integer"
    assert isinstance(std_smiles, bool), "std_smiles must be a boolean"
    # assert isinstance(ecfp_length, int), "ecfp_length must be an integer"
    if not os.path.exists(output_path):
        # make dir
        os.makedirs(output_path)


    # Read the data
    papyrus_ = Papyrus(
        path=papyrus_path,
        chunksize=1000000,
        accession=None,
        activity_type=activity,
        protein_class=None,
        std_smiles=std_smiles,
        verbose_files=verbose_files,
    )
    df = papyrus_()
    # step 1: group the dataframe by protein target
    # print(df_xc50.shape)
    grouped = df.groupby('accession')
    # step 2: count the number of measurements for each protein target
    counts = grouped['accession'].count()
    # step 3: sort the counts in descending order
    sorted_counts = counts.sort_values(ascending=False)  # by='counts',
    # step 4: select the 20 protein targets with the highest counts
    top_targets = sorted_counts.head(n_top)
    # step 5: filter the original dataframe to only include rows corresponding to these 20 protein targets
    filtered_df = df[df['accession'].isin(top_targets.index)]
    # step 6: filter the dataframe to only include rows with a pchembl value mean
    filtered_df = filtered_df[filtered_df['pchemblValueMean'].notna()]

    # step 7: multitask pivoting
    if multitask:
        pivoted = pd.pivot_table(
            filtered_df,
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
        df = filtered_df[["smiles", "accession", "pchemblValueMean"]]
        target_col = ['pchemblValueMean']
    # step 8: generate the ecfp descriptors
    df = generate_ecfp(df, 2, 1024, False, False)
    df = generate_ecfp(df, 4, 2048, False, False)

    # step 9: split and save the dataframes
    train_path = os.path.join(output_path, "train.pkl")
    val_path = os.path.join(output_path, "val.pkl")
    test_path = os.path.join(output_path, "test.pkl")
    all_path = os.path.join(output_path, "all.pkl")
    target_col_path = os.path.join(output_path, "target_col.pkl")

    train_data, test_data = train_test_split(
        df, test_size=0.3, shuffle=True, random_state=42
    )
    val_data, test_data = train_test_split(
        test_data, test_size=0.5, shuffle=True, random_state=42
    )

    for file_path, data in zip(
            [train_path, val_path, test_path, all_path, target_col_path],
            [train_data, val_data, test_data, df, target_col]):
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    return train_data, val_data, test_data, df


class PapyrusDataset(Dataset):
    def __init__(
            self,
            file_path: str = "data/dataset/train.pkl",
            input_col: str = "ecfp1024",
            target_col: Union[str, List, None] = None,
            device="cuda",
            # smiles_col: str = "smiles",
            # length: int = 1024,
            # radius: int = 0,
    ):
        folder_path = os.path.dirname(file_path)
        with open(file_path, 'rb') as file:
            self.data = pickle.load(file)

        # self.data = self.data.to(device)

        self.input_col = input_col.lower()
        # self.smiles_col = smiles_col.lower()
        if target_col is None:
            # Assuming your DataFrame is named 'df'
            target_col_path = os.path.join(folder_path, "target_col.pkl")
            with open(target_col_path, 'rb') as file:
                target_col = pickle.load(file)

        self.target_col = target_col
        # self.length = length
        # if radius == 0:
        #     radius_dict = {1024: 2, 2048: 4, 4096: 4}
        #     self.radius = radius_dict[length]
        # else:
        #     self.radius = radius
        # self.x_data = self.df[input_col]
        # self.y_data = self.df[target_col]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        # x_smiles = row[self.smiles_col]
        x_sample = torch.tensor(row[self.input_col]).to(torch.float)
        y_sample = torch.tensor(row[self.target_col]).to(torch.float)
        return x_sample, y_sample

        # return (x_smiles, x_sample), y_sample
        # x_smiles = self.data.iloc[idx][self.input_col]
        # x_sample = ECFP_from_smiles(x_smiles, self.radius, self.length) # TODO : calculate ECFP for all before - not in the dataloader
        # TODO you can also have cuda-tensor on gpu you can index from there - avoiding moving it from RAM -> GPU
        # TODO dump them as pickle file with the ECFP calculated
        # x_sample = np.array(x_sample, dtype=bool).astype(np.float32)


        # y_sample = self.data.iloc[idx][self.target_col]

          # .unsqueeze(1)
        # x_sample = torch.tensor(self.x_data[idx])
        # y_sample = torch.tensor(self.y_data[idx]).to(torch.float)



def build_top_dataset(
        data_path="data/", #"data/" , "data/papyrus_filtered_high_quality_01_standardized.csv",
        activity="xc50",
        n_top=20,
        descriptors="ecfp",
        multitask=True,
        std_smiles=True,
        ecfp_length=1024,
        ecfp_radius=None,
        export=True,
):
    assert activity.lower() in ["xc50", "kx"], "activity must be either xc50 or kx"
    assert descriptors.lower() in ["ecfp", "mol", "none", None], "descriptors must be either ecfp or mol or none"
    assert isinstance(n_top, int), "n_top must be an integer"
    assert isinstance(std_smiles, bool), "std_smiles must be a boolean"
    assert isinstance(ecfp_length, int), "ecfp_length must be an integer"
    assert isinstance(ecfp_radius, int) or ecfp_radius == None, "ecfp_radius must be an integer or None"
    assert isinstance(export, bool), "export must be a boolean"

    act_dict = {
        "xc50": ["IC50", "EC50"],
        "kx": ["Ki", "Kd"]}
    activity = activity.lower()

    papyrus_ = Papyrus(
        path=data_path,
        chunksize=1000000,
        accession=None,
        activity_type=act_dict[activity],
        protein_class=None,
        std_smiles=std_smiles,
        verbose_files=export,
    )
    df = papyrus_()

    # if os.path.isfile(data_path):
    #     dtypes = {
    #         "typeIC50": 'str',  # 'int32',
    #         "typeEC50": 'str',  # 'int32',
    #         "typeKi": 'str',  # 'int32',
    #         "typeKd": 'str',  # 'int32',
    #         "relation": 'str',
    #         "activityClass": 'str',
    #         "pchemblValue": 'str',
    #         "pchemblValueMean": 'float64',
    #         "pchemblValueStdDev": 'float64',
    #         "pchemblValueSEM": 'float64',
    #         "pchemblValueN": 'int32',
    #         "pchemblValueMedian": 'float64',
    #         "pchemblValueMAD": 'float64',
    #     }
    #
    #     df = pd.read_csv(data_path, dtype=dtypes, index_col=0, header=0, low_memory=False)
    #
    # else:
    #     papyrus_ = Papyrus(
    #         path=data_path,
    #         chunksize=1000000,
    #         accession=None,
    #         activity_type=act_dict[activity],
    #         protein_class=None,
    #         verbose_files=True
    #     )
    #     df = papyrus_()
    #     df.to_csv(data_path+f"{date}_{activity}.csv")
    # Filtering the top x targets in number of datapoints.
    # step 1: group the dataframe by protein target
    # print(df_xc50.shape)
    grouped = df.groupby('accession')
    # step 2: count the number of measurements for each protein target
    counts = grouped['accession'].count()
    # step 3: sort the counts in descending order
    sorted_counts = counts.sort_values(ascending=False) # by='counts',
    # step 4: select the 20 protein targets with the highest counts
    top_targets = sorted_counts.head(n_top)
    # step 5: filter the original dataframe to only include rows corresponding to these 20 protein targets
    filtered_df = df[df['accession'].isin(top_targets.index)]

    if multitask:
        # pivot the dataframe
        pivoted = pd.pivot_table(
            filtered_df,
            index='smiles',
            columns='accession',
            values='pchemblValueMean',
            aggfunc='first'
        )
        # reset the index to make the 'smiles' column a regular column
        pivoted = pivoted.reset_index()
        # replace any missing values with NaN
        df = pivoted.fillna(value=np.nan)

    else:
        df = filtered_df[["smiles", "accession", "pchemblValueMean"]]

    if ecfp_radius == 0 or ecfp_radius == None:
        radius_dict = {1024: 2, 2048: 4, 4096: 4}
        ecfp_radius = radius_dict[ecfp_length]
    else:
        pass

    if descriptors.lower() == "ecfp":
        # calculate ECFP fingerprints
        # df = generate_ecfp(df, 2, 1024, False, False)
        df = generate_ecfp(df, ecfp_radius, ecfp_length, False, False)

    elif descriptors.lower() == "mol":
        # calculate mol descriptors
        df = generate_mol_descriptors(df, 'smiles', None)
    else:
        pass


    # ## calculate properties
    # # calculate ECFP fingerprints
    # df = generate_ecfp(df, 2, 1024, False, False)
    # # calculate mol descriptors
    # df = generate_mol_descriptors(df, 'smiles', None)
    if export:
        df.to_csv(data_path+f"{date}_{activity}_top_{n_top}_multitask_{multitask}_descriptors_{descriptors}.csv")

    return df





