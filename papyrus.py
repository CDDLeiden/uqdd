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
import logging

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
from chemutils import standardize_df, generate_ecfp, generate_mol_descriptors

import torch
from torch.utils.data import Dataset

class Papyrus:
    def __init__(
            self,
            path: str = "data/",
            chunksize: int = 1000000,
            accession: Union[None, str, List] = None,
            activity_type: Union[None, str, List] = None,
            protein_class: Union[None, str, List] = None,
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
        self.keep_type = activity_type
        self.keep_protein_class = protein_class
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

        if self.papyrus_file:
            self.log.info("PapyrusApi processing input from previously processed file")
            self.df_filtered = pd.read_csv(self.papyrus_path, index_col=0)
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

            # self.df_filtered = consume_chunks(filter_2, progress=True, total=int(60000000 / self.chunksize)) #

            filter_3 = keep_organism(
                data=filter_2, protein_data=self.papyrus_protein_data, organism='Homo sapiens (Human)'
            )
            # self.df_filtered = consume_chunks(filter_3, progress=True, total=int(60000000 / self.chunksize))

            # if self.keep_type:
            #     filter_4 = keep_type(
            #         data=filter_3, activity_types=self.keep_type
            #     )
            # else:
            #     filter_4 = filter_3

            filter_4 = keep_type(
                data=filter_3, activity_types=self.keep_type
            )
            self.df_filtered = consume_chunks(filter_4, progress=True, total=int(60000000 / self.chunksize))
            # if self.keep_accession:
            #     filter_5 = keep_accession(
            #         data=filter_4, accession=self.keep_accession
            #     )
            # else:
            #     filter_5 = filter_4



            # self.df_filtered = consume_chunks(filter_5, progress=True, total=int(60000000/self.chunksize))

            # Renaming before any processing for consistency
            self.df_filtered.rename(columns=self.cols_rename_map, inplace=True)

            if self.verbose_files:
                self.df_filtered.to_csv("data/papyrus_filtered_high_quality_00_preprocessed.csv")

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
            self.df_filtered.to_csv("data/papyrus_filtered_high_quality_01_standardized.csv")
            df_nan.to_csv("data/papyrus_filtered_high_quality_02_NaN_smiles.csv")
            df_dup.to_csv("data/papyrus_filtered_high_quality_03_dup_smiles.csv")

        # converting pchembl values into list of floats
        self.df_filtered["pchemblValue"] = (
            self.df_filtered["pchemblValue"]
            .str.split(";")
            .apply(lambda x: [float(i) for i in x] if type(x) != float else x
                   ))

        # # calculate ECFP fingerprints
        # self.df_filtered = generate_ecfp(self.df_filtered, 2, 1024, False, False)
        #
        # # calculate mol descriptors
        # self.df_filtered = generate_mol_descriptors(self.df_filtered, 'smiles', None)
        # TODO Adding Molecular and Protein Descriptors ????
        # mol_descriptors = self.molecular_descriptors()
        # protein_descriptors = self.protein_descriptors()

        return self.df_filtered #, mol_descriptors, protein_descriptors

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
            mol_descriptors.to_csv("data/papyrus_filtered_high_quality_04_mol_desc.csv")

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
            protein_descriptors.to_csv("data/papyrus_filtered_high_quality_05_protein_desc.csv")

        self.log.info("Protein descriptors shape: {}".format(protein_descriptors.shape))

        return protein_descriptors

from chemutils import ECFP_from_smiles
class PapyrusDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            input_col: str = "smiles",
            target_col: str = "pchemblValueMean",
    ):
        self.data = data
        self.input_col = input_col
        self.target_col = target_col
        # self.x_data = self.df[input_col]
        # self.y_data = self.df[target_col]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(f"{idx=}")
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x_smiles = self.data.iloc[idx][self.input_col]
        x_sample = ECFP_from_smiles(x_smiles, 2, 1024)
        x_sample = torch.tensor(x_sample).to(torch.float)

        y_sample = self.data.iloc[idx][self.target_col]
        y_sample = torch.tensor(y_sample).to(torch.float) #.unsqueeze(1)
        # x_sample = torch.tensor(self.x_data[idx])
        # y_sample = torch.tensor(self.y_data[idx]).to(torch.float)

        return x_sample, y_sample

# # Example usage
# data = pd.read_csv('data/papyrus_filtered_high_quality_xc50_00_preprocessed.csv')
# papyrus_dataset = PapyrusDataset(data)
# papyrus_dataloader = DataLoader(papyrus_dataset, batch_size=32, shuffle=True)