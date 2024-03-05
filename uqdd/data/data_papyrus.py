import logging
import argparse
from pathlib import Path
from typing import Union, List
import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from papyrus_scripts.download import download_papyrus
from papyrus_scripts.preprocess import (
    keep_quality,
    keep_match,
    keep_type,
    keep_organism,
    consume_chunks,
)
from papyrus_scripts.reader import read_papyrus, read_protein_set
from uqdd import DATA_DIR, DATASET_DIR
from uqdd.utils import create_logger, get_config
from uqdd.utils_chem import standardize_df, get_chem_desc
from uqdd.utils_prot import get_embeddings
from uqdd.data.utils_data import (
    split_data,
    export_df,
    load_df,
    load_pickle,
    check_if_processed_file,
    export_tasks,
)


class Papyrus:
    def __init__(
        self,
        accession: Union[None, str, List] = None,
        activity_type: Union[None, str, List] = None,
        std_smiles: bool = True,
        verbose_files: bool = False,
    ):
        # LOGGING
        self.log = create_logger(
            name="Papyrus", file_level="debug", stream_level="info"
        )
        self.config = get_config("papyrus")
        self.chunksize = self.config.get("chunksize", 1000000)
        self.cols_to_keep = self.config.get("cols_to_keep", None)
        self.dtypes, self.dtypes_protein = self.config.get(
            "dtypes", {}
        ), self.config.get("dtypes_protein", {})
        self.keep_accession = accession
        self.activity_key, self.activity_type = self._parse_activity_key(activity_type)
        (
            self.output_path,
            self.pap_filepath,
            self.pap_path,
            self.pap_file,
            self.pap_protpath,
        ) = self._setup_path(self.activity_key)
        self.std_smiles, self.verbose_files = std_smiles, verbose_files
        self.df_filtered, self.papyrus_protein_data = self._load_or_process_papyrus()

        self.desc_chems = [
            "ecfp1024",
            "ecfp2048",
            "mold2",
            "cddd",
            "fingerprint",
            "moldesc",
            # "mordred",
            # "moe",
            # "graph2d",
        ]
        self.desc_prots = [
            "esm1_t12",  # TODO
            None,
            "unirep",
            "esm1_t34",
            "esm1_t6",
            "esm1b",
            "esm_msa1",
            "esm_msa1b",
            "esm1v",
            "protbert",
            "protbert_bfd",
            "ankh-base",
            "ankh-large",
        ]

    def __call__(
        self,
        n_targets: int = -1,
        desc_prot: Union[str, None] = None,
        desc_chem: Union[str, None] = None,
        recalculate: bool = False,
        split_type: str = "random",
        split_proportions=None,
        first_run_all_splits: bool = False,
        first_run_all_descs: bool = False,
        file_ext: str = "pkl",
        batch_size: int = 2,
    ):
        # step 0: assertions
        assert (
            desc_prot or desc_chem or first_run_all_descs
        ), "Either a descriptor must be provided or calculate all of them by setting first_run_all_descs=True"
        topx = f"top{n_targets}" if n_targets > 0 else "all"
        output_folder = self.output_path / topx

        # step 1: get the data top-x or all
        df, label_col = self._get_n_targets_or_all_dataset(n_targets=n_targets)

        # if (not first_run_all_descs and not recalculate) or split_type != "all":
        # TODO fixing this with first_run_all_descs and split all
        files_exist, files_paths = check_if_processed_file(
            data_name="papyrus",
            activity_type=self.activity_key,
            n_targets=n_targets,
            split_type=split_type,
            desc_prot=desc_prot,
            desc_chem=desc_chem,
            file_ext=file_ext,
        )
        # search for the file otherwise it will be calculated
        if not recalculate and files_exist:
            self.log.warning(
                "Found the processed files, if you want to recalculate set recalculate=True"
            )
            return self._load_processed_files(files_paths, split_type)

        if first_run_all_descs:
            self.desc_prots = self.desc_prots if n_targets <= 0 else [None]

            df, desc_prots, desc_chems = self.merge_descriptors(
                df=df,
                desc_prot=self.desc_prots,
                desc_chem=self.desc_chems,
                batch_size=batch_size,
            )
            desc_iter = list(
                itertools.product(desc_prots, desc_chems)
            )  # Only including the ones that were actually calculated

        else:
            # step 2: merge the descriptors
            df, desc_prots, desc_chems = self.merge_descriptors(
                df=df,
                desc_prot=[desc_prot],
                desc_chem=[desc_chem],
                batch_size=batch_size,
            )
            desc_iter = [(desc_prots, desc_chems)]

        # step 3: split the data
        splitted_data_dict = self._split(
            df, split_type, split_proportions, first_run_all_splits
        )

        for desc_prot, desc_chem in desc_iter:
            # step 4: filter the columns
            cols_to_include = self._cols_to_include(
                desc_chem, desc_prot, n_targets, label_col
            )
            # step 5: export the data
            self._export_datasets(
                splitted_data_dict,
                output_folder,
                desc_prot,
                desc_chem,
                cols_to_include=cols_to_include,
            )

        return splitted_data_dict

    # def merge_all_descs(self, df):
    #
    #     for desc_chem in self.desc_chems:
    #         try:
    #             df = self.merge_descriptors(df, desc_prot=None, desc_chem=desc_chem)
    #         except Exception as e:
    #             self.log.error(f"Error: {e} - {desc_chem} not calculated")
    #             self.desc_chems.remove(desc_chem)
    #             continue
    #
    #     for desc_prot in self.desc_prots:
    #         try:
    #             df = self.merge_descriptors(df, desc_prot=desc_prot, desc_chem=None)
    #         except Exception as e:
    #             self.log.error(f"Error: {e} - {desc_prot} not calculated")
    #             self.desc_prots.remove(desc_prot)
    #             continue
    #
    #     return df

    @staticmethod
    def _export_datasets(
        data_dict,
        output_folder,
        desc_prot=None,
        desc_chem=None,
        file_ext="pkl",
        cols_to_include=None,
    ):
        """
        Exports each DataFrame in the data dictionary to the corresponding path in files_paths.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing split_type as key, and another
            dict as value of dataframes with keys
            as 'train', 'val', and 'test'.
        output_folder : Path
            folder path where the files will be saved.
        desc_prot : str or None or List
            Descriptor type for the protein.
        desc_chem : str or None or List
            Descriptor type for the chemical compound.
        file_ext : str (default='pkl')
            File extension to use for the saved files.
        """
        output_folder.mkdir(parents=True, exist_ok=True)

        for split_type, subdict in data_dict.items():
            prefix = (
                f"{split_type}_{desc_prot}_{desc_chem}"
                if desc_prot
                else f"{split_type}_{desc_chem}"
            )
            for subset, df in subdict.items():
                file_path = output_folder / f"{prefix}_{subset}.{file_ext}"
                if cols_to_include:
                    df = df[cols_to_include]
                export_df(df, file_path=file_path)

    @staticmethod
    def _split(df, split_type, split_proportions, first_run_all_splits=False):
        if split_proportions is None:
            split_proportions = [0.7, 0.15, 0.15]

        if first_run_all_splits:
            split_type = "all"

        splitted_data_dict = split_data(
            df,
            split_type=split_type,
            smiles_col="SMILES",
            time_col="Year",
            fractions=split_proportions,
            seed=42,
        )
        return splitted_data_dict

    @staticmethod
    def _cols_to_include(
        desc_chem: list, desc_prot: list, n_targets: int, label_col: list
    ):
        cols_to_include = ["SMILES", *desc_chem]  # * used to unpack the list ['ecfp']
        cols_to_include = (
            cols_to_include + ["target_id", *desc_prot, "Year"]
            if n_targets <= 0 and desc_prot
            else cols_to_include
        )
        cols_to_include += label_col
        if None in cols_to_include:
            cols_to_include.remove(None)

        return cols_to_include

    def _get_n_targets_or_all_dataset(
        self,
        n_targets=20,
    ):
        if n_targets > 0:
            df, top_targets = self._get_top_targets(self.df_filtered, n_targets)

            # if multitask:
            pivoted = pd.pivot_table(
                df,
                index="SMILES",
                columns="accession",
                values="pchembl_value_Mean",
                aggfunc="first",
            )
            # reset the index to make the "smiles" column a regular column
            pivoted = pivoted.reset_index()
            # replace any missing values with NaN
            df = pivoted.fillna(value=np.nan)
            label_col = list(top_targets.index)
            export_tasks(
                data_name="papyrus",
                activity=self.activity_key,
                n_targets=n_targets,
                label_col=label_col,
            )
        else:
            # dataset/papyrus/xc50/all/
            df = self.df_filtered.copy()  # the whole thing
            label_col = ["pchembl_value_Mean"]

        return df, label_col

    def merge_descriptors(
        self,
        df=None,
        desc_prot: Union[str, List[str], None] = None,
        desc_chem: Union[str, List[str], None] = None,
        # all_descs: bool = False,
        batch_size: int = 4,
    ):
        # "unirep", "esm", "protbert", "msa", "all"
        # "mold2", "mordred", "cddd", "fingerprint", "moe", "all"
        # check descriptors saved file:
        # if all_descs:
        #     desc_prot = self.desc_prots
        #     desc_chem = self.desc_chems
        # else:  # make sure it is in a list format
        #     desc_prot = [desc_prot] if desc_prot else []
        #     desc_chem = [desc_chem] if desc_chem else []

        # desc_prot = [desc_prot] if desc_prot and isinstance(desc_prot, str) else []
        # desc_chem = [desc_chem] if desc_chem and isinstance(desc_chem, str) else []

        if df is None:
            df = self.df_filtered.copy()

        if desc_prot:  # This is not MT then
            for desc in desc_prot:
                try:
                    self.log.info(f"Merging protein descriptors {desc}")
                    # target_id or sequence ...
                    query_col, df = self._prepare_query_col(df, desc)
                    df = get_embeddings(
                        df,
                        embedding_type=desc,
                        query_col=query_col,
                        batch_size=batch_size,
                    )
                except Exception as e:
                    self.log.error(f"Error: {e} - {desc} not calculated")
                    desc_prot.remove(desc)
                    continue

        if desc_chem:
            for desc in desc_chem:
                try:
                    self.log.info(f"Merging molecular descriptors {desc}")
                    # connectivity or SMILES
                    query_col, _ = self._prepare_query_col(df, desc)
                    df = get_chem_desc(df, desc_type=desc, query_col=query_col)

                except Exception as e:
                    self.log.error(f"Error: {e} - {desc} not calculated")
                    desc_chem.remove(desc)
                    continue

        return df, desc_prot, desc_chem

    def _prepare_query_col(self, df, desc):
        if desc is None:
            return None, df
        elif desc in [
            "esm1_t34",
            "esm1_t12",
            "esm1_t6",
            "esm1b",
            "esm_msa1",
            "esm_msa1b",
            "esm1v",
            "protbert",
            "protbert_bfd",
            "ankh-base",
            "ankh-large",
        ]:  # here we need to merge sequences and set as query_col
            query_col = "Sequence"
            df = self.merge_protein_sequences(df, target_id_col="target_id")

        elif desc == "unirep":
            query_col = "target_id"

        elif desc in ["mold2", "mordred", "cddd", "fingerprint"]:  # , "moe"
            query_col = "connectivity"

        elif desc in ["ecfp1024", "ecfp2048", "moldesc"]:  # , "graph2d"
            query_col = "SMILES"

        else:
            raise ValueError(f"Descriptor {desc} not supported")

        return query_col, df

    def merge_protein_sequences(self, df, target_id_col="target_id"):
        return df.merge(
            self.papyrus_protein_data[["target_id", "Sequence"]],
            left_on=target_id_col,
            right_on="target_id",
            how="left",
        )

    @staticmethod
    def _setup_path(activity_key):
        activity_key = activity_key.lower()
        out_path = DATASET_DIR / "papyrus" / activity_key
        out_path.mkdir(parents=True, exist_ok=True)
        pap_filepath = out_path / "papyrus_filtered.csv"
        pap_file = pap_filepath.is_file()
        pap_path = pap_filepath if pap_file else DATA_DIR
        pap_protpath = DATASET_DIR / "papyrus" / "papyrus_proteins.csv"

        return out_path, pap_filepath, pap_path, pap_file, pap_protpath

    def _load_or_process_papyrus(self):
        if self.pap_file:
            self.log.info("Loading previously processed Papyrus data ...")
            df_filtered, papyrus_protein_data = self._load_papyrus_files()

        else:
            self.log.info("PapyrusApi processing input from Raw Papyrus database")
            df_filtered, papyrus_protein_data = self._preprocess_papyrus()

        return df_filtered, papyrus_protein_data

    def _load_papyrus_files(self):
        df_filtered = pd.read_csv(self.pap_path, dtype=self.dtypes, low_memory=False)
        papyrus_protein_data = pd.read_csv(
            self.pap_protpath,
            dtype=self.dtypes_protein,
            low_memory=False,
        )

        return df_filtered, papyrus_protein_data

    @staticmethod
    def _load_processed_files(files_paths, split_type, **kwargs):
        keys = ["train", "val", "test"]

        return {
            split_type: {key: load_df(f, **kwargs) for key, f in zip(keys, files_paths)}
        }

    def _preprocess_papyrus(self):
        papyrus_data, papyrus_protein_data = self._reader()
        df_filtered = self._filter_raw_file(papyrus_data, papyrus_protein_data)

        df_nan, df_dup = None, None
        # SMILES standardization
        if self.std_smiles:
            df_filtered, df_nan, df_dup = self._sanitize_smiles(df_filtered, self.log)
        # Keeping only the columns we need
        if self.cols_to_keep:
            df_filtered = df_filtered[self.cols_to_keep]

        # Forcing verbose here to avoid reprocessing
        df_filtered.to_csv(self.pap_filepath, index=False)
        papyrus_protein_data.to_csv(self.pap_protpath, index=False)
        # Verbosing other files
        if self.verbose_files:
            self._verbosing_files(df_nan, df_dup)

        return df_filtered, papyrus_protein_data

        # df_nan, df_dup = None, None
        # # SMILES standardization
        # if self.std_smiles:
        #     self.df_filtered, df_nan, df_dup = self._sanitize_smiles()
        #
        # if self.cols_to_keep:
        #     self.df_filtered = self.df_filtered[self.cols_to_keep]
        #
        # # Verbosing files
        # if self.verbose_files:
        #     self._verbosing_files(self.df_filtered, df_nan, df_dup)
        #
        # return self.df_filtered

    @staticmethod
    def _get_top_targets(df, n_targets=20):
        # step 1: group the dataframe by protein target
        grouped = df.groupby("accession")
        # step 2: count the number of measurements for each protein target
        counts = grouped["accession"].count()
        # step 3: sort the counts in descending order
        sorted_counts = counts.sort_values(ascending=False)  # , by="counts"
        # step 4: select the x protein targets with the highest counts
        top_targets = sorted_counts.head(n_targets)
        # step 5: filter the original dataframe to only include rows corresponding to these x protein targets
        filtered_df = df[df["accession"].isin(top_targets.index)]
        # step 6: filter the dataframe to only include rows with a pchembl value mean
        filtered_df = filtered_df[filtered_df["pchembl_value_Mean"].notna()]
        return filtered_df, top_targets

    @staticmethod
    def _parse_activity_key(activity_type):
        activity_type = activity_type.lower()
        if isinstance(activity_type, str) and activity_type in ["xc50", "kx"]:
            act_dict = {"xc50": ["IC50", "EC50"], "kx": ["Ki", "Kd"]}
            activity_key = activity_type
            activity_type = act_dict[activity_type]
        # elif isinstance(activity_type, str) and activity_type in [
        #     "ic50",
        #     "ec50",
        #     "kd",
        #     "ki",
        # ]:
        #     activity_key = activity_type
        #
        # elif isinstance(activity_type, list):
        #     activity_key = "_".join(activity_type)
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
        papyrus_protein_data = read_protein_set(
            source_path=self.pap_path, version="latest"
        )
        return papyrus_data, papyrus_protein_data

    def _filter_raw_file(self, papyrus_data, papyrus_protein_data):
        filter_1 = keep_quality(data=papyrus_data, min_quality="High")
        filter_2 = keep_match(data=filter_1, column="Protein_Type", values="WT")
        filter_3 = keep_type(data=filter_2, activity_types=self.activity_type)
        filter_4 = keep_organism(
            data=filter_3,
            protein_data=papyrus_protein_data,
            organism="Homo sapiens (Human)",  # self.keep_organism
        )
        df_filtered = consume_chunks(filter_4, progress=True, total=60)

        # IMPORTANT - WE HERE HAVE TO SET THE STD_SMILES TO TRUE
        self.std_smiles = True

        return df_filtered

    @staticmethod
    def _sanitize_smiles(df_filtered, logger=None):
        logger.info("Sanitizing the SMILES")
        df_filtered, df_nan, df_dup = standardize_df(
            df=df_filtered,
            smiles_col="SMILES",
            other_dup_col="accession",
            drop=True,
            sorting_col="Year",
            keep="last",  # keeps latest year datapoints if duplicated
            logger=logger,
        )

        return df_filtered, df_nan, df_dup

    def _verbosing_files(self, df_nan, df_dup):
        data_dict = {
            "papyrus_NaN_smiles.csv": df_nan,
            "papyrus_dup_smiles.csv": df_dup,
        }
        for file, data in data_dict.items():
            if data:
                data.to_csv(self.output_path / f"{file}", index=False)


class PapyrusDataset(Dataset):
    def __init__(
        self,
        file_path: Union[str, Path] = None,
        desc_prot: Union[str, None] = None,
        desc_chem: Union[str, None] = None,
        **kwargs,
    ) -> None:

        self.data = load_df(file_path, **kwargs)
        dir_path = file_path.parent

        labels_filepath = dir_path / "target_col.pkl"
        self.MT = labels_filepath.is_file()
        self.label_col = (
            load_pickle(labels_filepath) if self.MT else "pchembl_value_Mean"
        )
        self.desc_prot = desc_prot
        self.desc_chem = desc_chem

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data.iloc[idx]
        if self.MT:
            # For multitask learning, use only chemical descriptors
            chem_desc = torch.tensor(sample[self.desc_chem], dtype=torch.float32)
            label = torch.tensor(sample[self.label_col], dtype=torch.float32)
            return chem_desc, label
        else:
            # For single-task learning, use both protein and chemical descriptors
            prot_desc = torch.tensor(sample[self.desc_prot], dtype=torch.float32)
            chem_desc = torch.tensor(sample[self.desc_chem], dtype=torch.float32)
            label = torch.tensor(sample[self.label_col], dtype=torch.float32)
            return prot_desc, chem_desc, label


def get_datasets(
    n_targets: int = -1,
    activity_type: str = "xc50",
    split_type: str = "random",
    desc_prot: Union[str, None] = None,
    desc_chem: Union[str, None] = None,
    ext: str = "pkl",
    logger: Union[None, logging.Logger] = None,
):
    activity_type = activity_type.lower()
    desc_chem = desc_chem.lower()
    desc_prot = desc_prot.lower() if desc_prot else None
    if logger is None:
        logger = create_logger(
            name="Papyrus_get_datasets", file_level="debug", stream_level="info"
        )

    dir_path = DATASET_DIR / "papyrus" / activity_type
    dir_path = dir_path / "all" if n_targets <= 0 else dir_path / f"top{n_targets}"

    if n_targets > 0:
        logger.info(
            "Initializing dataset for multitask learning; only chemical descriptors will be used."
        )
        filename_prefix = f"{split_type}_{desc_chem}"
    else:
        if desc_prot:
            logger.info(
                "Initializing pcm dataset for single-task learning; both protein and chemical descriptors will be used."
            )
            filename_prefix = f"{split_type}_{desc_prot}_{desc_chem}"
        else:
            logger.info(
                "Initializing dataset for single-task learning; only chemical descriptors will be used."
            )
            filename_prefix = f"{split_type}_{desc_chem}"

    datasets = {}
    for subset in ["train", "val", "test"]:
        file_path = dir_path / f"{filename_prefix}_{subset}.{ext}"
        if not file_path.is_file():
            # TODO calculate them here if not found
            raise FileNotFoundError(
                f"File {subset} not found: {file_path} - you need to precalculate the dataset "
                f"features and splits first."
            )
        datasets[subset] = PapyrusDataset(
            file_path, desc_prot=desc_prot, desc_chem=desc_chem
        )

    return datasets
    # try:
    #     paths = {
    #         split: os.path.join(DATASET_DIR, activity, split_type, f"{split}.pkl")
    #         for split in ["train", "val", "test"]
    #     }
    #     datasets = {}
    #
    #     for input_col in ["ecfp1024", "ecfp2048"]:
    #         for split, dataset_path in paths.items():
    #             key = f"{split}_{input_col}"
    #             datasets[key] = PapyrusDatasetMT(
    #                 dataset_path, input_col=input_col, device=device
    #             )
    #     return datasets

    # except Exception as e:
    #     raise RuntimeError(f"Error loading datasets: {e}")


# class _PapyrusDatasetMT(Dataset):
#     def __init__(
#         self,
#         file_path: Union[str, Path] = DATASET_DIR
#         / "xc50"
#         / "all"
#         / "random"
#         / "train.pkl",
#         input_col: str = "ecfp1024",
#         target_col: Union[str, List, None] = None,
#         device: str = "cuda",
#     ) -> None:
#         """
#         Parameters
#         ----------
#         file_path: str or Path
#         input_col: str
#         target_col: str or List or None
#         device: str
#         """
#         folder_path = file_path.parent
#         with open(file_path, "rb") as file:
#             self.data = pickle.load(file)
#
#         self.input_col = input_col.lower()
#
#         if target_col is None:  # has to be changed if all and not MT
#             # Assuming your DataFrame is named "df"
#             target_col_path = folder_path / "target_col.pkl"
#             with open(target_col_path, "rb") as file:
#                 target_col = pickle.load(file)
#
#         self.target_col = target_col
#
#         self.input_data = (
#             torch.from_numpy(np.stack(self.data[self.input_col].values))
#             .to(torch.float)
#             .to(device)
#         )
#         self.target_data = (
#             torch.tensor(self.data[self.target_col].values).to(torch.float).to(device)
#         )
#
#         del self.data
#
#     def __len__(self):
#         return len(self.input_data)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         x_sample = self.input_data[idx]
#         y_sample = self.target_data[idx]
#
#         return x_sample, y_sample
#

# class _PapyrusDataset(Dataset):
#     def __init__(
#         self,
#         file_path: Union[str, Path] = DATASET_DIR
#         / "xc50"
#         / "all"
#         / "random"
#         / "train.pkl",
#         chem_xcol: str = "ecfp1024",
#         prot_xcol: str = "protein_embeddings",
#         label_ycol: Union[str, List, None] = "pchembl_value_Mean",
#         # input_col: str = "ecfp1024",
#         device: object = "cuda",
#     ) -> None:
#         folder_path = file_path.parent
#         with open(file_path, "rb") as file:
#             self.data = pickle.load(file)
#
#         self.chem_xcol = chem_xcol.lower()
#         self.prot_xcol = prot_xcol.lower()
#         # TODO check if label_ycol is None
#         # if label_ycol is None: # MT
#
#         self.label_ycol = label_ycol
#
#         if label_ycol is None:  # has to be changed if all and not MT
#             # Assuming your DataFrame is named "df"
#             target_col_path = folder_path / "target_col.pkl"
#             with open(target_col_path, "rb") as file:
#                 target_col = pickle.load(file)
#
#         self.target_col = target_col
#
#         self.input_data = (
#             torch.from_numpy(np.stack(self.data[self.chem_xcol].values))
#             .to(torch.float)
#             .to(device)
#         )
#         self.target_data = (
#             torch.tensor(self.data[self.target_col].values).to(torch.float).to(device)
#         )
#
#         del self.data
#
#     def __len__(self):
#         return len(self.input_data)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         x_sample = self.input_data[idx]
#         y_sample = self.target_data[idx]
#
#         return x_sample, y_sample
#

# def run_all_combs(args):
#     # Define all possible values for each argument
#     activity_types = ["xc50", "kx"]
#     n_targets = args.n_targets
#     desc_prots = (
#         [
#             None,
#             "unirep",
#             "esm1_t34",
#             "esm1_t12",
#             "esm1_t6",
#             "esm1b",
#             "esm_msa1",
#             "esm_msa1b",
#             "esm1v",
#             "protbert",
#             "protbert_bfd",
#             "ankh-base",
#             "ankh-large",
#         ]
#         if n_targets <= 0
#         else [None]
#     )
#
#     desc_chems = [
#         "ecfp1024",
#         "ecfp2048",
#         "mold2",
#         "cddd",
#         "fingerprint",
#         "moldesc",
#         # "mordred",
#         # "moe",
#         # "graph2d",
#     ]
#     i = 0
#     for act_key in activity_types:
#         papyrus = Papyrus(
#             activity_type=act_key,
#             std_smiles=True,
#             verbose_files=False,
#         )
#         call_args = list(itertools.product(desc_prots, desc_chems))
#         for call_arg in call_args:
#             # Call the instance with the current combination of call argument values
#             papyrus.log.info(f"{i} Running Papyrus with the following arguments:")
#             papyrus.log.info(
#                 f"act_key: {act_key}, n_top: {args.n_targets}, call_arg: {call_arg}"
#             )
#             i += 1
#             try:
#                 papyrus(
#                     n_targets=args.n_targets,
#                     desc_prot=call_arg[0],
#                     desc_chem=call_arg[1],
#                     recalculate=args.recalculate,
#                     first_run_all_splits=True,
#                     file_ext=args.file_ext,  # "pkl",
#                     batch_size=args.batch_size,  # 2,
#                 )
#             except Exception as e:
#                 papyrus.log.error(f"Error: {e}")
#                 papyrus.log.error(
#                     f"act_key: {act_key}, n_top: {args.n_targets}, call_arg: {call_arg}"
#                 )
#                 continue
#


def run_papyrus(args):
    # Create an instance of the Papyrus class with the arguments provided by the user
    papyrus = Papyrus(
        accession=args.accession,
        activity_type=args.activity_type,
        std_smiles=args.std_smiles,
        verbose_files=args.verbose_files,
    )

    # Call the instance with the arguments provided by the user
    split_proportions = (
        [float(item) for item in args.split_proportions.split(",")]
        if args.split_proportions
        else None
    )
    papyrus(
        n_targets=args.n_targets,
        desc_prot=args.desc_prot,
        desc_chem=args.desc_chem,
        recalculate=args.recalculate,
        split_type=args.split_type,
        split_proportions=split_proportions,
        first_run_all_splits=args.first_run_all_splits,
        first_run_all_descs=args.all_combinations,
        file_ext=args.file_ext,
        batch_size=args.batch_size,
    )


def main():
    parser = argparse.ArgumentParser(description="Papyrus class interface")

    # Argument to specify whether to calculate all combinations of arguments
    parser.add_argument(
        "--all_combinations",
        action="store_true",
        help="Calculate all combinations of arguments",
    )

    # Arguments for Papyrus class initialization
    parser.add_argument("--accession", type=str, default=None, help="Accession")
    parser.add_argument("--activity_type", type=str, default=None, help="Activity type")
    parser.add_argument("--std_smiles", action="store_true", help="Standardize SMILES")

    # Arguments for Papyrus class call
    parser.add_argument(
        "--desc_prot", type=str, default=None, help="Protein descriptor"
    )
    parser.add_argument(
        "--desc_chem", type=str, default=None, help="Chemical descriptor"
    )
    parser.add_argument("--split_type", type=str, default="random", help="Split type")

    parser.add_argument("--verbose_files", action="store_true", help="Verbose files")

    parser.add_argument(
        "--first_run_all_splits", action="store_true", help="First run all splits"
    )

    # optional arguments even with all combs
    parser.add_argument(
        "--n-targets", type=int, default=-1, help="Number of top targets"
    )

    parser.add_argument("--recalculate", action="store_true", help="Recalculate")
    parser.add_argument(
        "--split_proportions",
        type=str,
        default=None,
        help="Split proportions 0.7,0.15,0.15",
    )
    parser.add_argument("--file_ext", type=str, default="pkl", help="File extension")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")

    args = parser.parse_args()

    run_papyrus(args)

    # if args.all_combinations:
    #     run_all_combs(args)
    #
    # else:
    #     run_papyrus(args)


#
# parser.add_argument(
#     "--std_smiles", type=bool, default=True, help="Standardize SMILES"
# )
# parser.add_argument(
#     "--verbose_files", type=bool, default=False, help="Verbose files"
# )
# parser.add_argument("--recalculate", type=bool, default=False, help="Recalculate")
# parser.add_argument(
#     "--first_run_all_splits", type=bool, default=False, help="First run all splits"
# )
#
if __name__ == "__main__":
    # main()
    # init args
    activity_type = "xc50"
    std_smiles = True
    verbose_files = False

    # call args
    n_targets = -1
    desc_prot = None  # "esm1_t12"  # "ankh-base"
    # "esm1b"
    # ValueError: You cant't pass sequence with length
    # more than 1024 with esm1b_t33_650M_UR50S, use esm1_t34_670M_UR100 or filter the sequence length

    desc_chem = None  # "ecfp2048"
    first_run_all_splits = True
    first_run_all_descs = True
    recalculate = True
    split_type = "random"
    split_proportions = [0.7, 0.15, 0.15]
    file_ext = "pkl"

    papyrus = Papyrus(
        activity_type=activity_type,
        std_smiles=std_smiles,
        verbose_files=verbose_files,
    )

    papyrus(
        n_targets=n_targets,
        desc_prot=desc_prot,
        desc_chem=desc_chem,
        recalculate=recalculate,
        split_type=split_type,
        split_proportions=split_proportions,
        first_run_all_splits=first_run_all_splits,
        first_run_all_descs=first_run_all_descs,
        file_ext=file_ext,
    )
# #
# train_loader = PapyrusDataset(
#     file_path=DATASET_DIR
#     / "papyrus"
#     / "xc50"
#     / "all"
#     / "random_ankh-base_ecfp2048_train.pkl",
#     desc_prot="ankh-base",
#     desc_chem="ecfp2048",
# )
# print(train_loader)
# print(train_loader[0])
# print(len(train_loader))
# print(train_loader[0][0].shape)
#
# datasets = get_papyrus_datasets(
#     n_targets=-1,
#     activity_type="xc50",
#     split_type="random",
#     desc_prot="ankh-base",
#     desc_chem="ecfp2048",
#     ext="pkl",
# )
#
# print(datasets)
