import logging
import argparse
from pathlib import Path
from typing import Union, List, Callable
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
    keep_accession,
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
    apply_label_scaling,
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
        self.logger = create_logger(
            name="Papyrus", file_level="debug", stream_level="info"
        )
        self.config = get_config("papyrus")
        self.chunk_size = self.config.get("chunksize", 1000000)
        self.cols_to_keep = self.config.get("cols_to_keep", None)
        self.dtypes, self.dtypes_protein = self.config.get(
            "dtypes", {}
        ), self.config.get("dtypes_protein", {})
        self.accession_filter = accession
        self.activity_key, self.activity_type = self._parse_activity_key(activity_type)
        (
            self.output_path,
            self.pap_filepath,
            self.pap_path,
            self.pap_protpath,
            self.pap_file,
        ) = self.setup_path(self.activity_key)
        self.std_smiles, self.verbose_files = std_smiles, verbose_files
        self.df_filtered, self.papyrus_protein_data = self.load_or_process_papyrus()

        self.desc_chems = [
            "ecfp1024",
            "ecfp2048",
            "mold2",
            "cddd",
            "fingerprint",
            "moldesc",
        ]
        self.desc_prots = [
            "ankh-base",
            "ankh-large",
            "esm1_t12",
            "esm1_t34",
            "esm1_t6",
            "esm1b",
            "esm_msa1",
            "esm_msa1b",
            "esm1v",
            "protbert",
            "protbert_bfd",
            "unirep",
        ]

    def __call__(
        self,
        n_targets: int = -1,
        descriptor_protein: Union[str, None] = None,
        descriptor_chemical: Union[str, None] = None,
        all_descriptors: bool = False,
        recalculate: bool = False,
        split_type: str = "random",
        split_proportions=None,
        file_ext: str = "pkl",
        batch_size: int = 2,
    ):
        # step 0: assertions
        assert (
            descriptor_protein or descriptor_chemical or all_descriptors
        ), "Either a descriptor must be provided or calculate all of them by setting all_descriptors=True"

        # get the data top-x or all
        df, label_col = self.get_targeted_dataset(n_targets=n_targets)

        split_idx = self.split(df, split_type, split_proportions, return_indices=True)

        split_types = (
            ["random", "scaffold", "time"] if split_type == "all" else [split_type]
        )

        if all_descriptors:
            args_combinations = (
                list(itertools.product(self.desc_prots, self.desc_chems, split_types))
                if n_targets <= 0
                else list(itertools.product([None], self.desc_chems, split_types))
            )
        else:
            args_combinations = [
                (descriptor_protein, descriptor_chemical, s_type)
                for s_type in split_types
            ]

        for descriptor_protein, descriptor_chemical, s_type in args_combinations:
            files_exist, files_paths = check_if_processed_file(
                data_name="papyrus",
                activity_type=self.activity_key,
                n_targets=n_targets,
                split_type=s_type,
                desc_prot=descriptor_protein,
                desc_chem=descriptor_chemical,
                file_ext=file_ext,
            )
            # search for the file otherwise it will be calculated
            if not recalculate and files_exist:
                self.logger.warning(
                    "Found the processed files, if you want to recalculate set recalculate=True"
                )
                continue

            df = self.merge_descriptors(
                df, descriptor_protein, descriptor_chemical, batch_size
            )

            res_dict = {
                sub: df.iloc[split_idx[s_type][sub]] for sub in ["train", "val", "test"]
            }

            cols_to_include = self.get_cols_to_include(
                descriptor_protein, descriptor_chemical, n_targets, label_col
            )

            # step 5: export the data
            self.export_dataset(res_dict, files_paths, cols_to_include)

    @staticmethod
    def setup_path(activity_key):
        activity_key = activity_key.lower()
        output_path = DATASET_DIR / "papyrus" / activity_key
        output_path.mkdir(parents=True, exist_ok=True)
        pap_filepath = output_path / "papyrus_filtered.csv"
        pap_file = pap_filepath.is_file()
        pap_path = pap_filepath if pap_file else DATA_DIR
        pap_protpath = DATASET_DIR / "papyrus" / "papyrus_proteins.csv"

        return output_path, pap_filepath, pap_path, pap_protpath, pap_file

    def load_or_process_papyrus(self):
        if self.pap_file:
            self.logger.info("Loading previously processed Papyrus data ...")
            df_filtered, papyrus_protein_data = self._load_papyrus_files()

        else:
            self.logger.info("PapyrusApi processing input from Raw Papyrus database")
            df_filtered, papyrus_protein_data = self._preprocess_papyrus()

        return df_filtered, papyrus_protein_data

    def get_targeted_dataset(
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

    @staticmethod
    def split(df, split_type, split_proportions, return_indices=False):
        if split_proportions is None:
            split_proportions = [0.7, 0.15, 0.15]

        split_data_dict = split_data(
            df,
            split_type=split_type,
            smiles_col="SMILES",
            time_col="Year",
            fractions=split_proportions,
            return_indices=return_indices,
            seed=42,
        )
        return split_data_dict

    @staticmethod
    def get_cols_to_include(
        desc_prot: str, desc_chem: str, n_targets: int, label_col: list
    ):
        cols_to_include = ["SMILES", desc_chem]  # * used to unpack the list ['ecfp']
        cols_to_include = (
            cols_to_include + ["target_id", desc_prot, "Year"]
            if n_targets <= 0 and desc_prot
            else cols_to_include
        )
        cols_to_include += label_col

        return list(filter(None, cols_to_include))

    @staticmethod
    def export_dataset(subsets_dict, files_paths, cols_to_include=None):
        for subset in ["train", "val", "test"]:
            export_df(
                subsets_dict[subset][cols_to_include], file_path=files_paths[subset]
            )

    def merge_protein_sequences(self, df, target_id_col="target_id"):
        if "Sequence" in df.columns:
            return df
        return df.merge(
            self.papyrus_protein_data[["target_id", "Sequence"]],
            left_on=target_id_col,
            right_on="target_id",
            how="left",
        )

    def merge_descriptors(
        self,
        df=None,
        desc_prot: str = None,
        desc_chem: str = None,
        batch_size: int = 4,
    ):
        try:
            df = self._merge_desc(df, desc_prot, get_embeddings, batch_size=batch_size)
            df = self._merge_desc(df, desc_chem, get_chem_desc)
        except Exception as e:
            self.logger.error(
                f"Error: {e} - {desc_prot} and/or {desc_chem} not calculated"
            )
        return df

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

    def _merge_desc(self, df, desc, desc_func, **kwargs):
        query_col, df = self._prepare_query_col(df, desc)

        if df is None:
            df = self.df_filtered.copy()
        if desc is None or desc in df.columns:
            return df
        self.logger.info(f"Merging {desc} descriptors")
        return desc_func(df, desc, query_col, **kwargs)

    @staticmethod
    def _parse_activity_key(activity_type):
        activity_type = activity_type.lower()
        if isinstance(activity_type, str) and activity_type in ["xc50", "kx"]:
            act_dict = {"xc50": ["IC50", "EC50"], "kx": ["Ki", "Kd"]}
            activity_key = activity_type
            activity_type = act_dict[activity_type]
        else:
            raise ValueError("activity_type must be a string or a list of strings")
        return activity_key, activity_type

    def _load_papyrus_files(self):
        df_filtered = pd.read_csv(self.pap_path, dtype=self.dtypes, low_memory=False)
        papyrus_protein_data = pd.read_csv(
            self.pap_protpath,
            dtype=self.dtypes_protein,
            low_memory=False,
        )

        return df_filtered, papyrus_protein_data

    def _download(self):
        self.logger.info("Downloading Papyrus data ...")
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
            chunksize=self.chunk_size,
            source_path=self.pap_path,
        )
        papyrus_protein_data = read_protein_set(
            source_path=self.pap_path, version="latest"
        )
        return papyrus_data, papyrus_protein_data

    def _preprocess_papyrus(self):
        papyrus_data, papyrus_protein_data = self._reader()
        df_filtered = self._filter_raw_file(papyrus_data, papyrus_protein_data)

        df_nan, df_dup = None, None
        # SMILES standardization
        if self.std_smiles:
            df_filtered, df_nan, df_dup = self._sanitize_smiles(
                df_filtered, self.logger
            )
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

    def _filter_raw_file(self, papyrus_data, papyrus_protein_data):
        filter_1 = keep_quality(data=papyrus_data, min_quality="High")
        filter_2 = keep_match(data=filter_1, column="Protein_Type", values="WT")
        filter_3 = keep_type(data=filter_2, activity_types=self.activity_type)
        filter_4 = keep_organism(
            data=filter_3,
            protein_data=papyrus_protein_data,
            organism="Homo sapiens (Human)",  # self.keep_organism
        )
        if self.accession_filter:
            filter_5 = keep_accession(filter_4, self.accession_filter)
        else:
            filter_5 = filter_4
        df_filtered = consume_chunks(filter_5, progress=True, total=60)

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
        label_scaling_func: Callable[[torch.Tensor], torch.Tensor] = None,
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
        self.label_scaling_func = label_scaling_func

        self.data = apply_label_scaling(
            self.data, self.label_col, self.label_scaling_func
        )

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
                f"features and splits first with Papyrus class."
            )
        datasets[subset] = PapyrusDataset(
            file_path, desc_prot=desc_prot, desc_chem=desc_chem
        )

    return datasets


def main():
    parser = argparse.ArgumentParser(description="Papyrus class interface")

    # Arguments for Papyrus class initialization
    parser.add_argument("--accession", type=str, default=None, help="Accession")
    parser.add_argument("--activity", type=str, default=None, help="Activity type")
    parser.add_argument(
        "--sanitize", action="store_true", help="Standardize and Sanitize SMILES"
    )

    # Arguments for Papyrus class call
    # Argument to specify whether to calculate all descriptors or specific ones
    parser.add_argument(
        "--all-descriptors",
        action="store_true",
        help="Calculate all (pairs) of descriptor combinations",
    )
    parser.add_argument(
        "--descriptor-protein", type=str, default=None, help="Protein descriptor"
    )
    parser.add_argument(
        "--descriptor-chemical", type=str, default=None, help="Chemical descriptor"
    )
    parser.add_argument("--split-type", type=str, default="all", help="Split type")

    parser.add_argument("--verbose", action="store_true", help="Verbose extra files")

    # optional arguments even with all combs
    parser.add_argument(
        "--n-targets", type=int, default=-1, help="Number of top targets"
    )

    parser.add_argument(
        "--recalculate", action="store_true", help="Force recalculation"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Split proportions 0.7,0.15,0.15",
    )
    parser.add_argument("--file-ext", type=str, default="pkl", help="File extension")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")

    args = parser.parse_args()

    # Create an instance of the Papyrus class with the arguments provided by the user
    papyrus = Papyrus(
        accession=args.accession,
        activity_type=args.activity,
        std_smiles=args.sanitize,
        verbose_files=args.verbose,
    )

    # Call the instance with the arguments provided by the user
    split_proportions = (
        [float(item) for item in args.splits.split(",")] if args.splits else None
    )
    papyrus(
        n_targets=args.n_targets,
        descriptor_protein=args.descriptor_protein,
        descriptor_chemical=args.descriptor_chemical,
        all_descriptors=args.all_descriptors,
        recalculate=args.recalculate,
        split_type=args.split_type,
        split_proportions=split_proportions,
        file_ext=args.file_ext,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
