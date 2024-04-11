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
from uqdd import DATA_DIR, DATASET_DIR, DEVICE
from uqdd.utils import create_logger, get_config
from uqdd.utils_chem import standardize_df, get_chem_desc
from uqdd.utils_prot import get_embeddings
from uqdd.data.utils_data import (
    split_data,
    export_dataset,
    load_df,
    load_pickle,
    export_pickle,
    load_desc_preprocessed,
    check_if_processed_file,
    export_tasks,
    apply_median_scaling,
    target_filtering, export_df, merge_scaffolds,
)


class Papyrus:
    def __init__(
        self,
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
        self.desc_prots = [
            "ankh-base",
            "ankh-large",
            # "esm1_t12",
            "esm1_t34",
            # "esm1_t6",
            # "esm_msa1",
            # "esm_msa1b",
            # "esm1v",
            "protbert",
            # "protbert_bfd",
            "unirep",
            # "esm1b",
        ]
        self.desc_chems = [
            # "ecfp1024",
            "ecfp2048",
            # "mold2",
            "mordred",
            "cddd",
            # "fingerprint",
            # "moldesc",
        ]
        self.MT = None  # placeholder for multitask learning

    def __call__(
        self,
        n_targets: int = -1,
        descriptor_protein: Union[str, None] = None,
        descriptor_chemical: Union[str, None] = None,
        all_descriptors: bool = False,
        recalculate: bool = False,
        split_type: str = "random",
        split_proportions: List[float] = None,
        max_k_clusters: int = 100,
        min_datapoints: int = 50,
        min_actives: int = 10,
        active_threshold: float = 6.5,
        only_normal: bool = True,
        file_ext: str = "pkl",
        batch_size: int = 2,
        verbose: bool = False,
    ):
        """
        Main function to process the Papyrus data and export the dataset with the desired descriptors and splits

        Parameters
        ----------
        n_targets : int
            Number of top targets to consider. If n_targets <= 0, all targets will be considered
        descriptor_protein : str
            Protein descriptor to use for the dataset. If None, no protein descriptor will be used
        descriptor_chemical: str
            Chemical descriptor to use for the dataset. If None, no chemical descriptor will be used
        all_descriptors : bool
            Calculate all possible combinations of descriptors
        recalculate: bool
            Recalculate the descriptors even if they have been previously calculated
        split_type: str
            Type of split to use. Options are 'random', 'scaffold', 'scaffold_cluster', 'time' or 'all'
        split_proportions : list
            Proportions for the train, validation and test splits
        max_k_clusters : int
            Maximum number of clusters to use for the scaffold cluster split
        min_datapoints : int
            Minimum number of datapoints for a target to be considered
        min_actives : int
            Minimum number of actives for a target to be considered
        active_threshold : float
            Threshold for the activity value to be considered as active
        only_normal : bool
            Consider only normal activity values
        file_ext : str
            File extension for the exported files
        batch_size : int
            Batch size for the descriptor calculations and merging of the descriptors

        Returns
        -------
        None
        """
        descriptor_protein = self._call_assertions(
            descriptor_protein, descriptor_chemical, all_descriptors, n_targets
        )

        # get the data top-x or all
        df, label_col = self.get_targeted_dataset(
            n_targets, min_datapoints, min_actives, active_threshold, only_normal
        )

        split_types = self._get_split_types(split_type)
        t_tag = "all" if n_targets <= 0 else f"top{n_targets}"
        # TODO change the output path to include t_tag
        export_mcs_path = Path(self.output_path) / t_tag / "mcs"
        figure_path = Path(self.output_path) / t_tag / "mcs_figures"

        if verbose:
            tar_tag = t_tag
            self.logger.info(f"Dataset loaded with {len(df)} datapoints")
            self.logger.info(f"Label column: {label_col}")
            if "target_id" in df.columns:

                unique_targets = df["target_id"].nunique()
                tar_tag += f"_{unique_targets}"
                self.logger.info(f"Unique Targets: {unique_targets}")
            if "SMILES" in df.columns:
                unique_smiles = df["SMILES"].nunique()
                self.logger.info(f"Unique SMILES: {unique_smiles}")

            export_df(df, Path(self.output_path) / t_tag / f"papyrus_filtered_{self.activity_key}_{tar_tag}.csv")

            self.logger.info("Calculating scaffolds")
            df = merge_scaffolds(df, "SMILES")
            export_df(df, Path(self.output_path) / t_tag / f"papyrus_filtered_{self.activity_key}_{tar_tag}_with_scaffolds.csv")
            self.logger.info(f"Unique scaffolds: {df['scaffold'].nunique()}")

        split_idx = self.split(
            df,
            split_types,
            split_proportions,
            max_k_clusters=max_k_clusters,
            fig_output_path=figure_path,
            export_mcs_path=export_mcs_path,
            return_indices=True,
            recalculate=recalculate,
        )

        if all_descriptors:
            args_combinations = (
                list(itertools.product(self.desc_prots, self.desc_chems, split_types))
                if n_targets <= 0
                else list(itertools.product([None], self.desc_chems, split_types))
            )
            d = len(split_types) * len(self.desc_chems)

        else:
            args_combinations = [
                (descriptor_protein, descriptor_chemical, s_type)
                for s_type in split_types
            ]
            d = len(split_types)

        for i, (descriptor_protein, descriptor_chemical, s_type) in enumerate(
            args_combinations, start=1
        ):
            try:
                self.logger.info(
                    f"Processing {descriptor_protein} and {descriptor_chemical} descriptors"
                )
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
                    df = load_desc_preprocessed(
                        df,
                        files_paths,
                        descriptor_protein,
                        descriptor_chemical,
                        "target_id",
                        "SMILES",
                    )
                    continue

                df = self.merge_descriptors(
                    df, descriptor_protein, descriptor_chemical, batch_size
                )

                res_dict = {
                    sub: df.iloc[split_idx[s_type][sub]]
                    for sub in ["train", "val", "test"]
                }

                cols_to_include = self.get_cols_to_include(
                    descriptor_protein, descriptor_chemical, n_targets, label_col
                )

                # step 5: export the data
                export_dataset(res_dict, files_paths, cols_to_include)

                if i % d == 0 and descriptor_protein:
                    self.logger.info(
                        f"Processed all {i} combinations of {descriptor_protein},"
                        f" deleting the column from the dataframe to save memory"
                    )
                    df.drop(columns=[descriptor_protein], inplace=True)

            except Exception as e:
                self.logger.error(f"Error: {e}")
                continue

    def _call_assertions(
        self, descriptor_protein, descriptor_chemical, all_descriptors, n_targets
    ):
        self.MT = n_targets > 0
        if n_targets > 0 and descriptor_protein:
            # log warning
            self.logger.warning(
                "For multitask learning, only chemical descriptors will be used."
                "Setting descriptor_protein to None"
            )
            descriptor_protein = None
        assert (
            descriptor_protein or descriptor_chemical or all_descriptors
        ), "Either a descriptor must be provided or calculate all of them by setting all_descriptors=True"
        return descriptor_protein

    def _get_split_types(self, split_type: str = None):
        if split_type == "all":
            if self.MT:
                return ["random", "scaffold"] # , "scaffold_cluster"
            return ["random", "scaffold", "time"] # , "scaffold_cluster"
        return [split_type]

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
        min_datapoints: int = 50,
        min_actives: int = 10,
        active_threshold: float = 6.5,
        only_normal: bool = True,
    ):
        df = target_filtering(
            self.df_filtered,
            "target_id",
            "pchembl_value_Mean",
            min_datapoints,
            min_actives,
            active_threshold,
            only_normal,
        )
        label_col = ["pchembl_value_Mean"]
        if n_targets > 0:
            df, top_targets = self._get_top_targets(df, n_targets)

            # Set the DataFrame's index to be both 'SMILES' and 'connectivity'
            # df.set_index(["SMILES", "connectivity"], inplace=True)

            # if multitask:
            pivoted = pd.pivot_table(
                df,
                index=["SMILES", "connectivity"],
                columns="accession",
                values="pchembl_value_Mean",
                aggfunc="first",
            )

            # reset the index to make the "smiles" column a regular column
            pivoted.reset_index(level=["SMILES", "connectivity"], inplace=True)
            # replace any missing values with NaN
            df = pivoted.fillna(value=np.nan)

            label_col = list(top_targets.index)
            export_tasks(
                data_name="papyrus",
                activity=self.activity_key,
                n_targets=n_targets,
                label_col=label_col,
            )
        # else:
        #     # dataset/papyrus/xc50/all/
        #     # df = self.df_filtered.copy()  # the whole thing
        #     label_col = ["pchembl_value_Mean"]

        return df, label_col

    def split(
        self,
        df,
        split_type,
        split_proportions,
        max_k_clusters=100,
        fig_output_path=None,
        export_mcs_path=None,
        return_indices=False,
        recalculate=False,
    ):
        # calculate the splits or load them if they exist
        split_path = (
            self.output_path
            / f"split_data_dict{'_indices' if return_indices else ''}.pkl"
        )
        if split_path.is_file() and not recalculate:
            self.logger.info("Loading previously calculated splits")
            split_data_dict = load_pickle(split_path)
        else:
            self.logger.info("Calculating the splits")
            if split_proportions is None:
                split_proportions = [0.7, 0.15, 0.15]
            split_data_dict = split_data(
                df,
                split_type=split_type,
                smiles_col="SMILES",
                time_col="Year",
                fractions=split_proportions,
                max_k_clusters=max_k_clusters,
                fig_output_path=fig_output_path,
                export_mcs_path=export_mcs_path,
                return_indices=return_indices,
                seed=42,
            )
            # save the splits
            self.logger.info(f"Exporting the splits to {split_path}")
            export_pickle(split_data_dict, split_path)

        return split_data_dict

    @staticmethod
    def get_cols_to_include(
        desc_prot: str, desc_chem: str, n_targets: int, label_col: list
    ):
        cols_to_include = ["SMILES", "connectivity", desc_chem]  # * used to unpack the list ['ecfp']
        cols_to_include = (
            cols_to_include + ["target_id", desc_prot, "Year"]
            if n_targets <= 0 and desc_prot
            else cols_to_include
        )
        cols_to_include += label_col

        return list(filter(None, cols_to_include))

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
            df = self._merge_desc(df, desc_chem, get_chem_desc)
            df = self._merge_desc(df, desc_prot, get_embeddings, batch_size=batch_size)
        except Exception as e:
            self.logger.error(
                f"Error within merge_descriptors func: {e} \n {desc_prot} and/or {desc_chem} not calculated"
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

        df_filtered = consume_chunks(filter_4, progress=True, total=60)  #

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
            suppress_exception=False,
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
        task_type: str = "regression",
        calc_median: bool = False,
        median_scaling: bool = False,
        median_point: float = 6.0,
        logger: Union[None, logging.Logger] = None,
        device=DEVICE,
        **kwargs,
    ) -> None:
        # self.device = device
        data = load_df(file_path, **kwargs)

        dir_path = Path(file_path).parent

        labels_filepath = dir_path / "target_col.pkl"
        self.MT = labels_filepath.is_file()

        self.label_col = (
            load_pickle(labels_filepath) if self.MT else ["pchembl_value_Mean"]
        )
        # self.desc_prot = desc_prot
        # self.desc_chem = desc_chem
        # TODO : relevant cols for the dataset
        # relevant_cols = [desc_prot, desc_chem, *self.label_col]
        # relevant_cols = list(filter(None, relevant_cols))
        # data = data[relevant_cols].dropna().reset_index(drop=True)

        self.task_type = task_type
        self.median_point = median_point
        self.median_scaling = median_scaling
        self.calc_median = calc_median

        data, self.median_point = apply_median_scaling(
            data,
            self.label_col,
            self.median_point,
            calc_median=self.calc_median,
            median_scaling=self.median_scaling,
            logger=logger,
        )

        if self.task_type == "classification":
            data[self.label_col] = (data[self.label_col] > self.median_point).astype(
                int
            )

            # self.data[self.label_col] = np.where(
            #     self.data[self.label_col] > self.median_point, 1, 0
            # )
            # self.data[self.label_col] = self.data[self.label_col].astype("category")
        # self.data.to(device)

        # if self.desc_chem is not None:
        #     self.chem_desc = (
        #         torch.from_numpy(np.stack(data[self.desc_chem].values))
        #         .float()
        #         .to(device)
        #     )
        #     # chem_desc_np = np.array(data[self.desc_chem].tolist())
        #     # self.chem_desc = torch.from_numpy(chem_desc_np).float()  # .to(device)
        # if not self.MT and self.desc_prot is not None:
        #     self.prot_desc = torch.from_numpy(np.stack(data[self.desc_prot].values))
        # Convert descriptor columns to tensors once, avoid conversion in __getitem__
        self.labels = torch.tensor(
            data[self.label_col].values, dtype=torch.float32, device=device
        )
        # if desc_chem is not None:
        np_desc_chem = np.stack(data[desc_chem].apply(pd.to_numeric, errors='coerce').values).astype(np.float32)
        np_desc_prot = np.stack(data[desc_prot].apply(pd.to_numeric, errors='coerce').values).astype(np.float32)

        self.chem_desc = torch.tensor(
            np_desc_chem, dtype=torch.float32, device=device
        )

        if desc_prot is not None:
            self.prot_desc = torch.tensor(
                np_desc_prot, dtype=torch.float32, device=device
            )
        else:  # create an empty tensor for the protein descriptors
            self.prot_desc = torch.zeros(
                self.chem_desc.shape[0], 1, dtype=torch.float32, device=device
            )
            # prot_desc_np = np.array(data[self.desc_prot].tolist())
            # self.prot_desc = torch.from_numpy(prot_desc_np).float()  # .to(device)
        # self.labels = torch.from_numpy(data[self.label_col].values).float().to(device)
        self.data = data
        # )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.prot_desc[idx], self.chem_desc[idx]), self.labels[idx]
        # if self.MT:
        #     return self.chem_desc[idx], self.labels[idx]
        # else:
        #     return (self.prot_desc[idx], self.chem_desc[idx]), self.labels[idx]


def get_datasets(
    n_targets: int = -1,
    activity_type: str = "xc50",
    split_type: str = "random",
    desc_prot: Union[str, None] = None,
    desc_chem: Union[str, None] = None,
    median_scaling: bool = False,
    task_type: str = "regression",
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
    median_point = 6.0
    for subset in ["train", "val", "test"]:
        file_path = dir_path / f"{filename_prefix}_{subset}.{ext}"
        if not file_path.is_file():
            # TODO calculate them here if not found
            raise FileNotFoundError(
                f"File {subset} not found: {file_path} - you need to precalculate the dataset "
                f"features and splits first with Papyrus class."
            )

        calc_median = (
            True
            if subset == "train" and (median_scaling or task_type == "classification")
            else False
        )

        dataset = PapyrusDataset(
            file_path,
            desc_prot=desc_prot,
            desc_chem=desc_chem,
            task_type=task_type,
            calc_median=calc_median,
            median_scaling=median_scaling,
            median_point=median_point,
            logger=logger,
        )
        median_point = dataset.median_point
        datasets[subset] = dataset
    dfs = pd.concat([datasets[subset].data for subset in ["train", "val", "test"]])
    logger.info(f"Dataset loaded with {len(dfs)} datapoints")
    logger.info(
        f"Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}"
    )
    # logger.info(f"Total unique Targets: {dfs['target_id'].nunique()}")
    # logger.info(f"Total unique SMILES: {dfs['SMILES'].nunique()}")

    return datasets


def main():
    parser = argparse.ArgumentParser(description="Papyrus class interface")

    # Arguments for Papyrus class initialization
    parser.add_argument("--activity", type=str, default=None, help="Activity type")
    parser.add_argument(
        "--sanitize", action="store_true", help="Standardize and Sanitize SMILES"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose extra files")
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

    # optional arguments even with all combs
    parser.add_argument(
        "--n-targets", type=int, default=-1, help="Number of top targets"
    )
    # Argument to specify whether to recalculate the descriptors True or False
    parser.add_argument(
        "--recalculate", type=bool, default=False, help="Recalculate descriptors"
    )

    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Split proportions 0.7,0.15,0.15",
    )
    parser.add_argument("--file-ext", type=str, default="pkl", help="File extension")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--max-k-clusters", type=int, default=100, help="Max k clusters"
    )
    parser.add_argument("--min-datapoints", type=int, default=50, help="Min datapoints")
    parser.add_argument("--min-actives", type=int, default=10, help="Min actives")
    parser.add_argument(
        "--active-threshold", type=float, default=6.5, help="Active threshold"
    )
    parser.add_argument(
        "--only-normal",
        type=bool,
        default=False,
        help="Consider only normal activity values",
    )

    args = parser.parse_args()

    # Create an instance of the Papyrus class with the arguments provided by the user
    papyrus = Papyrus(
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
        max_k_clusters=args.max_k_clusters,
        min_datapoints=args.min_datapoints,
        min_actives=args.min_actives,
        active_threshold=args.active_threshold,
        only_normal=args.only_normal,
        file_ext=args.file_ext,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
    # # Example of how to use the Papyrus class
    # activity_type = "kx"
    # std_smiles = True
    # verbose_files = False
    #
    # # call args
    # n_targets = 20
    # desc_prot = None  # "ankh-base"  # For testing - it should become none
    # desc_chem = None  # "mordred"
    # split_type = "random"
    # all_descs = True
    # recalculate = True
    # split_proportions = [0.7, 0.15, 0.15]
    # file_ext = "pkl"
    #
    # papyrus = Papyrus(
    #     activity_type=activity_type,
    #     std_smiles=std_smiles,
    #     verbose_files=verbose_files,
    # )
    #
    # papyrus(
    #     n_targets=n_targets,
    #     descriptor_protein=desc_prot,
    #     descriptor_chemical=desc_chem,
    #     all_descriptors=all_descs,
    #     recalculate=recalculate,
    #     split_type=split_type,
    #     split_proportions=split_proportions,
    #     max_k_clusters=100,
    #     min_datapoints=50,
    #     min_actives=10,
    #     active_threshold=6.5,
    #     only_normal=False,
    #     file_ext=file_ext,
    #     verbose=True
    # )
    # #
    # # reg_dataset = get_datasets(
    # #     n_targets=n_targets,
    # #     activity_type=activity_type,
    # #     split_type=split_type,
    # #     desc_prot=desc_prot,
    # #     desc_chem=desc_chem,
    # #     median_scaling=True,
    # #     task_type="regression",
    # # )
    #
    # # cl_dataset = get_datasets(
    # #     n_targets=n_targets,
    # #     activity_type=activity_type,
    # #     split_type=split_type,
    # #     desc_prot=desc_prot,
    # #     desc_chem=desc_chem,
    # #     median_scaling=False,
    # #     task_type="classification",
    # # )
    # # cl_dataset_med = get_datasets(
    # #     n_targets=n_targets,
    # #     activity_type=activity_type,
    # #     split_type=split_type,
    # #     desc_prot=desc_prot,
    # #     desc_chem=desc_chem,
    # #     median_scaling=True,
    # #     task_type="classification",
    # # )
    #
    # print("Done")