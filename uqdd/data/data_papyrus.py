import argparse
import itertools
import logging
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from papyrus_scripts.download import download_papyrus
from papyrus_scripts.preprocess import (
    keep_quality,
    keep_match,
    keep_type,
    keep_organism,
    consume_chunks,
)
from papyrus_scripts.reader import read_papyrus, read_protein_set
from torch.utils.data import Dataset

from uqdd import DATA_DIR, DATASET_DIR, DEVICE
from uqdd.data.utils_data import (
    split_data,
    export_dataset,
    load_desc_preprocessed,
    check_if_processed_file,
    export_tasks,
    apply_median_scaling,
    target_filtering,
    fig_label_distribution_across_splits,
    fig_label_distribution,
    from_split_data_to_idx,
)
from uqdd.utils import (
    create_logger,
    get_config,
    load_pickle,
    save_df,
    load_df,
)
from uqdd.utils_chem import standardize_df, get_chem_desc, merge_scaffolds
from uqdd.utils_prot import get_embeddings


class Papyrus:
    """
    End-to-end data pipeline for preparing the Papyrus dataset for PCM and MT setups.

    This class handles reading, filtering, descriptor computation/merging, splitting,
    and exporting datasets ready for training and evaluation.

    Notes
    -----
    - When `n_targets > 0` (multitask learning), only chemical descriptors are used.
    - When `n_targets <= 0` (PCM), both protein and chemical descriptors can be merged.
    - The mkdocstrings configuration expects NumPy-style docstrings (numpydoc).
    """

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
            "esm1_t34",
            "protbert",
            "unirep",
        ]
        self.desc_chems = [
            "ecfp2048",
            "mordred",
            "cddd",
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
            stratify_by: Union[str, None] = None,
            max_k_clusters: int = 500,
            optimal_k: int = None,
            min_datapoints: int = 50,
            min_actives: int = 10,
            active_threshold: float = 6.5,
            only_normal: bool = False,
            file_ext: str = "pkl",
            batch_size: int = 2,
            verbose: bool = False,
    ):
        """
        Process the Papyrus dataset and export it with the specified descriptors and splits.

        Parameters
        ----------
        n_targets : int, optional
            Number of top targets to consider. If ``<= 0``, all targets will be included.
            Default is ``-1``.
        descriptor_protein : str or None, optional
            Protein descriptor to use. If ``None``, no protein descriptor is used.
        descriptor_chemical : str or None, optional
            Chemical descriptor to use. If ``None``, no chemical descriptor is used.
        all_descriptors : bool, optional
            Whether to calculate all possible descriptor combinations. Default is ``False``.
        recalculate : bool, optional
            Force recalculation of descriptors even if they exist. Default is ``False``.
        split_type : str, optional
            Type of dataset split to use (e.g., "random", "scaffold", "scaffold_cluster", "time").
            Default is "random".
        split_proportions : list of float, optional
            Proportions for train/val/test, e.g., ``[0.7, 0.15, 0.15]``.
        stratify_by : str or None, optional
            Column used for stratifying splits (e.g., "target_id", "scaffold").
        max_k_clusters : int, optional
            Maximum number of clusters for scaffold clustering. Default is ``500``.
        optimal_k : int or None, optional
            Precomputed optimal number of clusters. Default is ``None``.
        min_datapoints : int, optional
            Minimum required datapoints per target. Default is ``50``.
        min_actives : int, optional
            Minimum required actives per target. Default is ``10``.
        active_threshold : float, optional
            Threshold for defining active compounds. Default is ``6.5``.
        only_normal : bool, optional
            Whether to include only normal activity values. Default is ``False``.
        file_ext : str, optional
            File extension for exported datasets. Default is ``"pkl"``.
        batch_size : int, optional
            Batch size for descriptor computations. Default is ``2``.
        verbose : bool, optional
            Whether to enable verbose logging. Default is ``False``.

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
        export_path = Path(self.output_path) / t_tag  # / "mcs"
        # figure_path = Path(self.output_path) / t_tag / "mcs_figures"

        if verbose:
            tar_tag = t_tag
            self.logger.debug(f"Dataset loaded with {len(df)} datapoints")
            self.logger.debug(f"Label column: {label_col}")
            if "target_id" in df.columns:
                unique_targets = df["target_id"].nunique()
                tar_tag += f"_{unique_targets}"
                self.logger.debug(f"Unique Targets: {unique_targets}")
            if "SMILES" in df.columns:
                unique_smiles = df["SMILES"].nunique()
                self.logger.debug(f"Unique SMILES: {unique_smiles}")

            save_df(
                df, export_path / f"papyrus_filtered_{self.activity_key}_{tar_tag}.csv"
            )

            self.logger.info("Calculating scaffolds")
            df = merge_scaffolds(df, "SMILES")
            save_df(
                df,
                export_path
                / f"papyrus_filtered_{self.activity_key}_{tar_tag}_with_scaffolds.csv",
            )
            self.logger.debug(f"Unique scaffolds: {df['scaffold'].nunique()}")
        #
        split_idx = self.split(
            df,
            split_types,
            split_proportions,
            label_col,
            stratify_by,
            max_k_clusters=max_k_clusters,
            optimal_k=optimal_k,
            # fig_output_path=figure_path,
            export_path=export_path,
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
            self,
            descriptor_protein: Optional[str],
            descriptor_chemical: Optional[str],
            all_descriptors: bool,
            n_targets: int,
    ) -> Optional[str]:
        """
        Ensure valid descriptor selection based on multitask settings.

        Parameters
        ----------
        descriptor_protein : str or None
            Selected protein descriptor.
        descriptor_chemical : str or None
            Selected chemical descriptor.
        all_descriptors : bool
            Whether to compute all possible descriptor combinations.
        n_targets : int
            Number of top targets to include.

        Returns
        -------
        str or None
            Adjusted protein descriptor value based on the multitask setting.
        """
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

    def _get_split_types(self, split_type: Optional[str] = None) -> List[str]:
        """
        Determine the applicable dataset split types.

        Parameters
        ----------
        split_type : str or None, optional
            Type of split requested.

        Returns
        -------
        list of str
            Applicable split types.
        """
        if split_type == "all":
            if self.MT:
                return [
                    "random",
                    "scaffold",
                    "scaffold_cluster",
                ]  # , "scaffold_cluster"
            return [
                "random",
                "scaffold",
                "time",
                "scaffold_cluster",
            ]  # , "scaffold_cluster"
        return [split_type]

    @staticmethod
    def setup_path(activity_key: str) -> Tuple[Path, Path, Path, Path, bool]:
        """
        Set up paths for processing and storing Papyrus data.

        Parameters
        ----------
        activity_key : str
            Key representing the activity type.

        Returns
        -------
        tuple
            ``(output_path, pap_filepath, pap_path, pap_protpath, pap_file_exists)``.
        """
        activity_key = activity_key.lower()
        output_path = DATASET_DIR / "papyrus" / activity_key
        output_path.mkdir(parents=True, exist_ok=True)
        pap_filepath = output_path / "papyrus_filtered.csv"
        pap_file = pap_filepath.is_file()
        pap_path = pap_filepath if pap_file else DATA_DIR
        pap_protpath = DATASET_DIR / "papyrus" / "papyrus_proteins.csv"

        return output_path, pap_filepath, pap_path, pap_protpath, pap_file

    def load_or_process_papyrus(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load existing Papyrus data or process it from raw sources.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            DataFrame containing filtered dataset and corresponding protein data.
        """
        if self.pap_file:
            self.logger.info("Loading previously processed Papyrus data ...")
            df_filtered, papyrus_protein_data = self._load_papyrus_files()

        else:
            self.logger.info("PapyrusApi processing input from Raw Papyrus database")
            df_filtered, papyrus_protein_data = self._preprocess_papyrus()

        return df_filtered, papyrus_protein_data

    def get_targeted_dataset(
            self,
            n_targets: int = 20,
            min_datapoints: int = 50,
            min_actives: int = 10,
            active_threshold: float = 6.5,
            only_normal: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Filter and retrieve a dataset based on target selection criteria.

        Parameters
        ----------
        n_targets : int, optional
            Number of top targets to retain. Default is ``20``.
        min_datapoints : int, optional
            Minimum datapoints required per target. Default is ``50``.
        min_actives : int, optional
            Minimum number of active compounds per target. Default is ``10``.
        active_threshold : float, optional
            Threshold for defining active compounds. Default is ``6.5``.
        only_normal : bool, optional
            Whether to include only normal activity values. Default is ``True``.

        Returns
        -------
        (pd.DataFrame, list of str)
            Filtered dataset and corresponding label columns.
        """
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
        return df, label_col

    def split(
            self,
            df: pd.DataFrame,
            split_type: str,
            split_proportions: Optional[List[float]],
            label_col: List[str],
            stratify_by: Optional[str] = None,
            max_k_clusters: int = 500,
            optimal_k: Optional[int] = None,
            export_path: Optional[Path] = None,
            recalculate: bool = False,
    ) -> Dict[str, List[int]]:
        """
        Split the dataset into training, validation, and test sets.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to be split.
        split_type : str
            The method of splitting (e.g., "random", "scaffold").
        split_proportions : list of float or None
            Proportions for train, validation, and test sets.
        label_col : list of str
            Label column(s).
        stratify_by : str or None, optional
            Column used for stratification.
        max_k_clusters : int, optional
            Maximum clusters for scaffold-based splits. Default is ``500``.
        optimal_k : int or None, optional
            Precomputed optimal number of clusters.
        export_path : Path or None, optional
            Path to save split indices.
        recalculate : bool, optional
            Whether to recalculate splits if they exist. Default is ``False``.

        Returns
        -------
        dict
            Dictionary containing train, validation, and test split indices.
        """
        if export_path is None:
            export_path = self.output_path
        fig_path = export_path / "data_figures"
        fig_path.mkdir(parents=True, exist_ok=True)
        data_splits = split_data(
            df,
            split_type=split_type,
            smiles_col="SMILES",
            time_col="Year",
            stratify_col=stratify_by,
            fractions=split_proportions,
            max_k_clusters=max_k_clusters,
            optimal_k=optimal_k,
            export_path=export_path,
            return_indices=False,
            recalculate=recalculate,
            seed=42,
            logger=self.logger,
        )
        # figures about the splits distribution
        if not self.MT:
            fig_label_distribution(df, label_col, fig_path)
            fig_label_distribution_across_splits(data_splits, label_col, fig_path)

        split_idx = from_split_data_to_idx(data_splits)

        return split_idx

    @staticmethod
    def get_cols_to_include(
            desc_prot: str, desc_chem: str, n_targets: int, label_col: list
    ):
        """
        Get the list of columns to include in the exported dataset.

        Parameters
        ----------
        desc_prot : str
            Protein descriptor.
        desc_chem : str
            Chemical descriptor.
        n_targets : int
            Number of top targets.
        label_col : list of str
            Label columns.

        Returns
        -------
        list of str
            Columns to include.
        """
        cols_to_include = [
            "SMILES",
            "connectivity",
            desc_chem,
        ]  # * used to unpack the list ['ecfp']
        cols_to_include = (
            cols_to_include + ["target_id", desc_prot, "Year"]
            if n_targets <= 0 and desc_prot
            else cols_to_include
        )
        cols_to_include += label_col

        return list(filter(None, cols_to_include))

    def merge_protein_sequences(
            self, df: pd.DataFrame, target_id_col: str = "target_id"
    ) -> pd.DataFrame:
        """
        Merge protein sequences into the dataset using target identifiers.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing target identifiers.
        target_id_col : str, optional
            Column name for target identifiers. Default is ``"target_id"``.

        Returns
        -------
        pd.DataFrame
            Dataset with merged protein sequences.
        """
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
            df: Optional[pd.DataFrame] = None,
            desc_prot: Optional[str] = None,
            desc_chem: Optional[str] = None,
            batch_size: int = 4,
    ) -> pd.DataFrame:
        """
        Merge chemical and protein descriptors into the dataset.

        Parameters
        ----------
        df : pd.DataFrame or None, optional
            Dataset to merge descriptors into. If ``None``, uses ``self.df_filtered``.
        desc_prot : str or None, optional
            Protein descriptor to merge.
        desc_chem : str or None, optional
            Chemical descriptor to merge.
        batch_size : int, optional
            Batch size for descriptor processing. Default is ``4``.

        Returns
        -------
        pd.DataFrame
            Dataset with merged descriptors.
        """
        try:
            df = self._merge_desc(df, desc_chem, get_chem_desc)
            df = self._merge_desc(df, desc_prot, get_embeddings, batch_size=batch_size)
        except Exception as e:
            self.logger.error(
                f"Error within merge_descriptors func: {e} \n {desc_prot} and/or {desc_chem} not calculated"
            )
        return df

    def _prepare_query_col(
            self, df: pd.DataFrame, desc: Optional[str]
    ) -> Tuple[Optional[str], pd.DataFrame]:
        """
        Determine the appropriate query column based on the descriptor type.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to process.
        desc : str or None
            Descriptor name.

        Returns
        -------
        (str or None, pd.DataFrame)
            The query column name and the updated dataset.
        """
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

    def _merge_desc(
            self, df: pd.DataFrame, desc: Optional[str], desc_func: callable, **kwargs
    ) -> pd.DataFrame:
        """
        Merge a single descriptor into the dataset using the provided function.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset.
        desc : str or None
            Descriptor to merge.
        desc_func : callable
            Function to compute the descriptor.
        **kwargs :
            Additional arguments for the descriptor function.

        Returns
        -------
        pd.DataFrame
            The dataset with the descriptor merged.
        """
        query_col, df = self._prepare_query_col(df, desc)

        if df is None:
            df = self.df_filtered.copy()
        if desc is None or desc in df.columns:
            return df
        self.logger.info(f"Merging {desc} descriptors")
        return desc_func(df, desc, query_col, **kwargs)

    @staticmethod
    def _parse_activity_key(
            activity_type: Union[str, List[str]]
    ) -> Tuple[str, List[str]]:
        """
        Parse and map the activity key to corresponding activity types.

        Parameters
        ----------
        activity_type : str or list of str
            The activity type(s) to parse.

        Returns
        -------
        (str, list of str)
            The activity key and corresponding list of activity types.
        """
        activity_type = activity_type.lower()
        if isinstance(activity_type, str) and activity_type in ["xc50", "kx"]:
            act_dict = {"xc50": ["IC50", "EC50"], "kx": ["Ki", "Kd"]}
            activity_key = activity_type
            activity_type = act_dict[activity_type]
        else:
            raise ValueError("activity_type must be a string or a list of strings")
        return activity_key, activity_type

    def _load_papyrus_files(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed Papyrus dataset and corresponding protein data.

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            The dataset and protein data.
        """
        df_filtered = pd.read_csv(self.pap_path, dtype=self.dtypes, low_memory=False)
        papyrus_protein_data = pd.read_csv(
            self.pap_protpath,
            dtype=self.dtypes_protein,
            low_memory=False,
        )

        return df_filtered, papyrus_protein_data

    def _download(self) -> None:
        """
        Download the Papyrus dataset if it is not already available.
        """
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

    def _reader(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read the Papyrus dataset from raw files.

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            The dataset and protein data.
        """
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

    def _preprocess_papyrus(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess raw Papyrus data, including filtering and standardization.

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            The filtered dataset and protein data.
        """
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
    def _get_top_targets(
            df: pd.DataFrame, n_targets: int = 20
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select the top protein targets based on the number of measurements.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset.
        n_targets : int, optional
            Number of top targets to select. Default is ``20``.

        Returns
        -------
        (pd.DataFrame, pd.Series)
            The filtered dataset and the selected top targets.
        """
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

    def _filter_raw_file(
            self, papyrus_data: pd.DataFrame, papyrus_protein_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply filtering criteria to the raw Papyrus dataset.

        Parameters
        ----------
        papyrus_data : pd.DataFrame
            Raw Papyrus dataset.
        papyrus_protein_data : pd.DataFrame
            Protein dataset.

        Returns
        -------
        pd.DataFrame
            Filtered dataset.
        """
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
        """
        Standardize SMILES strings in the dataset.

        Parameters
        ----------
        df_filtered : pd.DataFrame
            Filtered dataset with SMILES to standardize.
        logger : logging.Logger or None, optional
            Logger for debugging purposes.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
            DataFrames with standardized SMILES, NaN entries, and duplicates.
        """
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
        """
        Save verbose files for NaN and duplicate SMILES entries.

        Parameters
        ----------
        df_nan : pd.DataFrame
            DataFrame with NaN SMILES entries.
        df_dup : pd.DataFrame
            DataFrame with duplicate SMILES entries.

        Returns
        -------
        None
        """
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
        """
        Initialize a dataset wrapper for Papyrus PCM/MT tasks.

        Parameters
        ----------
        file_path : str or Path
            Path to the dataset file.
        desc_prot : str or None, optional
            Protein descriptor name, if applicable.
        desc_chem : str or None, optional
            Chemical descriptor name.
        task_type : {"regression", "classification"}, optional
            Type of learning task. Default is ``"regression"``.
        calc_median : bool, optional
            Whether to calculate the median point from the training dataset.
        median_scaling : bool, optional
            Whether to apply median scaling to labels.
        median_point : float, optional
            The median threshold for classification tasks. Default is ``6.0``.
        logger : logging.Logger or None, optional
            Logger for debugging purposes.
        device : torch.device
            PyTorch device to store tensors.
        """
        # self.device = device
        data = load_df(file_path, **kwargs)

        dir_path = Path(file_path).parent

        labels_filepath = dir_path / "target_col.pkl"
        self.MT = labels_filepath.is_file()

        self.label_col = (
            load_pickle(labels_filepath) if self.MT else ["pchembl_value_Mean"]
        )
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

        self.labels = torch.tensor(
            data[self.label_col].values, dtype=torch.float32, device=device
        )
        np_desc_chem = np.stack(
            data[desc_chem].apply(pd.to_numeric, errors="coerce").values
        ).astype(np.float32)
        self.chem_desc = torch.tensor(np_desc_chem, dtype=torch.float32, device=device)

        if desc_prot is not None:
            np_desc_prot = np.stack(
                data[desc_prot].apply(pd.to_numeric, errors="coerce").values
            ).astype(np.float32)
            self.prot_desc = torch.tensor(
                np_desc_prot, dtype=torch.float32, device=device
            )
        else:  # create an empty tensor for the protein descriptors
            self.prot_desc = torch.zeros(
                self.chem_desc.shape[0], 1, dtype=torch.float32, device=device
            )
        self.data = data

    def __len__(self) -> int:
        """Return the number of data points in the dataset."""
        return len(self.labels)

    def __getitem__(
            self, idx: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Retrieve a single data point from the dataset."""
        return (self.prot_desc[idx], self.chem_desc[idx]), self.labels[idx]


def get_datasets(
        n_targets: int = -1,
        activity_type: str = "xc50",
        split_type: str = "random",
        desc_prot: Optional[str] = None,
        desc_chem: Optional[str] = None,
        median_scaling: bool = False,
        task_type: str = "regression",
        ext: str = "pkl",
        logger: Optional[logging.Logger] = None,
        device=DEVICE,
) -> Dict[str, PapyrusDataset]:
    """
    Load Papyrus datasets for training, validation, and testing.

    Parameters
    ----------
    n_targets : int, optional
        Number of top targets to include; if ``<= 0``, includes all targets.
    activity_type : str, optional
        Type of activity measurement (e.g., "xc50").
    split_type : str, optional
        Type of data split ("random", "scaffold", "time", etc.).
    desc_prot : str or None, optional
        Protein descriptor name.
    desc_chem : str or None, optional
        Chemical descriptor name.
    median_scaling : bool, optional
        Whether to apply median scaling to labels.
    task_type : {"regression", "classification"}, optional
        Task type.
    ext : str, optional
        File extension of the dataset files.
    logger : logging.Logger or None, optional
        Logger for debugging purposes.
    device : torch.device
        PyTorch device to store tensors.

    Returns
    -------
    dict of str -> PapyrusDataset
        Dictionary containing the training, validation, and test datasets.
    """
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
        logger.debug(
            "Initializing dataset for multitask learning; only chemical descriptors will be used."
        )
        filename_prefix = f"{split_type}_{desc_chem}"
    else:
        if desc_prot:
            logger.debug(
                "Initializing pcm dataset for single-task learning; both protein and chemical descriptors will be used."
            )
            filename_prefix = f"{split_type}_{desc_prot}_{desc_chem}"
        else:
            logger.debug(
                "Initializing dataset for single-task learning; only chemical descriptors will be used."
            )
            filename_prefix = f"{split_type}_{desc_chem}"

    datasets = {}
    median_point = 6.0
    for subset in ["train", "val", "test"]:
        file_path = dir_path / f"{filename_prefix}_{subset}.{ext}"
        if not file_path.is_file():
            logger.warning(
                f"File {subset} not found: {file_path} - "
                f"calculating it now with default settings - "
                f"for non-default settings, please use Papyrus class first."
            )
            Papyrus(activity_type=activity_type)(
                n_targets=n_targets,
                descriptor_protein=desc_prot,
                descriptor_chemical=desc_chem,
                split_type=split_type,
            )
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"File {subset} still not found: {file_path} "
                    f"even after trying to calculate with Papyrus class"
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
            device=device,
        )
        median_point = dataset.median_point
        datasets[subset] = dataset
    dfs = pd.concat([datasets[subset].data for subset in ["train", "val", "test"]])
    logger.debug(f"Dataset loaded with {len(dfs)} datapoints")
    logger.debug(
        f"Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}"
    )

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
        "--max-k-clusters", type=int, default=10000, help="Max k clusters"
    )
    parser.add_argument("--optimal-k", type=int, default=None, help="Optimal k")
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
    parser.add_argument(
        "--stratify-by", type=str, default="cluster", help="Stratify by column"
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
        stratify_by=args.stratify_by,
        max_k_clusters=args.max_k_clusters,
        optimal_k=args.optimal_k,
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
    # # desc_prot = "ankh-large"  # "ankh-base"  # For testing - it should become none
    # # desc_chem = "ecfp2048"  # "mordred"
    # split_type = "all"
    # stratify_by = "cluster" #"target_id" # "cluster" # "scaffold"
    # all_descs = True
    # recalculate = True
    # split_proportions = [0.7, 0.15, 0.15]
    # file_ext = "pkl"
    # max_k = 1000000
    # # optimal_k = 11974
    #
    # papyrus = Papyrus(
    #     activity_type=activity_type,
    #     std_smiles=std_smiles,
    #     verbose_files=verbose_files,
    # )
    #
    # papyrus(
    #     n_targets=n_targets,
    #     # descriptor_protein=desc_prot,
    #     # descriptor_chemical=desc_chem,
    #     all_descriptors=all_descs,
    #     recalculate=recalculate,
    #     split_type=split_type,
    #     split_proportions=split_proportions,
    #     stratify_by=stratify_by,
    #     max_k_clusters=max_k,
    #     # optimal_k=optimal_k,
    #     min_datapoints=50,
    #     min_actives=10,
    #     active_threshold=6.5,
    #     only_normal=False,
    #     file_ext=file_ext,
    #     verbose=True
    # )
