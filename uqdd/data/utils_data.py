import logging
import os
import pickle
from pathlib import Path
from typing import Union, List, Dict, Tuple, Callable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split

from uqdd import DATASET_DIR
from uqdd.utils import load_pickle, save_df, load_df, save_pickle
from uqdd.utils_chem import merge_scaffolds, clustering

string_types = (type(b""), type(""))


def export_tasks(
        data_name: str, activity: str, n_targets: int, label_col: List[str]
) -> None:
    """
    Export the selected task labels to a pickle file.

    Parameters
    ----------
    data_name : str
        Name of the dataset.
    activity : str
        Activity type.
    n_targets : int
        Number of targets.
    label_col : list of str
        List of label column names.

    Returns
    -------
    None
    """
    topx = f"top{n_targets}" if n_targets > 0 else "all"
    target_col_path = DATASET_DIR / data_name / activity / topx / "target_col.pkl"
    os.makedirs(target_col_path.parent, exist_ok=True)
    with open(target_col_path, "wb") as file:
        pickle.dump(label_col, file)
    logging.info(f"Tasks exported to {target_col_path}")


def export_dataset(
        subsets_dict: Dict[str, pd.DataFrame],
        files_paths: Dict[str, Path],
        cols_to_include: List[str] = None,
) -> None:
    """
    Save dataset splits to files.

    Parameters
    ----------
    subsets_dict : dict of str -> pd.DataFrame
        Dictionary of dataset splits.
    files_paths : dict of str -> Path
        File paths for each split.
    cols_to_include : list of str, optional
        Columns to include in the saved dataset.

    Returns
    -------
    None
    """
    for subset in ["train", "val", "test"]:
        save_df(subsets_dict[subset][cols_to_include], file_path=files_paths[subset])


def merge_preprocessed_desc(
        df: pd.DataFrame, preprocessed_df: pd.DataFrame, matching_col: str, desc_col: str
) -> pd.DataFrame:
    """
    Merge preprocessed descriptors into the main dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset.
    preprocessed_df : pd.DataFrame
        Preprocessed descriptor dataset.
    matching_col : str
        Column to match entries on.
    desc_col : str
        Descriptor column to merge.

    Returns
    -------
    pd.DataFrame
        Updated dataset with descriptors merged.
    """
    desc_mapper = (
        preprocessed_df[[matching_col, desc_col]]
        .set_index(matching_col)[desc_col]
        .to_dict()
    )
    df[desc_col] = df[matching_col].map(desc_mapper)
    return df


def load_desc_preprocessed(
        df: pd.DataFrame,
        files_paths: Dict[str, Path],
        desc_prot: str = None,
        desc_chem: str = None,
        prot_matching_col: str = "target_id",
        chem_matching_col: str = "SMILES",
        **kwargs,
) -> pd.DataFrame:
    """
    Load and merge preprocessed descriptors into the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset.
    files_paths : dict of str -> Path
        Paths to descriptor files.
    desc_prot : str or None, optional
        Protein descriptor column name.
    desc_chem : str or None, optional
        Chemical descriptor column name.
    prot_matching_col : str, optional
        Column name for protein matching. Default is ``"target_id"``.
    chem_matching_col : str, optional
        Column name for chemical matching. Default is ``"SMILES"``.

    Returns
    -------
    pd.DataFrame
        Dataset with descriptors merged.
    """
    dfs = []
    for f in files_paths.values():
        if not f.is_file():
            raise FileNotFoundError(f"File not found: {f}")
        dfs.append(load_df(f, **kwargs))
    file_df = pd.concat(dfs, axis=0)
    del dfs  # saving memory
    if desc_prot is not None and desc_prot not in df.columns:
        df = merge_preprocessed_desc(df, file_df, prot_matching_col, desc_prot)

    if desc_chem is not None and desc_chem not in df.columns:
        df = merge_preprocessed_desc(df, file_df, chem_matching_col, desc_chem)

    return df


def get_topx(n_targets: int) -> str:
    """
    Return a formatted string for the number of targets.

    Parameters
    ----------
    n_targets : int
        Number of targets.

    Returns
    -------
    str
        Formatted string representing the number of targets.
    """
    return f"top{n_targets}" if n_targets > 0 else "all"


def get_tasks(data_name: str, activity: str, n_targets: int) -> List[str]:
    """
    Load task labels from a pickle file.

    Parameters
    ----------
    data_name : str
        Name of the dataset.
    activity : str
        Activity type.
    n_targets : int
        Number of targets.

    Returns
    -------
    list of str
        Task labels.
    """
    topx = get_topx(n_targets)
    target_col_path = DATASET_DIR / data_name / activity / topx / "target_col.pkl"
    target_col = load_pickle(target_col_path)
    if target_col is None:
        raise RuntimeError(f"Error loading tasks from {target_col_path}")
    return target_col


def get_dataset_sizes(datasets: Dict[str, pd.DataFrame]) -> None:
    """
    Log the sizes of the dataset splits.

    Parameters
    ----------
    datasets : dict of str -> pd.DataFrame
        Dataset splits.
    """
    for name, dataset in datasets.items():
        logging.info(f"{name} set size: {len(dataset)}")


def get_data_info(
        train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute summary statistics on dataset splits.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training dataset.
    val_data : pd.DataFrame
        Validation dataset.
    test_data : pd.DataFrame
        Test dataset.

    Returns
    -------
    pd.DataFrame
        Dataframe summarizing dataset counts.
    """
    combined_data = pd.concat(
        [train_data, val_data, test_data], keys=["train", "val", "test"]
    )
    combined_data.reset_index(inplace=True)
    count_data = combined_data.groupby("level_0").count()
    count_data = count_data.pivot_table(columns="level_0")
    count_data.reset_index(inplace=True)

    return count_data


def create_split_dict(
        split_type: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create a dictionary containing dataset splits.

    Parameters
    ----------
    split_type : str
        Type of split (e.g., "random").
    train_df : pd.DataFrame
        Training dataset.
    val_df : pd.DataFrame
        Validation dataset.
    test_df : pd.DataFrame
        Test dataset.

    Returns
    -------
    dict of str -> dict of str -> pd.DataFrame
        Dictionary containing dataset splits.
    """
    out = {
        split_type: {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }
    }
    return out


def from_split_data_to_idx(
        split_dict: Dict[str, Dict[str, pd.DataFrame]]
) -> Dict[str, Dict[str, List[int]]]:
    """
    Convert dataset splits to index lists.

    Parameters
    ----------
    split_dict : dict of str -> dict of str -> pd.DataFrame
        Dataset splits.

    Returns
    -------
    dict of str -> dict of str -> list of int
        Index lists for dataset splits.
    """
    return {
        split_type: {
            "train": split_dict[split_type]["train"].index.tolist(),
            "val": split_dict[split_type]["val"].index.tolist(),
            "test": split_dict[split_type]["test"].index.tolist(),
        }
        for split_type in split_dict.keys()
    }


def random_split(
        df: pd.DataFrame,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        stratify_col: str = None,
        seed: int = 42,
        print_info: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split a DataFrame into training, validation, and test sets based on specified fractions.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be split.
    train_frac : float
        Fraction for the training set.
    val_frac : float
        Fraction for the validation set.
    test_frac : float
        Fraction for the test set.
    stratify_col : str or None, optional
        Column for stratified splitting.
    seed : int
        Random seed for reproducibility.
    print_info : bool
        Whether to print split details.

    Returns
    -------
    dict of str -> pd.DataFrame
        Dictionary containing split datasets.
    """
    st = None
    if stratify_col == "scaffold" and "scaffold" not in df.columns:
        df = merge_scaffolds(df)
        st = df[stratify_col]
    elif stratify_col:
        assert (
                stratify_col in df.columns
        ), f"Column {stratify_col} not found in the DataFrame"
        st = df[stratify_col]

    train_df, val_df = train_test_split(
        df, test_size=val_frac, train_size=train_frac, stratify=st, random_state=seed
    )
    # now we get the rest of the data
    test_df = df.drop(train_df.index).drop(val_df.index)

    if print_info:
        print(
            f"Random Split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )

    return create_split_dict("random", train_df, val_df, test_df)


def separate_min_count_df(
        df: pd.DataFrame, counting_col: str = "scaffold", threshold_count: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate dataframe into subsets based on minimum count threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    counting_col : str, optional
        Column name to count occurrences.
    threshold_count : int, optional
        Minimum count threshold.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        DataFrames containing below and above threshold counts.
    """
    counts = df[counting_col].value_counts()
    below_classes = counts[counts < threshold_count].index
    df_below_class = df[df[counting_col].isin(below_classes)]
    df_above_class = df[~df[counting_col].isin(below_classes)]

    return df_below_class, df_above_class


def random_split_stratified(
        df: pd.DataFrame,
        stratify_by: str = "scaffold",
        max_k: int = 500,
        optimal_k: Union[int, None] = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        export_path: Union[str, None] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split a dataset into training, validation, and test sets with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be split.
    stratify_by : str, optional
        The column used for stratified splitting. Default is ``"scaffold"``.
    max_k : int, optional
        Maximum number of clusters for scaffold clustering. Default is ``500``.
    optimal_k : int or None, optional
        Optimal number of clusters for scaffold clustering. Default is ``None``.
    train_frac : float, optional
        Fraction used for training. Default is ``0.7``.
    val_frac : float, optional
        Fraction used for validation. Default is ``0.15``.
    test_frac : float, optional
        Fraction used for testing. Default is ``0.15``.
    seed : int, optional
        Random seed for reproducibility. Default is ``42``.
    export_path : str or None, optional
        Path to export clustering results. Default is ``None``.

    Returns
    -------
    dict of str -> dict of str -> pd.DataFrame
        Split datasets keyed by split type.
    """
    if stratify_by == "scaffold" and "scaffold" not in df.columns:
        df = merge_scaffolds(df)
    elif stratify_by == "cluster" and "cluster" not in df.columns:
        if "scaffold" not in df.columns:
            df = merge_scaffolds(df)
        df = clustering(
            df,
            "scaffold",
            max_k=max_k,
            optimal_k=optimal_k,
            withH=False,
            export_mcs_path=export_path,
        )

    df_below, df_above = separate_min_count_df(
        df, counting_col=stratify_by, threshold_count=3
    )
    below_train, below_val, below_test = random_split(
        df_below,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        print_info=False,
    )["random"].values()
    above_train, above_val, above_test = random_split(
        df_above,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        stratify_col=stratify_by,
        seed=seed,
        print_info=False,
    )["random"].values()
    train_df, val_df, test_df = (
        pd.concat([below_train, above_train]),
        pd.concat([below_val, above_val]),
        pd.concat([below_test, above_test]),
    )
    print(
        f"Random Split Stratified By {stratify_by} - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    return create_split_dict(
        "random",
        train_df,
        val_df,
        test_df,
    )


def scaffold_split(
        df: pd.DataFrame,
        smiles_col: str = "smiles",
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split a dataset into training, validation, and test sets based on scaffold structure.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be split.
    smiles_col : str, optional
        The column containing SMILES strings. Default is ``"smiles"``.
    train_frac : float, optional
        Fraction used for training. Default is ``0.7``.
    val_frac : float, optional
        Fraction used for validation. Default is ``0.15``.
    test_frac : float, optional
        Fraction used for testing. Default is ``0.15``.
    seed : int, optional
        Random seed for reproducibility. Default is ``42``.

    Returns
    -------
    dict of str -> dict of str -> pd.DataFrame
        Split datasets keyed by split type.
    """
    # set random seed
    np.random.seed(seed)
    if "scaffold" not in df.columns:
        df = merge_scaffolds(df, smiles_col=smiles_col)

    # get unique scaffolds
    scaffolds = list(df["scaffold"].unique())

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(scaffolds)
    len_scaffolds = len(scaffolds)
    # calculate number of compounds for each split
    num_train = int(train_frac * len_scaffolds)
    num_val = int(val_frac * len_scaffolds)
    num_test = int(test_frac * len_scaffolds)
    # split scaffolds
    scaffold_train = scaffolds[:num_train]
    scaffold_val = scaffolds[num_train: num_train + num_val]
    scaffold_test = scaffolds[num_train + num_val:]

    # split dataframe
    train_df = df[df["scaffold"].isin(scaffold_train)]
    val_df = df[df["scaffold"].isin(scaffold_val)]
    test_df = df[df["scaffold"].isin(scaffold_test)]
    print(
        f"Scaffold Split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # create result dictionary for return
    return create_split_dict("scaffold", train_df, val_df, test_df)


def scaffold_cluster_split(
        df: pd.DataFrame,
        smiles_col: str = "smiles",
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        max_k: int = 500,
        optimal_k: Union[int, None] = None,
        withH: bool = False,
        export_mcs_path: Union[str, None] = None,
        seed: int = 42,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split a dataset into training, validation, and test sets using scaffold clustering.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be split.
    smiles_col : str, optional
        The column containing SMILES strings. Default is ``"smiles"``.
    train_frac : float, optional
        Fraction used for training. Default is ``0.7``.
    val_frac : float, optional
        Fraction used for validation. Default is ``0.15``.
    test_frac : float, optional
        Fraction used for testing. Default is ``0.15``.
    max_k : int, optional
        Maximum number of clusters for scaffold clustering. Default is ``500``.
    optimal_k : int or None, optional
        Optimal number of clusters for scaffold clustering. Default is ``None``.
    withH : bool, optional
        Whether to consider hydrogens in clustering. Default is ``False``.
    export_mcs_path : str or None, optional
        Path to export clustering results. Default is ``None``.
    seed : int, optional
        Random seed for reproducibility. Default is ``42``.

    Returns
    -------
    dict of str -> dict of str -> pd.DataFrame
        Split datasets keyed by split type.
    """
    # set random seed
    np.random.seed(seed)
    if "scaffold" not in df.columns:
        df = merge_scaffolds(df, smiles_col=smiles_col)

    # clustering scaffolds
    df = clustering(
        df,
        "scaffold",
        max_k=max_k,
        optimal_k=optimal_k,
        withH=withH,
        export_mcs_path=export_mcs_path,
    )
    # get unique scaffolds
    clusters = list(df["cluster"].unique())
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(clusters)
    len_clusters = len(clusters)
    # calculate number of compounds for each split
    num_train = int(train_frac * len_clusters)
    num_val = int(val_frac * len_clusters)
    num_test = int(test_frac * len_clusters)
    # split scaffolds
    clusters_train = clusters[:num_train]
    clusters_val = clusters[num_train: num_train + num_val]
    clusters_test = clusters[num_train + num_val:]

    # split dataframe
    train_df = df[df["cluster"].isin(clusters_train)]
    val_df = df[df["cluster"].isin(clusters_val)]
    test_df = df[df["cluster"].isin(clusters_test)]
    print(
        f"Scaffold Clustered Split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # create result dictionary for return
    return create_split_dict("scaffold_cluster", train_df, val_df, test_df)


def time_split(
        df: pd.DataFrame,
        time_col: str = "year",
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split a dataset into training, validation, and test sets based on a time column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be split.
    time_col : str
        The column containing the time information.
    train_frac : float
        Fraction used for training. Default is ``0.7``.
    val_frac : float
        Fraction used for validation. Default is ``0.15``.
    test_frac : float
        Fraction used for testing. Default is ``0.15``.

    Returns
    -------
    dict of str -> dict of str -> pd.DataFrame
        Split datasets keyed by split type.
    """

    # order df by time_col and split
    df = df.sort_values(by=time_col)
    train_df = df.iloc[: int(train_frac * len(df))]
    val_df = df.iloc[int(train_frac * len(df)): int((train_frac + val_frac) * len(df))]
    test_df = df.iloc[int((train_frac + val_frac) * len(df)) :]
    print(
        f"Time Split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # create result dictionary for return
    return create_split_dict("time", train_df, val_df, test_df)


def split_data(
        df: pd.DataFrame,
        split_type: Union[str, List[str]] = "random",
        smiles_col: str = "smiles",
        time_col: str = "year",
        stratify_col: Union[str, None] = None,
        fractions: Union[List[float], None] = None,
        max_k_clusters: int = 500,
        optimal_k: Union[int, None] = None,
        export_path: Union[str, None, Path] = None,
        return_indices: bool = False,
        recalculate: bool = False,
        seed: int = 42,
        logger: Union[logging.Logger, None] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split a dataset into training, validation, and test sets based on the specified split type.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    split_type : str or list of str, optional
        Type of split to use (or ``"all"``). Default is ``"random"``.
    smiles_col : str, optional
        Column containing SMILES strings. Default is ``"smiles"``.
    time_col : str, optional
        Column containing time information. Default is ``"year"``.
    stratify_col : str or None, optional
        Column used for stratified splitting. Default is ``None``.
    fractions : list of float or None, optional
        Fractions for train/val/test (sum must be 1). Default is ``[0.7, 0.15, 0.15]``.
    max_k_clusters : int, optional
        Maximum clusters for scaffold clustering. Default is ``500``.
    optimal_k : int or None, optional
        Optimal number of clusters for scaffold clustering. Default is ``None``.
    export_path : str or Path or None, optional
        Path to export clustering/split results. Default is ``None``.
    return_indices : bool, optional
        Whether to return indices instead of dataframes. Default is ``False``.
    recalculate : bool, optional
        Whether to recalculate splits if they exist. Default is ``False``.
    seed : int, optional
        Random seed for reproducibility. Default is ``42``.
    logger : logging.Logger or None, optional
        Logger instance. Default is ``None``.

    Returns
    -------
    dict of str -> dict of str -> pd.DataFrame
        Split datasets keyed by split type.
    """
    logger = logger or logging.getLogger(__name__)
    if fractions is None:
        fractions = [0.7, 0.15, 0.15]
    if sum(fractions) != 1:
        raise ValueError(
            f"The sum of train_frac, val_frac, and test_frac in arg list `fractions` must be 1 not {sum(fractions)}."
        )
    if not 2 <= len(fractions) <= 3:
        raise ValueError(
            f"Expected 2 or 3 fractions in arg list `fractions` but got {len(fractions)}."
        )

    train_frac, test_frac = fractions[0], fractions[-1]
    val_frac = fractions[1] if len(fractions) == 3 else 0.0

    if stratify_col == "scaffold" or split_type in [
        "scaffold",
        "scaffold_cluster",
        "all",
    ]:
        if "scaffold" not in df.columns:
            df = merge_scaffolds(df, smiles_col=smiles_col)

    # POSTPONED for now - only one split at a time can be done here
    all_data = {}
    split_type = (
        ["random", "time", "scaffold", "scaffold_cluster"]
        if split_type == "all"
        else split_type
    )
    if isinstance(split_type, list):
        for t in split_type:
            # Recursion here
            sub_dict = split_data(
                df,
                split_type=t,
                smiles_col=smiles_col,
                time_col=time_col,
                stratify_col=stratify_col,
                max_k_clusters=max_k_clusters,
                optimal_k=optimal_k,
                fractions=fractions,
                export_path=export_path,
                return_indices=return_indices,
                recalculate=recalculate,
                seed=seed,
                logger=logger,
            )
            all_data.update(sub_dict)

    elif split_type in ["random", "time", "scaffold", "scaffold_cluster"]:
        try:
            split_file_name = (
                f"{split_type}_split_dict{'_indices' if return_indices else ''}.pkl"
            )
            split_file_path = None
            if export_path:
                split_file_path = Path(export_path) / split_file_name
                if split_file_path.exists() and not recalculate:
                    logger.info(f"Loading splits from {split_file_path}")
                    return load_pickle(split_file_path)

                optimal_k_path = Path(export_path) / f"mcs_optimal_k.pkl"
                if optimal_k_path.exists():
                    optimal_k = optimal_k or load_pickle(optimal_k_path)

            logger.info(f"Splitting the data using {split_type} split")
            func_key = {
                "random": (
                    random_split,
                    {"stratify_col": stratify_col},
                ),
                "random_stratified": (
                    random_split_stratified,
                    {
                        "max_k": max_k_clusters,
                        "optimal_k": optimal_k,
                        "stratify_by": stratify_col,
                        "export_path": export_path,
                    },
                ),
                "scaffold": (scaffold_split, {"smiles_col": smiles_col}),
                "scaffold_cluster": (
                    scaffold_cluster_split,
                    {
                        "smiles_col": smiles_col,
                        "max_k": max_k_clusters,
                        "optimal_k": optimal_k,
                        "withH": False,
                        "export_mcs_path": export_path,
                    },
                ),
                "time": (time_split, {"time_col": time_col}),
            }

            split_type_key = (
                "random_stratified"
                if split_type == "random" and stratify_col
                else split_type
            )
            split_func, split_kwargs = func_key[split_type_key]

            sub_dict = split_func(
                df,
                **split_kwargs,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                seed=seed,
            )
            sub_dict = from_split_data_to_idx(sub_dict) if return_indices else sub_dict
            all_data.update(sub_dict)
            if export_path and split_file_path is not None:
                logger.info(f"Saving splits to {split_file_path}")
                save_pickle(sub_dict, split_file_path)

        except Exception as e:
            logging.error(f"Error splitting the data: {e}")
            logging.error(
                f"Split type: {split_type}, stratify_col: {stratify_col}, fractions: {fractions}"
            )

            all_data = create_split_dict(
                split_type, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            )
    else:
        raise ValueError(
            f"split_type must be one of 'random', 'scaffold', 'scaffold_cluster' or 'time', instead we got {split_type}"
        )

    return all_data


def stratified_distribution(
        df: pd.DataFrame, stratified_col: str = "scaffold"
) -> pd.DataFrame:
    """
    Calculate the distribution of a stratified column.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    stratified_col : str, optional
        The column to stratify by. Default is ``"scaffold"``.

    Returns
    -------
    dict
        Stratified value counts (normalized).
    """
    stratified_dist = df[stratified_col].value_counts(normalize=True).to_dict()
    return stratified_dist


def get_dist_df(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        stratified_col: str = "scaffold",
) -> pd.DataFrame:
    """
    Generate a DataFrame of stratified distributions across dataset splits.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.
    val_df : pd.DataFrame
        Validation dataset.
    test_df : pd.DataFrame
        Test dataset.
    stratified_col : str, optional
        Column used for stratification. Default is ``"scaffold"``.

    Returns
    -------
    pd.DataFrame
        Stratified distributions for each split.
    """
    train_dist = stratified_distribution(train_df, stratified_col)
    val_dist = stratified_distribution(val_df, stratified_col)
    test_dist = stratified_distribution(test_df, stratified_col)

    # Combine scaffold distributions into a DataFrame for comparison
    dist_df = pd.DataFrame(
        {"train": train_dist, "val": val_dist, "test": test_dist}
    ).fillna(0)

    return dist_df


def plot_scaffold_distribution(
        dist_df: pd.DataFrame,
        stratified_col: str = "scaffold",
        split_type: str = "random",
        output_path: Union[str, None] = None,
) -> None:
    """
    Plot the scaffold distribution across dataset splits.

    Parameters
    ----------
    dist_df : pd.DataFrame
        DataFrame containing scaffold distributions.
    stratified_col : str, optional
        Column used for stratification. Default is ``"scaffold"``.
    split_type : str, optional
        Type of dataset split. Default is ``"random"``.
    output_path : str or None, optional
        Path to save the output plot. Default is ``None``.

    Returns
    -------
    None
    """
    # Plot the scaffold distributions
    dist_df.plot(kind="bar", figsize=(24, 12))

    plt.title(
        f"{split_type.capitalize()} split - {stratified_col.capitalize()} Distribution in Train, Validation, "
        f"and Test Sets"
    )
    plt.xlabel(f"{stratified_col.capitalize()}")
    plt.ylabel(f"Fraction of {stratified_col.capitalize()}")
    # make the x axis labels more readable
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        file_path = (
                Path(output_path) / f"{split_type}_{stratified_col}_distribution.png"
        )
        plt.savefig(file_path)
    plt.show()


def check_distribution_js_similarity(dist_df: pd.DataFrame) -> bool:
    """
    Compute the Jensen-Shannon divergence to check similarity between dataset distributions.

    Parameters
    ----------
    dist_df : pd.DataFrame
        The DataFrame containing the scaffold distributions.

    Returns
    -------
    bool
        True if the distributions are similar (JS divergence below a threshold), False otherwise.
    """
    # Calculate the Jensen-Shannon divergence between the distributions
    train_val_js = jensenshannon(dist_df["train"], dist_df["val"])
    train_test_js = jensenshannon(dist_df["train"], dist_df["test"])
    val_test_js = jensenshannon(dist_df["val"], dist_df["test"])

    # Check if the JS divergences are below a threshold
    threshold = 0.1
    js_diff = (
            train_val_js < threshold
            and train_test_js < threshold
            and val_test_js < threshold
    )
    print(f"Train vs. Val JS divergence: {train_val_js:.4f}")
    print(f"Train vs. Test JS divergence: {train_test_js:.4f}")
    print(f"Val vs. Test JS divergence: {val_test_js:.4f}")

    return js_diff


def check_distribution_similarity(dist_df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Compute the mean absolute differences between train, validation, and test distributions.

    Parameters
    ----------
    dist_df : pd.DataFrame
        The DataFrame containing the scaffold distributions.

    Returns
    -------
    (float, float, float)
        Mean absolute differences between train-val, train-test, and val-test distributions.
    """
    train_val_diff = np.abs(dist_df["train"] - dist_df["val"]).mean()
    train_test_diff = np.abs(dist_df["train"] - dist_df["test"]).mean()
    val_test_diff = np.abs(dist_df["val"] - dist_df["test"]).mean()

    print(
        f"Mean absolute difference between train and val distributions: {train_val_diff:.4f}"
    )
    print(
        f"Mean absolute difference between train and test distributions: {train_test_diff:.4f}"
    )
    print(
        f"Mean absolute difference between val and test distributions: {val_test_diff:.4f}"
    )

    return train_val_diff, train_test_diff, val_test_diff


def check_distribution(
        split_dict: Dict[str, Dict[str, pd.DataFrame]],
        stratified_col: str = "scaffold",
        output_path: Union[str, None] = None,
) -> None:
    """
    Evaluate the scaffold distribution across train, validation, and test splits.

    Parameters
    ----------
    split_dict : dict of str -> dict of str -> pd.DataFrame
        Dataset splits.
    stratified_col : str, optional
        Column used for stratification. Default is ``"scaffold"``.
    output_path : str or None, optional
        Path to save the output plot. Default is ``None``.

    Returns
    -------
    None
    """
    split_types = list(split_dict.keys())

    for split_type in split_types:
        train_df = split_dict[split_type]["train"]
        val_df = split_dict[split_type]["val"]
        test_df = split_dict[split_type]["test"]
        dist_df = get_dist_df(train_df, val_df, test_df, stratified_col)

        plot_scaffold_distribution(dist_df, stratified_col, split_type, output_path)

        js_diff = check_distribution_js_similarity(dist_df)
        print(
            f"JS divergence between scaffold distributions in {split_type} split: {js_diff}"
        )
        check_distribution_similarity(dist_df)
        print("")


def check_if_processed_file(
        data_name: str = "papyrus",
        activity_type: str = "xc50",
        n_targets: int = -1,
        split_type: str = "random",
        desc_prot: Union[str, None] = None,
        desc_chem: str = "ecfp1024",
        file_ext: str = "pkl",
) -> Tuple[bool, Dict[str, Path]]:
    """
    Check whether dataset files for a specific split and descriptor combination already exist.

    Parameters
    ----------
    data_name : str, optional
        Name of the dataset. Default is ``"papyrus"``.
    activity_type : str, optional
        Type of activity data. Default is ``"xc50"``.
    n_targets : int, optional
        Number of targets. Default is ``-1`` for all targets.
    split_type : str, optional
        Type of data split. Default is ``"random"``.
    desc_prot : str or None, optional
        Protein descriptor type.
    desc_chem : str, optional
        Chemical descriptor type. Default is ``"ecfp1024"``.
    file_ext : str, optional
        File extension for saved data. Default is ``"pkl"``.

    Returns
    -------
    (bool, dict of str -> Path)
        Boolean indicating whether all files exist and a dictionary of expected file paths.
    """
    topx = f"top{n_targets}" if n_targets > 0 else "all"
    output_path = DATASET_DIR / data_name / activity_type / topx
    prefix = (
        f"{split_type}_{desc_prot}_{desc_chem}"
        if desc_prot
        else f"{split_type}_{desc_chem}"
    )

    files_paths = {
        subset: output_path / f"{prefix}_{subset}.{file_ext}"
        for subset in ["train", "val", "test"]
    }

    files_exist = all(Path(file_p).exists() for file_p in files_paths.values())

    return files_exist, files_paths


def apply_label_scaling(
        df: pd.DataFrame,
        label_col: Union[str, List[str]],
        label_scaling_func: Union[Callable, None] = None,
) -> pd.DataFrame:
    """
    Apply a scaling function to the label column(s) of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    label_col : str or list of str
        Column(s) to be scaled.
    label_scaling_func : callable or None, optional
        Scaling function applied to the label column(s).

    Returns
    -------
    pd.DataFrame
        The DataFrame with the label column(s) scaled.
    """
    if label_scaling_func is not None:
        if isinstance(label_col, list):
            for col in label_col:
                df[col] = label_scaling_func(df[col])
        else:
            df[label_col] = label_scaling_func(df[label_col])
    return df


def apply_median_scaling(
        df: pd.DataFrame,
        label_col: Union[str, List[str]],
        train_median: Union[float, List[float]] = 6.0,
        calc_median: bool = False,
        median_scaling: bool = False,
        logger: Union[logging.Logger, None] = None,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Apply median scaling to label columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    label_col : str or list of str
        Column(s) to apply median scaling.
    train_median : float or list of float, optional
        Median value(s) to subtract. Default is ``6.0``.
    calc_median : bool, optional
        If True, calculate median from the dataset. Default is ``False``.
    median_scaling : bool, optional
        If True, apply median scaling. Default is ``False``.
    logger : logging.Logger or None, optional
        Logger for logging messages.

    Returns
    -------
    (pd.DataFrame, list of float)
        Modified DataFrame and the computed median values.
    """
    if isinstance(train_median, float):
        train_median = [train_median]
    if isinstance(label_col, str):
        label_col = [label_col]

    if calc_median:
        train_median = df[label_col].median().tolist()

    if median_scaling:
        if logger:
            logger.info(f"Applying median scaling to label columns: {label_col}")
        df[label_col] = df[label_col] - train_median
    return df, train_median


def subtract_label_median(
        df_label_series: pd.Series, median: Union[float, None] = None
) -> Tuple[pd.Series, float]:
    """
    Subtract the median value from a label series.

    Parameters
    ----------
    df_label_series : pd.Series
        The label series.
    median : float or None, optional
        Median value to subtract. If ``None``, it is computed from the series.

    Returns
    -------
    (pd.Series, float)
        Modified series and used median value.
    """
    if not median:
        median = df_label_series.median()
    print(f"Subtracting median {median} from label series")
    return df_label_series - median, median


def get_label_scaling_func(scaling_type: str, **kwargs) -> Union[Callable, None]:
    """
    Return the appropriate label scaling function based on the specified type.

    Parameters
    ----------
    scaling_type : str
        The type of scaling to apply (e.g., 'median', 'standard', 'minmax').
    **kwargs : dict
        Additional arguments for the scaling function.

    Returns
    -------
    callable or None
        The selected scaling function.
    """
    if scaling_type == "median":
        return subtract_label_median

    elif scaling_type == "standard":
        from sklearn.preprocessing import StandardScaler

        return StandardScaler(**kwargs).fit
    elif scaling_type == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        return MinMaxScaler(**kwargs).fit
    elif scaling_type == "robust":
        from sklearn.preprocessing import RobustScaler

        return RobustScaler(**kwargs).fit
    elif scaling_type == "log":
        return np.log
    elif scaling_type == "log1p":
        return np.log1p
    elif scaling_type == "log2":
        return np.log2
    elif scaling_type == "log10":
        return np.log10
    elif scaling_type == "none":
        return None
    else:
        raise ValueError(f"Unsupported scaling type: {scaling_type}")


def check_normality(series: pd.Series) -> Tuple[float, float, bool]:
    """
    Check normality of a data series using the Shapiro-Wilk test.

    Parameters
    ----------
    series : pd.Series
        The data series to test.

    Returns
    -------
    (float, float, bool)
        Test statistic, p-value, and normality assumption result.
    """
    from scipy.stats import shapiro

    stat, p = shapiro(series)
    alpha = 0.05
    return stat, p, p > alpha


def target_filtering(
        df: pd.DataFrame,
        target_col: str = "target_id",
        label_col: str = "pchembl_value_Mean",
        min_datapoints: int = 50,
        min_actives: int = 10,
        activity_threshold: float = 6.5,
        normal: bool = False,
) -> pd.DataFrame:
    """
    Filter the dataset based on the number of datapoints and active compounds.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    target_col : str, optional
        Column containing the target IDs. Default is ``"target_id"``.
    label_col : str, optional
        Column containing the labels. Default is ``"pchembl_value_Mean"``.
    min_datapoints : int, optional
        Minimum number of datapoints required for a target. Default is ``50``.
    min_actives : int, optional
        Minimum number of active compounds required for a target. Default is ``10``.
    activity_threshold : float, optional
        Activity threshold to use for filtering. Default is ``6.5``.
    normal : bool, optional
        If True, assumes labels are normally distributed and applies normality check.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with targets meeting the criteria.
    """
    # Filter targets based on the number of datapoints and active compounds
    target_counts = df[target_col].value_counts()
    active_counts = df[df[label_col] >= activity_threshold][target_col].value_counts()
    target_counts = target_counts.to_frame(name="datapoints")
    active_counts = active_counts.to_frame(name="actives")
    target_counts = target_counts.join(active_counts, how="left").fillna(0)
    target_counts["fraction_actives"] = (
            target_counts["actives"] / target_counts["datapoints"]
    )
    target_counts = target_counts[
        (target_counts["datapoints"] >= min_datapoints)
        & (target_counts["actives"] >= min_actives)
        ]

    # check normality
    if normal:
        normality_results = (
            df.groupby(target_col)
            .filter(lambda x: len(x) >= 3)
            .groupby(target_col)
            .apply(
                lambda x: pd.Series(
                    check_normality(x[label_col]), index=["stat", "p", "normal"]
                )
            )
        )
        target_counts = target_counts.join(normality_results, how="inner")
        target_counts = target_counts[target_counts["normal"] == True]

    # Filter the dataset based on the target IDs
    filtered_df = df[df[target_col].isin(target_counts.index)].reset_index(drop=True)

    return filtered_df


def check_homoscedasticity(y_true: pd.Series, y_pred: pd.Series) -> bool:
    """
    Check homoscedasticity using Bartlett's test.

    Parameters
    ----------
    y_true : pd.Series
        The true values.
    y_pred : pd.Series
        The predicted values.

    Returns
    -------
    bool
        True if homoscedasticity assumption holds, False otherwise.
    """
    from scipy.stats import bartlett  # , levene

    stat, p = bartlett(y_true, y_pred)
    alpha = 0.05
    if p > alpha:
        return True
    else:
        return False


def get_target_data_distribution(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Compute the count, mean, and standard deviation of labels per target.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    target_col : str
        Column containing the target IDs.

    Returns
    -------
    pd.DataFrame
        Dataframe with target ID, count, mean, and standard deviation of labels.
    """
    distribution = df[target_col].value_counts()
    distribution = distribution.reset_index()

    # calculate mean and std of labels per each target and add it to col of distribution
    mean_std = df.groupby(target_col).agg(["mean", "std"]).reset_index()
    distribution = distribution.merge(mean_std, on=target_col)

    distribution.columns = ["target_id", "count", "mean", "std"]

    # calculate mean of means and mean of stds
    mean_mean = distribution["mean"].mean()
    mean_std = distribution["std"].mean()

    print(f"Mean of means: {mean_mean:.2f}, Mean of stds: {mean_std:.2f}")

    return distribution


def fig_target_data_distribution(
        df: pd.DataFrame, target_col: str, output_path: Union[str, None] = None
) -> None:
    """
    Plot the distribution of data points per target.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    target_col : str
        Column containing the target IDs.
    output_path : str or None, optional
        Path to save the output plot.

    Returns
    -------
    None
    """
    distribution = get_target_data_distribution(df, target_col)
    plt.figure(figsize=(10, 6))
    sns.histplot(distribution["count"], bins=100, kde=True)
    plt.title("Target Data Distribution")
    plt.xlabel("Number of data points per target")
    plt.ylabel("Number of targets")
    plt.tight_layout()

    if output_path:
        file_path = Path(output_path) / "target_data_distribution.png"
        plt.savefig(file_path)

    plt.show()


def fig_label_distribution(
        df: pd.DataFrame,
        label_col: str = "pchembl_value_Mean",
        output_path: Union[str, None] = None,
) -> None:
    """
    Plot the distribution of label values.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    label_col : str, optional
        Column containing the labels. Default is ``"pchembl_value_Mean"``.
    output_path : str or None, optional
        Path to save the output plot.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[label_col], bins=100, kde=True)
    plt.title(f"{label_col} Distribution")
    plt.xlabel(f"{label_col} values")
    plt.ylabel("Number of data points")
    plt.tight_layout()

    if output_path:
        file_path = Path(output_path) / "label_distribution.png"
        plt.savefig(file_path)


def fig_label_distribution_across_splits(
        split_dict: Dict[str, Dict[str, pd.DataFrame]],
        label_col: str = "pchembl_value_Mean",
        output_path: Union[str, None] = None,
) -> None:
    """
    Plot the label distribution across dataset splits.

    Parameters
    ----------
    split_dict : dict of str -> dict of str -> pd.DataFrame
        Dictionary containing dataset splits.
    label_col : str or list of str, optional
        Column containing the labels. If a list, the first element is used.
    output_path : str or None, optional
        Path to save the output plot.

    Returns
    -------
    None
    """
    if isinstance(label_col, list):
        label_col = label_col[0]
    split_types = list(split_dict.keys())
    for split_type in split_types:
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        for i, subset in enumerate(["train", "val", "test"]):
            df = split_dict[split_type][subset]
            sns.histplot(df[label_col], bins=100, kde=True, ax=axes[i])
            axes[i].set_title(
                f"{split_type.capitalize()} Split - {subset.capitalize()} - {label_col} Distribution"
            )
            axes[i].set_xlabel(f"{label_col} values")
            axes[i].set_ylabel("Number of data points")

        plt.tight_layout()
        if output_path:
            file_path = (
                    Path(output_path)
                    / f"{split_type}_split_label_distribution_across_splits.png"
            )
            plt.savefig(file_path)
