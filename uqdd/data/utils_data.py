import os
import pickle
import logging
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from uqdd import DATASET_DIR, DEVICE, FIGS_DIR, TODAY
from uqdd.utils_chem import generate_scaffold, clustering
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

string_types = (type(b""), type(""))


def export_pickle(data, file_path):
    """Helper function to export a pickle file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(filepath):
    """Helper function to load a pickle file."""
    try:
        with open(filepath, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
    except Exception as e:
        logging.error(f"Error loading file {filepath}: {e}")


def export_df(
    df, file_path=None, output_path="./", filename="exported_file", ext="csv", **kwargs
):
    """
    Exports a DataFrame to a specified file format.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be exported.
    file_path : str or Path, optional
        The full file path for export.
        If provided, output_path, filename, and ext are ignored.
        Default is None.
    output_path : str, optional
        The path to the output directory. Default is the current directory.
    filename : str, optional
        The name of the output file. Default is 'exported_file'.
    ext : str, optional
        The file extension to use. Default is 'csv'.
    """
    if df.empty:
        return logging.warning("DataFrame is empty. Nothing to export.")

    if file_path:
        path_obj = Path(file_path)
        output_path = path_obj.parent
        filename = path_obj.stem
        ext = path_obj.suffix.lstrip(".")
    else:
        if not filename.endswith(ext):
            filename = f"{filename}.{ext}"
        file_path = os.path.join(output_path, filename)

    os.makedirs(output_path, exist_ok=True)

    if not filename.endswith(ext):
        filename = f"{filename}.{ext}"
    if ext == "csv":
        df.to_csv(file_path, index=False, **kwargs)
    elif ext == "parquet":
        df.to_parquet(file_path, index=False, **kwargs)
    elif ext == "feather":
        df.to_feather(file_path, **kwargs)
    elif ext == "pkl":
        df.to_pickle(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    logging.info(f"DataFrame exported to {file_path}")


def load_df(input_path, **kwargs):
    """
    Loads a DataFrame from a specified file format.

    Parameters
    ----------
    input_path : str or Path
        The path to the input file.
    """
    # we want to get the extension from the Path object
    if isinstance(input_path, string_types):
        input_path = Path(input_path)
    ext = input_path.suffix.lstrip(".")
    if ext == "csv":
        return pd.read_csv(input_path, **kwargs)
    elif ext == "parquet":
        return pd.read_parquet(input_path, **kwargs)
    elif ext == "pkl":
        return pd.read_pickle(input_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension for loading: {ext} provided")


def export_tasks(data_name, activity, n_targets, label_col):
    topx = f"top{n_targets}" if n_targets > 0 else "all"
    target_col_path = DATASET_DIR / data_name / activity / topx / "target_col.pkl"
    os.makedirs(target_col_path.parent, exist_ok=True)
    # export to pickle
    with open(target_col_path, "wb") as file:
        pickle.dump(label_col, file)
    logging.info(f"Tasks exported to {target_col_path}")


def export_dataset(subsets_dict, files_paths, cols_to_include=None):
    for subset in ["train", "val", "test"]:
        export_df(subsets_dict[subset][cols_to_include], file_path=files_paths[subset])


def merge_preprocessed_desc(df, preprocessed_df, matching_col, desc_col):
    desc_mapper = (
        preprocessed_df[[matching_col, desc_col]]
        .set_index(matching_col)[desc_col]
        .to_dict()
    )
    df[desc_col] = df[matching_col].map(desc_mapper)
    return df


def load_desc_preprocessed(
    df,
    files_paths,
    desc_prot=None,
    desc_chem=None,
    prot_matching_col="target_id",
    chem_matching_col="SMILES",
    **kwargs,
):
    dfs = []
    for f in files_paths.values():
        if not f.is_file():
            raise FileNotFoundError(f"File not found: {f}")
        dfs.append(load_df(f, **kwargs))
    file_df = pd.concat(dfs, axis=0)
    del dfs  # saving memory
    if desc_prot is not None and desc_prot not in df.columns:
        df = merge_preprocessed_desc(df, file_df, prot_matching_col, desc_prot)
        # # we create a dictionary mapper here for unique desc_prot values
        # protein_descriptors = protein_descriptors[["target_id", desc_type]]
        # protein_descriptors_mapper = protein_descriptors.set_index("target_id")[
        #     desc_type
        # ].to_dict()
        # df = df.merge(
        #     file_df[[prot_matching_col, desc_prot]],
        #     left_on=prot_matching_col,
        #     right_on=prot_matching_col,
        #     how="left",
        # )

    if desc_chem is not None and desc_chem not in df.columns:
        df = merge_preprocessed_desc(df, file_df, chem_matching_col, desc_chem)
        # df = df.merge(
        #     file_df[[chem_matching_col, desc_chem]],
        #     left_on=chem_matching_col,
        #     right_on=chem_matching_col,
        #     how="left",
        # )

    return df


def get_topx(n_targets):
    return f"top{n_targets}" if n_targets > 0 else "all"


def get_tasks(data_name, activity, n_targets):
    topx = get_topx(n_targets)
    target_col_path = DATASET_DIR / data_name / activity / topx / "target_col.pkl"
    target_col = load_pickle(target_col_path)
    if target_col is None:
        raise RuntimeError(f"Error loading tasks from {target_col_path}")
    return target_col


def get_dataset_sizes(datasets):
    """
    Logs the sizes of the datasets.
    """
    for name, dataset in datasets.items():
        logging.info(f"{name} set size: {len(dataset)}")


def get_data_info(train_data, val_data, test_data):
    combined_data = pd.concat(
        [train_data, val_data, test_data], keys=["train", "val", "test"]
    )
    combined_data.reset_index(inplace=True)
    count_data = combined_data.groupby("level_0").count()
    count_data = count_data.pivot_table(columns="level_0")
    count_data.reset_index(inplace=True)

    return count_data


def create_split_dict(split_type, train_df, val_df, test_df):  # , **kwargs):
    out = {
        split_type: {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }
    }
    return out


def random_split(
    df: pd.DataFrame,
    train_frac=0.7,
    val_frac=0.15,
    test_frac=0.15,
    return_indices=False,
    seed=42,
) -> dict:
    """
    Splits a DataFrame into training, validation, and test sets based on specified fractions.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be split.
    train_frac : float
        The fraction of the dataset to be used as the training set.
    val_frac : float
        The fraction of the dataset to be used as the validation set.
    test_frac : float
        The fraction of the dataset to be used as the test set.
    seed : int
        The random seed for reproducible splits.
    Returns:
    --------
    train_df : pandas.DataFrame or list
        The training dataframe or list of indices.
    val_df : pandas.DataFrame or list
        The validation dataframe or list of indices.
    test_df : pandas.DataFrame or list
        The testing dataframe or list of indices.
    """
    # First split: separate the training set from the rest
    rest_frac = val_frac + test_frac
    train_df, temp_df = train_test_split(df, test_size=rest_frac, random_state=seed)

    # Adjust the proportion for the second split
    # The proportion of validation set out of the rest (val + test)
    adjusted_val_frac = val_frac / rest_frac

    # Second split: separate the validation set from the test set
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - adjusted_val_frac, random_state=seed
    )
    if return_indices:
        return create_split_dict(
            "random",
            train_df.index.tolist(),
            val_df.index.tolist(),
            test_df.index.tolist(),
        )
    return create_split_dict("random", train_df, val_df, test_df)


def scaffold_split(
    df,
    smiles_col="smiles",
    train_frac=0.7,
    val_frac=0.15,
    test_frac=0.15,
    return_indices=False,
    seed=42,
) -> dict:
    """
    Splits dataframe into scaffold splits.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    smiles_col : str, optional
        The name of the column containing the SMILES strings. Default is 'smiles'.
    train_frac : float, optional
        The fraction of the data to use for training. Default is 0.7.
    val_frac : float, optional
        The fraction of the data to use for validation. Default is 0.15.
    test_frac : float, optional
        The fraction of the data to use for testing. Default is 0.15.
    return_indices : bool, optional
        If True, returns the indices of the split dataframes instead of the dataframes themselves.
    seed : int, optional
        The random seed to use for splitting the data. Default is 42.

    Returns
    -------
    train_df : pandas.DataFrame or list
        The training dataframe or list of indices.
    val_df : pandas.DataFrame or list
        The validation dataframe or list of indices.
    test_df : pandas.DataFrame or list
        The testing dataframe or list of indices.
    """
    # set random seed
    np.random.seed(seed)

    # calculate scaffolds for each smiles string # concurrent.futures.ProcessPoolExecutor
    unique_smiles = df[smiles_col].unique().tolist()

    with ProcessPoolExecutor() as executor:
        scaffolds = list(
            tqdm(
                executor.map(generate_scaffold, unique_smiles),
                total=len(unique_smiles),
                desc="Generating scaffolds",
            )
        )

    smi_sc_mapper = {smi: scaffold for smi, scaffold in zip(unique_smiles, scaffolds)}
    df["scaffold"] = df[smiles_col].map(smi_sc_mapper)

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
    scaffold_val = scaffolds[num_train : num_train + num_val]
    scaffold_test = scaffolds[num_train + num_val :]

    # split dataframe
    train_df = df[df["scaffold"].isin(scaffold_train)]
    val_df = df[df["scaffold"].isin(scaffold_val)]
    test_df = df[df["scaffold"].isin(scaffold_test)]

    if return_indices:
        return create_split_dict(
            "scaffold",
            train_df.index.tolist(),
            val_df.index.tolist(),
            test_df.index.tolist(),
        )
    # create result dictionary for return
    return create_split_dict("scaffold", train_df, val_df, test_df)


def scaffold_cluster_split(
    df,
    smiles_col="smiles",
    train_frac=0.7,
    val_frac=0.15,
    test_frac=0.15,
    max_k=100,
    batch_size=10000,
    withH=False,
    fig_output_path=None,
    export_mcs_path=None,
    return_indices=False,
    seed=42,
) -> dict:
    # set random seed
    np.random.seed(seed)

    # calculate scaffolds for each smiles string # concurrent.futures.ProcessPoolExecutor
    unique_smiles = df[smiles_col].unique().tolist()

    with ProcessPoolExecutor() as executor:
        scaffolds = list(
            tqdm(
                executor.map(generate_scaffold, unique_smiles),
                total=len(unique_smiles),
                desc="Generating scaffolds",
            )
        )

    smi_sc_mapper = {smi: scaffold for smi, scaffold in zip(unique_smiles, scaffolds)}
    df["scaffold"] = df[smiles_col].map(smi_sc_mapper)

    # clustering scaffolds
    df = clustering(
        df,
        "scaffold",
        max_k=max_k,
        withH=withH,
        fig_output_path=fig_output_path or FIGS_DIR / f"{TODAY}_scaffold_clustering/",
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
    clusters_val = clusters[num_train : num_train + num_val]
    clusters_test = clusters[num_train + num_val :]

    # split dataframe
    train_df = df[df["cluster"].isin(clusters_train)]
    val_df = df[df["cluster"].isin(clusters_val)]
    test_df = df[df["cluster"].isin(clusters_test)]
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if return_indices:
        return create_split_dict(
            "scaffold_cluster",
            train_df.index.tolist(),
            val_df.index.tolist(),
            test_df.index.tolist(),
        )
    # create result dictionary for return
    return create_split_dict("scaffold_cluster", train_df, val_df, test_df)


def time_split(
    df,
    time_col="year",
    train_frac=0.7,
    val_frac=0.15,
    test_frac=0.15,
    return_indices=False,
    **kwargs,
):
    """
    Splits a DataFrame into training, validation, and test sets based on a time column.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be split.
    time_col: str
        The name of the column containing the time information.
    train_frac: float
        The fraction of the dataset to be used as the training set.
    val_frac: float
        The fraction of the dataset to be used as the validation set.
    test_frac: float
        The fraction of the dataset to be used as the test set.
    return_indices: bool
        If True, returns the indices of the split dataframes instead of the dataframes themselves.

    Returns
    -------
    train_df : pandas.DataFrame or list
        The training dataframe or list of indices.
    val_df : pandas.DataFrame or list
        The validation dataframe or list of indices.
    test_df : pandas.DataFrame or list
        The testing dataframe or list of indices.
    """

    # order df by time_col and split
    df = df.sort_values(by=time_col)
    train_df = df.iloc[: int(train_frac * len(df))]
    val_df = df.iloc[int(train_frac * len(df)) : int((train_frac + val_frac) * len(df))]
    test_df = df.iloc[int((train_frac + val_frac) * len(df)) :]
    if return_indices:
        return create_split_dict(
            "time",
            train_df.index.tolist(),
            val_df.index.tolist(),
            test_df.index.tolist(),
        )
    # create result dictionary for return
    return create_split_dict("time", train_df, val_df, test_df)


def split_data(
    df: pd.DataFrame,
    split_type: Union[str, List[str]] = "random",
    smiles_col: str = "smiles",
    time_col: str = "year",
    fractions: Union[List[float], None] = None,
    max_k_clusters: int = 100,
    fig_output_path: Union[str, None] = None,
    export_mcs_path: Union[str, None] = None,
    return_indices: bool = False,
    seed: int = 42,
) -> dict:
    """
    Splits a DataFrame into training, validation, and test sets based on the specified split type.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    split_type : str, optional
        The type of split to use. Options are 'random', 'scaffold', 'scaffold_cluster' and 'time'. Default is 'random'.
    smiles_col : str, optional
        The name of the column containing the SMILES strings. Default is 'smiles'.
    time_col : str, optional
        The name of the column containing the time information. Default is 'year'.
    max_k_clusters : int, optional
        The maximum number of clusters to use for scaffold clustering. Default is 100.
    fractions : list, optional
        A list of fractions to use for the training, validation, and test sets. Default is [0.7, 0.15, 0.15].
        The sum of the fractions must be 1. If only two fractions are provided, the second fraction is used for the test set.
    fig_output_path : str, optional
        The path to the output directory for scaffold clustering figures. Default is None.
    export_mcs_path : str, optional
        The path to the output directory for exporting the maximum common substructures. Default is None.
    return_indices : bool, optional
        If True, returns the indices of the split dataframes instead of the dataframes themselves. Default is False.
    seed : int, optional
        The random seed to use for splitting the data. Default is 42.


    Returns
    -------
    dict
        A dictionary with key as split_type and value is another dict containing the training, validation, and test dataframes.
    """
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

    func_key = {
        "random": (random_split, {}),
        "scaffold": (scaffold_split, {"smiles_col": smiles_col}),
        "scaffold_cluster": (
            scaffold_cluster_split,
            {
                "smiles_col": smiles_col,
                "max_k": max_k_clusters,
                "withH": False,
                "fig_output_path": fig_output_path,
                "export_mcs_path": export_mcs_path,
            },
        ),
        "time": (time_split, {"time_col": time_col}),
    }

    # POSTPONED for now - only one split at a time can be done here
    all_data = {}
    split_type = (
        ["random", "scaffold", "scaffold_cluster", "time"]
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
                max_k_clusters=max_k_clusters,
                fractions=fractions,
                fig_output_path=fig_output_path,
                export_mcs_path=export_mcs_path,
                return_indices=return_indices,
                seed=seed,
            )
            all_data.update(sub_dict)

    elif split_type in func_key.keys():
        try:
            split_func, split_kwargs = func_key[split_type]
            sub_dict = split_func(
                df,
                **split_kwargs,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                return_indices=return_indices,
                seed=seed,
            )
            all_data.update(sub_dict)
        except Exception as e:
            logging.error(f"Error splitting the data: {e}")
            all_data = create_split_dict(
                split_type, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            )
    else:
        raise ValueError("split_type must be one of 'random', 'scaffold', or 'time'.")

    return all_data


def check_if_processed_file(
    data_name="papyrus",
    activity_type="xc50",
    n_targets=-1,
    split_type="random",
    desc_prot=None,
    desc_chem="ecfp1024",
    file_ext="pkl",
):

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


def apply_label_scaling(df, label_col, label_scaling_func=None):
    """
    Applies a scaling function to the label column(s) of a DataFrame.
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    label_col: str or list
        The name of the column(s) to be scaled.
    label_scaling_func: function, optional
        The scaling function to be applied to the label column(s). Default is None.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with the label column(s) scaled.
    """
    # if label_scaling_func == subtract_label_median:
    #     kwargs = {"median": median}
    #
    # else:
    #     kwargs = {}

    if label_scaling_func is not None:
        if isinstance(label_col, list):
            for col in label_col:
                df[col] = label_scaling_func(df[col])
        else:
            df[label_col] = label_scaling_func(df[label_col])
            # df[label_col] = df[label_col].apply(label_scaling_func)
    return df


def apply_median_scaling(
    df,
    label_col,
    train_median=6.0,
    calc_median=False,
    median_scaling=False,
    logger=None,
):
    if isinstance(train_median, float):
        train_median = [train_median]
    if isinstance(label_col, str):
        label_col = [label_col]

    if calc_median:
        train_median = df[label_col].median().tolist()

    if median_scaling:
        logger.info(f"Applying median scaling to label columns: {label_col}")
        # for col, median in zip(label_col, train_median):
        df[label_col] = df[label_col] - train_median
    return df, train_median


def subtract_label_median(df_label_series, median=None):
    if not median:
        median = df_label_series.median()
    print(f"Subtracting median {median} from label series")
    return df_label_series - median, median


# TODO : check this func and add it to get_datasets
def get_label_scaling_func(scaling_type, **kwargs):
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


def check_normality(series):
    from scipy.stats import shapiro

    stat, p = shapiro(series)
    alpha = 0.05
    return stat, p, p > alpha
    # alpha = 0.05
    # if p > alpha:
    #     return True
    # else:
    #     return False


def target_filtering(
    df,
    target_col="target_id",
    label_col="pchembl_value_Mean",
    min_datapoints=50,
    min_actives=10,
    activity_threshold=6.5,
    normal=True,
):
    """
    Filters the dataset based on the number of datapoints and active compounds.
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    target_col : str, optional
        The name of the column containing the target IDs. Default is 'target_id'.
    label_col : str, optional
        The name of the column containing the labels. Default is 'pchembl_value_Mean'.
    min_datapoints : int, optional
        The minimum number of datapoints required for a target. Default is 50.
    min_actives : int, optional
        The minimum number of active compounds required for a target. Default is 10.
    activity_threshold : float, optional
        The activity threshold to use for filtering. Default is 6.5.
    normal : bool, optional
        If True, the labels are assumed to be normally distributed. Default is True.
    Returns
    -------
    pandas.DataFrame
        The filtered DataFrame.
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

        # target_counts[["stat", "p", "normal"]] = target_counts.index.map(
        #     lambda x: check_normality(df[df[target_col] == x][label_col])
        # )
        # target_counts = target_counts[target_counts["normal"] == True]

    # Filter the dataset based on the target IDs
    filtered_df = df[df[target_col].isin(target_counts.index)].reset_index(drop=True)

    return filtered_df


def check_homoscedasticity(y_true, y_pred):
    from scipy.stats import bartlett, levene

    stat, p = bartlett(y_true, y_pred)
    alpha = 0.05
    if p > alpha:
        return True
    else:
        return False


# def check_multicollinearity(df, threshold=0.9):
#     """
#     Check for multicollinearity in the features of a DataFrame.
#     Parameters
#     ----------
#     df : pandas.DataFrame
#         The input DataFrame.
#     threshold : float, optional
#         The threshold for the VIF score. Default is 0.9.
#     Returns
#     -------
#     bool
#         True if multicollinearity is present, False otherwise.
#     """
#     from statsmodels.stats.outliers_influence import variance_inflation_factor
#
#     vif = pd.DataFrame()
#     vif["features"] = df.columns
#     vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
#     return vif[vif["VIF"] > threshold].shape[0] > 0


# deprecated
# def _check_if_processed_file(
#     data_name="papyrus",
#     activity_type="xc50",
#     n_targets=-1,
#     split_type="random",
#     desc_prot=None,
#     descriptor_chemical="ecfp1024",
#     file_ext="pkl",
# ):
#
#     topx = f"top{n_targets}" if n_targets > 0 else "all"
#     # if split_type == "all":
#     #     all_files_exist = []
#     #     all_file_paths = {}
#     #     for st in ["random", "scaffold", "time"]:
#     #         files_exist, files_paths = check_if_processed_file(
#     #             data_name, activity_type, n_targets, st, desc_prot, descriptor_chemical, file_ext
#     #         )
#     #
#     #         all_files_exist.append(files_exist)
#     #         all_file_paths.update(files_paths)
#     #     return True, []
#     output_path = DATASET_DIR / data_name / activity_type / topx / split_type
#
#     desc = f"{desc_prot}_{descriptor_chemical}" if desc_prot else descriptor_chemical
#
#     results = {
#         split_type: {
#             subset: output_path / f"{desc}{'_' if desc else ''}{subset}.{file_ext}"
#             for subset in ["train", "val", "test"]
#         }
#     }
#     results[split_type]["all_exists"] = all(file_p.exists() for file_p in results[split_type].values())
#
#     return results
