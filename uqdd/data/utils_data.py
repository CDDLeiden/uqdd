import os
import pickle
import logging
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from uqdd import DATASET_DIR, DEVICE
from uqdd.utils_chem import generate_scaffold
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

string_types = (type(b""), type(""))


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


def get_tasks(data_name, activity, n_targets):
    topx = f"top{n_targets}" if n_targets > 0 else "all"
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
    split_type: str = "random",
    smiles_col: str = "smiles",
    time_col: str = "year",
    fractions: Union[List[float], None] = None,
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
        The type of split to use. Options are 'random', 'scaffold', and 'time'. Default is 'random'.
    smiles_col : str, optional
        The name of the column containing the SMILES strings. Default is 'smiles'.
    time_col : str, optional
        The name of the column containing the time information. Default is 'year'.
    fractions : list, optional
        A list of fractions to use for the training, validation, and test sets. Default is [0.7, 0.15, 0.15].
        The sum of the fractions must be 1. If only two fractions are provided, the second fraction is used for the test set.
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
        "time": (time_split, {"time_col": time_col}),
    }

    # POSTPONED for now - only one split at a time can be done here

    all_data = {}
    if split_type == "all":
        for t in ["random", "scaffold", "time"]:
            # Recursion here
            sub_dict = split_data(
                df,
                split_type=t,
                smiles_col=smiles_col,
                time_col=time_col,
                fractions=fractions,
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

    files_exist = all(Path(file_p).exists() for file_p in files_paths)

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
    if label_scaling_func is not None:
        if isinstance(label_col, list):
            for col in label_col:
                df[col] = df[col].apply(label_scaling_func)
        else:
            df[label_col] = df[label_col].apply(label_scaling_func)
    return df


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
