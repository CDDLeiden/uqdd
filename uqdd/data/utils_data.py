import os
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from uqdd import DATASET_DIR, DEVICE
from uqdd.utils_chem import generate_scaffold

# from uqdd.data.data_papyrus import PapyrusDatasetMT

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


def export_df(df, output_path="./", filename="exported_file", ext="csv", **kwargs):
    """
    Exports a DataFrame to a specified file format.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be exported.
    output_path : str, optional
        The path to the output directory. Default is the current directory.
    filename : str, optional
        The name of the output file. Default is 'exported_file'.
    ext : str, optional
        The file extension to use. Default is 'csv'.
    """
    os.makedirs(output_path, exist_ok=True)

    if not filename.endswith(ext):
        filename = f"{filename}.{ext}"
    output_file_path = os.path.join(output_path, filename)
    if ext == "csv":
        df.to_csv(output_file_path, index=False, **kwargs)
    elif ext == "parquet":
        df.to_parquet(output_file_path, index=False, **kwargs)
    elif ext == "feather":
        df.to_feather(output_file_path, index=False, **kwargs)
    elif ext == "pkl":
        df.to_pickle(output_file_path, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    logging.info(f"DataFrame exported to {output_file_path}")


def load_df(input_path, **kwargs):
    """
    Loads a DataFrame from a specified file format.

    Parameters
    ----------
    input_path : str
        The path to the input file.
    """
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path, **kwargs)
    elif input_path.endswith(".parquet"):
        return pd.read_parquet(input_path, **kwargs)
    elif input_path.endswith(".pkl"):
        return pd.read_pickle(input_path, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file extension for loading: {os.path.splitext(input_path)[1]}"
        )


def get_tasks(activity, split):
    target_col_path = os.path.join(DATASET_DIR, activity, split, "target_col.pkl")
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


# TODO check this function
# def get_datasets(activity, split_type, device=DEVICE):
#     try:
#         paths = {
#             split: os.path.join(DATASET_DIR, activity, split_type, f"{split}.pkl")
#             for split in ["train", "val", "test"]
#         }
#         datasets = {}
#
#         for input_col in ["ecfp1024", "ecfp2048"]:
#             for split, dataset_path in paths.items():
#                 key = f"{split}_{input_col}"
#                 datasets[key] = PapyrusDatasetMT(
#                     dataset_path, input_col=input_col, device=device
#                 )
#         return datasets
#
#     except Exception as e:
#         raise RuntimeError(f"Error loading datasets: {e}")


def get_data_info(train_data, val_data, test_data):
    combined_data = pd.concat(
        [train_data, val_data, test_data], keys=["train", "val", "test"]
    )
    combined_data.reset_index(inplace=True)
    count_data = combined_data.groupby("level_0").count()
    count_data = count_data.pivot_table(columns="level_0")
    count_data.reset_index(inplace=True)

    return count_data


def create_split_dict(split_type, train_df, val_df, test_df, **kwargs):
    out = {
        split_type: {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "info": get_data_info(train_df, val_df, test_df),
        }
    }
    out[split_type].update(kwargs)
    return out


def random_split(
    df: pd.DataFrame, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42, **kwargs
):
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
    **kwargs
        Additional keyword arguments to be included in the result dictionary.
    Returns:
    --------
    train_df : pandas.DataFrame
        The training dataframe.
    val_df : pandas.DataFrame
        The validation dataframe.
    test_df : pandas.DataFrame
        The testing dataframe.
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

    # create result dictionary for return
    return create_split_dict("random", train_df, val_df, test_df, **kwargs)


def scaffold_split(
    df,
    smiles_col="smiles",
    train_frac=0.7,
    val_frac=0.15,
    test_frac=0.15,
    seed=42,
    **kwargs,
):
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
    seed : int, optional
        The random seed to use for splitting the data. Default is 42.
    kwargs
        Additional keyword arguments to be included in the result dictionary.
    Returns
    -------
    train_df : pandas.DataFrame
        The training dataframe.
    val_df : pandas.DataFrame
        The validation dataframe.
    test_df : pandas.DataFrame
        The testing dataframe.
    """
    # set random seed
    np.random.seed(seed)

    # calculate scaffolds for each smiles string
    df["scaffold"] = df[smiles_col].apply(generate_scaffold)

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

    # create result dictionary for return
    return create_split_dict("scaffold", train_df, val_df, test_df, **kwargs)


def time_split(
    df, time_col="year", train_frac=0.7, val_frac=0.15, test_frac=0.15, **kwargs
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
    kwargs
        Additional keyword arguments to be included in the result dictionary.
    Returns
    -------
    train_df : pandas.DataFrame
        The training dataframe.
    val_df : pandas.DataFrame
        The validation dataframe.
    test_df : pandas.DataFrame
        The testing dataframe.
    """
    # order df by time_col and split
    df = df.sort_values(by=time_col)
    train_df = df.iloc[: int(train_frac * len(df))]
    val_df = df.iloc[int(train_frac * len(df)) : int((train_frac + val_frac) * len(df))]
    test_df = df.iloc[int((train_frac + val_frac) * len(df)) :]

    # create result dictionary for return
    return create_split_dict("time", train_df, val_df, test_df, **kwargs)


def split_data(
    df: pd.DataFrame,
    split_type: str = "random",
    smiles_col: str = "smiles",
    time_col: str = "year",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    **kwargs,
):
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
    train_frac : float, optional
        The fraction of the data to use for training. Default is 0.7.
    val_frac : float, optional
        The fraction of the data to use for validation. Default is 0.15.
    test_frac : float, optional
        The fraction of the data to use for testing. Default is 0.15.
    seed : int, optional
        The random seed to use for splitting the data. Default is 42.
    kwargs
        Additional keyword arguments to be included in the result dictionary.

    Returns
    -------
    dict
        A dictionary with key as split_type and value is another dict containing the training, validation, and test dataframes.
    """
    # Validate that the fractions sum up to 1
    total_frac = train_frac + val_frac + test_frac
    if total_frac != 1:
        raise ValueError("The sum of train_frac, val_frac, and test_frac must be 1.")

    all_data = {}
    func_key = {
        "random": (random_split, {}),
        "scaffold": (scaffold_split, {"smiles_col": smiles_col}),
        "time": (time_split, {"time_col": time_col}),
    }

    if split_type == "all":
        for t in ["random", "scaffold", "time"]:
            sub_dict = split_data(
                df,
                split_type=t,
                smiles_col=smiles_col,
                time_col=time_col,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                seed=seed,
                **kwargs,
            )
            all_data.update(sub_dict)

    elif split_type in func_key.keys():
        # here we either get output_path from kwargs or use the default DATASET_DIR; either merged with split_type
        output_path = os.path.join(kwargs.pop("output_path", DATASET_DIR), split_type)
        split_func, split_kwargs = func_key[split_type]
        sub_dict = split_func(
            df,
            **split_kwargs,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            output_path=output_path,
            **kwargs,
        )
        all_data.update(sub_dict)

    else:
        raise ValueError("split_type must be one of 'random', 'scaffold', or 'time'.")

    return all_data
