import argparse
import ast
import os
import json
import pickle
from pathlib import Path
from typing import List, Union, Tuple  # , List, Tuple, Any, Set, Dict
import logging
import numpy as np
import pandas as pd
from uqdd import CONFIG_DIR, LOGS_DIR

string_types = (type(b""), type(""))


def float_or_none(value):
    if value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: '{value}'")


def create_logger(name="logger", file_level="debug", stream_level="info"):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    file_level = levels.get(file_level.lower(), logging.DEBUG)
    stream_level = levels.get(stream_level.lower(), logging.INFO)
    # Set root logger to the lowest level between file_level and stream_level
    logging.getLogger().setLevel(stream_level)
    log = logging.getLogger(name)
    # log.setLevel(min(file_level, stream_level))
    # # Set logger's level to the lower between file_level and stream_level
    # log.setLevel(min(file_level, stream_level))

    formatter = logging.Formatter(
        fmt="%(asctime)s:%(levelname)s:%(name)s:%(message)s:%(relativeCreated)d",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    out_log = os.path.join(LOGS_DIR, f"{name}.log")
    file_handler = logging.FileHandler(out_log, mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(file_level)
    log.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(stream_level)
    log.addHandler(stream_handler)

    log.debug(f"Logger {name} initialized")
    return log


def get_config(
    config_name: str,
    config_dir: Union[str, Path] = CONFIG_DIR,
    split_key=None,
    activity_key=None,
    **kwargs,
):
    config_path = Path(config_dir) / f"{config_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file '{config_path}' does not exist. Ensure the file is present in '{CONFIG_DIR}'."
        )

    with open(config_path) as f:
        config = json.load(f)
        if activity_key is not None:
            config = config[activity_key]
        if split_key is not None:
            config = config[split_key]

    if kwargs:
        config.update(kwargs)

    return config


# Define a custom function to aggregate the columns
def custom_agg(x):
    if len(x) > 1:
        unique = x.nunique()
        if unique == 1:
            return x.iloc[0]
        else:
            return x.dropna().unique().tolist()
    else:
        return x.iloc[0]


def check_na(
    df, cols: Union[List[str], str] = "smiles", nan_dup_source="", logger=None
):
    """
    Check for NaN values in the specified column(s) of the given DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check for NaN values.
    cols : Union[List[str], str], default='smiles'
        A list of column names or a single column name to check for NaN values.
    nan_dup_source : str, optional
        Source of NaN values in the DataFrame.
    logger : Logger, optional
        Logger object to use for logging.

    Returns:
    --------
    df : pandas.DataFrame
        A copy of the DataFrame with rows containing NaN values in the specified columns removed.
    df_nan : pandas.DataFrame
        A copy of the DataFrame with rows containing NaN values in the specified columns.

    Raises:
    -------
    AssertionError :
        If any column from `cols` is not present in the DataFrame.

    Example:
    --------
    # >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 4, 5], 'C': [7, 8, 9]})
    # >>> df_clean, df_nan = check_na(df, cols=['A', 'B'], nan_dup_source='original')
    # >>> print(df_clean)
       A    B  C
    0  1.0  4.0  7
    1  2.0  5.0  8
    # >>> print(df_nan)
         A   B  C nan_dup_source
    0  NaN NaN  9       original
    """
    if not isinstance(cols, list):
        cols = [cols]

    assert all([col in df.columns for col in cols]), (
        f"One or more columns from `cols` for checking NaNs, not in the dataframe"
        f"The dataframe cols are {df.columns}"
    )

    try:
        na_mask = df[cols].isna().any(axis=1)
        df_nan = df[na_mask].copy()
        df = df[~na_mask].copy()
        if nan_dup_source and (na_mask.sum() > 0):
            # (nan_df.shape[0] > 0):
            # (na_mask.sum() > 0):
            df_nan.loc[:, "nan_dup_source"] = nan_dup_source

        if logger:
            logger.info(
                f"Checked the {cols} column(s) for NaN values "
                f"Number of NaN values: {df_nan.shape[0]}"
            )
    except Exception as e:
        df_nan = pd.DataFrame(columns=df.columns)  # empty dataframe
        logger.error(
            f"Unexpected Error while checking for NaN empty values in {cols} in check_na() function: {e}"
        )

    return df, df_nan


def check_duplicates(
    df: pd.DataFrame,
    cols: Union[List[str], str],
    drop: bool = True,
    sorting_col: str = "",
    keep: Union[bool, str] = "first",
    nan_dup_source: str = "",
    logger=None,
):
    """
    Check for duplicates in the specified column(s) of the given DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check for duplicates.
    cols : Union[List[str], str]
        A list of column names or a single column name to check for duplicates.
    drop : bool, optional, default=True
        If True, drop the duplicate rows.
    sorting_col : str, optional, default=""
        The column by which to sort the duplicates DataFrame.
    keep : Union[bool, str], optional, default='first'
        If drop is True, this parameter indicates which duplicate values to keep.
        If keep is 'first', the first occurrence of the duplicate value is kept.
        If keep is 'last', the last occurrence of the duplicate value is kept.
        If keep is False, all occurrences of the duplicate value are dropped.
    nan_dup_source : str, optional, default=""
        Source of duplicate values in the DataFrame.
    logger : Logger, optional
        Logger object to use for logging.

    Returns:
    --------
    df : pandas.DataFrame
        A copy of the DataFrame with duplicate rows removed if drop is True.
    df_dup : pandas.DataFrame
        A copy of the DataFrame with only duplicate rows if any.

    Raises:
    -------
    AssertionError :
        If any column from `cols` is not present in the DataFrame.
        If sorting_col is not present in the DataFrame.
        If keep is not one of 'first', 'last', or False.

    Example:
    --------
    # >>> df = pd.DataFrame({'A': [1, 2, 2], 'B': [4, 5, 4], 'C': [7, 8, 9]})
    # >>> df_clean, df_dup = check_duplicates(df, cols=['A', 'B'], keep='last')
    # >>> print(df_clean)
       A  B  C
    0  1  4  7
    2  2  4  9
    # >>> print(df_dup)
       A  B  C
    1  2  5  8
    """
    # df = df[[x, y]].
    if not isinstance(cols, list):
        cols = [cols]
    assert all([col in df.columns for col in cols]), (
        "One or more columns from `cols` for checking duplicates, not in the dataframe"
        f"The dataframe cols are {df.columns}"
    )
    assert sorting_col in df.columns or sorting_col == "", (
        "sorting_col not in the dataframe" f"The dataframe cols are {df.columns}"
    )
    assert keep in [
        "first",
        "last",
        False,
    ], "keep should be either 'first', 'last' or False"

    try:
        dup_mask = df.duplicated(subset=cols, keep=False)
        df_dup = df[dup_mask].copy()
        if nan_dup_source and dup_mask.sum() > 0:
            df_dup.loc[:, "nan_dup_source"] = nan_dup_source

        if sorting_col and (not df_dup.empty):
            s_cols = cols + [sorting_col]  # WRONG: s_cols = cols.append(sorting_col)
            df_dup = df_dup.sort_values(by=s_cols, ascending=True)

        if drop:
            # now we get the ones to keep
            to_keep = df_dup.drop_duplicates(subset=cols, keep=keep)
            # now we drop the ones to keep from the duplicates dataframe
            to_drop = df_dup.drop(to_keep.index)
            df = df.drop(to_drop.index)

        if logger:
            logger.info(
                f"Checked the {cols} column(s) for duplicates "
                f"Number of ALL duplicates: {df_dup.shape[0]}"
            )

    except Exception as e:
        df_dup = pd.DataFrame(columns=df.columns)  # empty dataframe
        logger.error(
            f"Unexpected Error while checking for duplicates in {cols} in check_duplicates() function: {e}"
        )

    return df, df_dup


def check_nan_duplicated(
    df,
    cols_nan: Union[List[str], str] = "smiles",
    cols_dup: Union[List[str], str] = "smiles",
    nan_dup_source="",
    drop=True,
    sorting_col="",
    keep="first",
    logger=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Check for NaN and duplicated values in a pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to check for NaN and duplicated values.
    cols_nan : Union[List[str], str], optional
        The column(s) to check for NaN values, by default "smiles".
    cols_dup : Union[List[str], str], optional
        The column(s) to check for duplicated values, by default "smiles".
    nan_dup_source : str, optional
        The source of NaN or duplicated values, by default "".
    drop : bool, optional
        Whether to drop the duplicated values, by default True.
    sorting_col : str, optional
        The column used for sorting in the case of duplicated values, by default "".
    keep : Union[bool, str], optional
        Whether to keep the first or last occurrence of duplicated values, or False to drop all, by default "first".
    logger : logger object, optional
        Logger object for logging the results of the check, by default None.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the filtered DataFrame (without NaN or duplicated values), a DataFrame containing the rows
        with NaN values, and a DataFrame containing the duplicated rows.
    """
    # NaN Check
    df_filtered, df_nan = check_na(df, cols_nan, nan_dup_source, logger)
    # Duplicates Check
    df_filtered, df_dup = check_duplicates(
        df_filtered,
        cols_dup,
        drop=drop,
        sorting_col=sorting_col,
        keep=keep,
        nan_dup_source=nan_dup_source,
        logger=logger,
    )

    return df_filtered, df_nan, df_dup


def parse_list(argument):
    try:
        val = ast.literal_eval(argument)
        if not (isinstance(val, list) or all(isinstance(x, int) for x in val)):
            raise ValueError
        return val
    except ValueError:
        raise argparse.ArgumentTypeError("Argument is not a list of integers")


def save_npy_file(array_data, filepath):
    """
    Save a numpy array to a file with a .pkl.npy or .npy extension.

    Args:
    array_data (numpy.ndarray): The numpy array to save.
    filepath (str): The path where the file will be saved.
    """
    # Save the array to the specified filepath
    np.save(filepath, array_data)


def load_npy_file(filepath):
    """
    Load a .pkl.npy file into a numpy array.

    Args:
    filepath (str): The path to the file to load.

    Returns:
    numpy.ndarray: The array stored in the file.
    """
    # Load the file as a numpy array
    return np.load(filepath)


def save_pickle(data, file_path):
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


def save_df(
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
        # filename = path_obj.stem
        ext = path_obj.suffix.lstrip(".")
    else:
        if not filename.endswith(ext):
            filename = f"{filename}.{ext}"
        # file_path = Path(output_path) / filename
        file_path = os.path.join(output_path, filename)

    # os.makedirs(output_path, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if ext == "csv":
        df.to_csv(file_path, index=False, **kwargs)
    elif ext == "parquet":
        df.to_parquet(file_path, index=False, **kwargs)
    elif ext == "feather":
        df.to_feather(file_path, **kwargs)
    elif ext == "pkl":
        df.to_pickle(file_path, **kwargs)
    elif ext == "xlsx":
        df.to_excel(file_path, index=False, **kwargs)
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


def split_list_by_sizes(input_list, sizes):
    # Verify that the input list and sizes list are valid
    if len(input_list) != sum(sizes):
        raise ValueError(
            "The sum of sizes must be equal to the length of the input list."
        )

    result = []
    start_index = 0

    for size in sizes:
        end_index = start_index + size
        result.append(input_list[start_index:end_index])
        start_index = end_index

    return result
