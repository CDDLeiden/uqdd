"""
General utilities for configuration, logging, serialization, and DataFrame I/O.

This module provides helpers to parse inputs, create loggers, load/update configs,
read/write arrays and pickles, and perform common DataFrame hygiene tasks.
"""

import argparse
import ast
import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Union, Tuple, Dict, Any

import numpy as np
import pandas as pd

from uqdd import CONFIG_DIR, LOGS_DIR

string_types = (type(b""), type(""))


def float_or_none(value: str) -> Union[float, None]:
    """
    Convert a string to a float, returning None if the value is 'none'.

    Parameters
    ----------
    value : str
        String value to be converted.

    Returns
    -------
    float or None
        Converted float value, or None if input equals 'none' (case-insensitive).

    Raises
    ------
    argparse.ArgumentTypeError
        If the input cannot be converted to a float.
    """
    if value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: '{value}'")


def create_logger(
        name: str = "logger", file_level: str = "debug", stream_level: str = "info"
) -> logging.Logger:
    """
    Initialize and return a configured Python logger.

    Parameters
    ----------
    name : str, optional
        Logger name. Default is "logger".
    file_level : str, optional
        Log level for file handler ("debug", "info", "warning", "error", "critical"). Default is "debug".
    stream_level : str, optional
        Log level for stream handler ("debug", "info", "warning", "error", "critical"). Default is "info".

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
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
    # to capture all messages intended for its handlers.
    # logging.getLogger().setLevel(stream_level)
    log = logging.getLogger(name)
    log.setLevel(min(file_level, stream_level))

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
        split_key: Union[str, None] = None,
        activity_key: Union[str, None] = None,
        **kwargs,
) -> Dict[str, Any]:
    """
    Load a JSON configuration and optionally select nested keys and apply overrides.

    Parameters
    ----------
    config_name : str
        Name of the configuration file (without extension).
    config_dir : str or Path, optional
        Directory containing config files. Default is CONFIG_DIR.
    split_key : str or None, optional
        Nested key to select a specific split from the config. Default is None.
    activity_key : str or None, optional
        Nested key to select a specific activity from the config. Default is None.
    **kwargs
        Additional key-value overrides applied to the loaded config.

    Returns
    -------
    Dict[str, Any]
        Loaded configuration dictionary, possibly nested selection and updated with overrides.

    Raises
    ------
    FileNotFoundError
        If the requested configuration file does not exist.
    """
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
def custom_agg(x: pd.Series) -> Union[Any, List[Any]]:
    """
    Aggregate a Series to either a single unique value or a list of unique values.

    Parameters
    ----------
    x : pd.Series
        Pandas Series to aggregate.

    Returns
    -------
    Any or List[Any]
        Single unique value if only one exists; otherwise a list of unique non-null values.
    """
    if len(x) > 1:
        unique = x.nunique()
        if unique == 1:
            return x.iloc[0]
        else:
            return x.dropna().unique().tolist()
    else:
        return x.iloc[0]


def check_na(
        df: pd.DataFrame,
        cols: Union[List[str], str] = "smiles",
        nan_dup_source: str = "",
        logger: Union[logging.Logger, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify rows containing NaN in specified columns and separate them.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : list of str or str, optional
        Column(s) to check for NaNs. Default is "smiles".
    nan_dup_source : str, optional
        Source label to annotate NaN rows. Default is empty string.
    logger : logging.Logger or None, optional
        Logger for info/error messages. Default is None.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Tuple of (filtered DataFrame without NaNs, DataFrame of NaN rows).

    Raises
    ------
    AssertionError
        If any requested column is missing from the DataFrame.
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
        logger: Union[logging.Logger, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify duplicate rows by columns, optionally sort and drop, and annotate.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : list of str or str
        Column(s) to consider for duplicate checking.
    drop : bool, optional
        Whether to drop duplicate rows from the input DataFrame. Default is True.
    sorting_col : str, optional
        Column used to sort duplicates before dropping. Default is empty string.
    keep : bool or {"first", "last"}, optional
        Which duplicate to keep when dropping. Default is "first".
    nan_dup_source : str, optional
        Source label to annotate duplicates. Default is empty string.
    logger : logging.Logger or None, optional
        Logger for info/error messages. Default is None.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Tuple of (DataFrame without dropped duplicates, DataFrame containing duplicates).

    Raises
    ------
    AssertionError
        If requested columns or sorting_col are missing, or keep is invalid.
    """
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
        df: pd.DataFrame,
        cols_nan: Union[List[str], str] = "smiles",
        cols_dup: Union[List[str], str] = "smiles",
        nan_dup_source: str = "",
        drop: bool = True,
        sorting_col: str = "",
        keep: Union[bool, str] = "first",
        logger: Union[logging.Logger, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Check both NaN and duplicates and split the DataFrame into clean, NaN, and duplicate subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols_nan : list of str or str, optional
        Column(s) to check for NaNs. Default is "smiles".
    cols_dup : list of str or str, optional
        Column(s) to check for duplicates. Default is "smiles".
    nan_dup_source : str, optional
        Source label to annotate NaN/duplicate rows. Default is empty string.
    drop : bool, optional
        Whether to drop duplicate rows. Default is True.
    sorting_col : str, optional
        Column to sort duplicates by. Default is empty string.
    keep : bool or {"first", "last"}, optional
        Which duplicate to keep when dropping. Default is "first".
    logger : logging.Logger or None, optional
        Logger for info/error messages. Default is None.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        Tuple of (filtered DataFrame, NaN rows, duplicate rows).
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


def parse_list(argument: str) -> List[int]:
    """
    Parse a string representation of a list into a list of integers.

    Parameters
    ----------
    argument : str
        String representation of a list, e.g. "[1, 2, 3]".

    Returns
    -------
    list of int
        Parsed list of integers.

    Raises
    ------
    argparse.ArgumentTypeError
        If the argument is not a list of integers.
    """
    try:
        val = ast.literal_eval(argument)
        if not (isinstance(val, list) or all(isinstance(x, int) for x in val)):
            raise ValueError
        return val
    except ValueError:
        raise argparse.ArgumentTypeError("Argument is not a list of integers")


def save_npy_file(array_data: np.ndarray, filepath: str) -> None:
    """
    Save a numpy array to a .npy file.

    Parameters
    ----------
    array_data : numpy.ndarray
        Array to save.
    filepath : str
        Output file path.

    Returns
    -------
    None
    """
    # Save the array to the specified filepath
    np.save(filepath, array_data)


def load_npy_file(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load a .npy file into a numpy array.

    Parameters
    ----------
    filepath : str or Path
        File path to load.

    Returns
    -------
    numpy.ndarray
        Loaded array.
    """
    # Load the file as a numpy array
    return np.load(filepath)


def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
    """
    Save Python data to a pickle file.

    Parameters
    ----------
    data : Any
        Data object to serialize.
    file_path : str or Path
        Destination file path.

    Returns
    -------
    None
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load data from a pickle file.

    Parameters
    ----------
    filepath : str or Path
        Pickle file path.

    Returns
    -------
    Any
        Deserialized Python object.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    Exception
        If any other error occurs during loading.
    """
    try:
        with open(filepath, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
    except Exception as e:
        logging.error(f"Error loading file {filepath}: {e}")


def save_df(
        df: pd.DataFrame,
        file_path: Union[str, None] = None,
        output_path: str | Path = "./",
        filename: str = "exported_file",
        ext: str = "csv",
        **kwargs,
) -> None:
    """
    Export a DataFrame to a file in a chosen format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.
    file_path : str or None, optional
        Full file path for export; if provided, overrides output_path/filename/ext. Default is None.
    output_path : str or Path, optional
        Directory path for exporting files. Default is "./".
    filename : str, optional
        Filename to use when file_path is not provided. Default is "exported_file".
    ext : str, optional
        Output file extension: one of {"csv", "parquet", "feather", "pkl", "xlsx"}. Default is "csv".
    **kwargs
        Additional keyword arguments passed through to the underlying pandas writer.

    Returns
    -------
    None
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


def load_df(input_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load a DataFrame from CSV, Parquet, or Pickle.

    Parameters
    ----------
    input_path : str or Path
        Path to the input file.
    **kwargs
        Additional keyword arguments forwarded to pandas readers.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    ValueError
        If the file extension is not one of supported {csv, parquet, pkl}.
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


def split_list_by_sizes(input_list: List[Any], sizes: List[int]) -> List[List[Any]]:
    """
    Split a list into multiple sublists based on specified sizes.

    Parameters
    ----------
    input_list : list of Any
        Input list to be split.
    sizes : list of int
        Sizes of the sublists; must sum to the length of input_list.

    Returns
    -------
    list of list of Any
        List of sublists split according to the given sizes.

    Raises
    ------
    ValueError
        If the sum of sizes does not equal the length of input_list.
    """
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
