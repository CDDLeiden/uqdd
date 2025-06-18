import argparse
import ast
import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Union, Tuple, Dict, Any

import numpy as np
import pandas as pd

from uqdd import CONFIG_DIR, LOGS_DIR

string_types = (type(b""), type(""))


def float_or_none(value: str) -> Union[float, None]:
    """
    Converts a string input to a float or None if 'none' is provided.

    Parameters:
    -----------
    value : str
        The string value to be converted.

    Returns:
    --------
    float or None
        The converted float value or None.
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
    Initializes and returns a logger with specified log levels.

    Parameters:
    -----------
    name : str, optional
        The name of the logger (default: "logger").
    file_level : str, optional
        The log level for file output (default: "debug").
    stream_level : str, optional
        The log level for stream output (default: "info").

    Returns:
    --------
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
    logging.getLogger().setLevel(stream_level)
    log = logging.getLogger(name)

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
    Loads a configuration JSON file and updates it with optional overrides.

    Parameters:
    -----------
    config_name : str
        The name of the configuration file (without extension).
    config_dir : str or Path, optional
        The directory containing the config files (default: CONFIG_DIR).
    split_key : str or None, optional
        The specific split key to extract from the config (default: None).
    activity_key : str or None, optional
        The specific activity key to extract from the config (default: None).
    kwargs : dict
        Additional key-value pairs to update the configuration.

    Returns:
    --------
    Dict[str, Any]
        The loaded and updated configuration.
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
    Custom aggregation function that returns a unique value or a list of unique values.

    Parameters:
    -----------
    x : pd.Series
        A Pandas Series to aggregate.

    Returns:
    --------
    Any or List[Any]
        A single unique value if only one exists, otherwise a list of unique values.
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
    Identifies and separates rows containing NaN values in specified columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check for NaN values.
    cols : Union[List[str], str], optional
        Columns to check for NaNs (default: "smiles").
    nan_dup_source : str, optional
        Source label for NaN values (default: "").
    logger : logging.Logger or None, optional
        Logger instance for logging (default: None).

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrame without NaNs and DataFrame containing NaNs.

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
    Identifies and optionally removes duplicate rows based on specified columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check for duplicates.
    cols : Union[List[str], str]
        Columns to consider for duplicate checking.
    drop : bool, optional
        Whether to drop duplicate rows (default: True).
    sorting_col : str, optional
        Column to sort duplicates by (default: "").
    keep : Union[bool, str], optional
        Determines which duplicate entries to keep ('first', 'last', or False for all) (default: 'first').
    nan_dup_source : str, optional
        Source label for duplicate values (default: "").
    logger : logging.Logger or None, optional
        Logger instance for logging (default: None).

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrame without duplicates and DataFrame containing duplicates.

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
    Checks for NaN and duplicated values in a DataFrame and separates them.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check.
    cols_nan : Union[List[str], str], optional
        Columns to check for NaNs (default: "smiles").
    cols_dup : Union[List[str], str], optional
        Columns to check for duplicates (default: "smiles").
    nan_dup_source : str, optional
        Source label for NaN or duplicated values (default: "").
    drop : bool, optional
        Whether to drop duplicate values (default: True).
    sorting_col : str, optional
        Column to sort duplicates by (default: "").
    keep : Union[bool, str], optional
        Determines which duplicate entries to keep ('first', 'last', or False for all) (default: 'first').
    logger : logging.Logger or None, optional
        Logger instance for logging (default: None).

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Filtered DataFrame without NaN or duplicated values, DataFrame with NaNs, and DataFrame with duplicates.
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
    Parses a string representation of a list into an actual list of integers.

    Parameters:
    -----------
    argument : str
        The string representation of a list.

    Returns:
    --------
    List[int]
        The parsed list of integers.
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
    Saves a numpy array to a file with a .npy extension.

    Parameters:
    -----------
    array_data : np.ndarray
        The numpy array to save.
    filepath : str
        The path where the file will be saved.

    Returns:
    --------
    None
    """
    # Save the array to the specified filepath
    np.save(filepath, array_data)


def load_npy_file(filepath: Union[str, Path]) -> np.ndarray:
    """
    Loads a .npy file into a numpy array.

    Parameters:
    -----------
    filepath : str
        The path to the file to load.

    Returns:
    --------
    np.ndarray
        The loaded numpy array.
    """
    # Load the file as a numpy array
    return np.load(filepath)


def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
    """
    Saves data to a pickle file.

    Parameters:
    -----------
    data : Any
        The data to be saved.
    file_path : str
        The file path where the pickle file will be stored.

    Returns:
    --------
    None
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Loads data from a pickle file.

    Parameters:
    -----------
    filepath : str
        The path to the pickle file.

    Returns:
    --------
    Any
        The data loaded from the pickle file.
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
    Exports a DataFrame to a specified file format.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to be exported.
    file_path : str or None, optional
        Full file path for export (default: None).
    output_path : str, optional
        The directory path for exporting files (default: "./").
    filename : str, optional
        The filename to use if file_path is not specified (default: "exported_file").
    ext : str, optional
        The file extension to use for saving (default: "csv").
    kwargs : dict
        Additional arguments for file export.

    Returns:
    --------
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
    Loads a DataFrame from a specified file format.

    Parameters:
    -----------
    input_path : str or Path
        The file path to load.
    kwargs : dict
        Additional arguments for file loading.

    Returns:
    --------
    pd.DataFrame
        The loaded DataFrame.
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
    Splits a list into multiple sublists based on specified sizes.

    Parameters:
    -----------
    input_list : List[Any]
        The list to be split.
    sizes : List[int]
        The sizes of the sublists.

    Returns:
    --------
    List[List[Any]]
        A list of sublists split according to the given sizes.
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
