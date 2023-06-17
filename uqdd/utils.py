from typing import List, Union, Tuple  # , List, Tuple, Any, Set, Dict

import numpy as np
import pandas as pd
# noinspection PyUnresolvedReferences
from rdkit.Chem.rdchem import Mol as RdkitMol

string_types = (type(b""), type(""))


########### PANDAS UTILS ###############
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
    >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 4, 5], 'C': [7, 8, 9]})
    >>> df_clean, df_nan = check_na(df, cols=['A', 'B'], nan_dup_source='original')
    >>> print(df_clean)
       A    B  C
    0  1.0  4.0  7
    1  2.0  5.0  8
    >>> print(df_nan)
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
        >>> df = pd.DataFrame({'A': [1, 2, 2], 'B': [4, 5, 4], 'C': [7, 8, 9]})
        >>> df_clean, df_dup = check_duplicates(df, cols=['A', 'B'], keep='last')
        >>> print(df_clean)
           A  B  C
        0  1  4  7
        2  2  4  9
        >>> print(df_dup)
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

        # num_of_dups = df_dup.shape[0]
        #
        # if num_of_dups == 0:
        #     return df, num_of_dups
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
        Whether or not to drop the duplicated values, by default True.
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


# def setup_primary_logging(log_file, level):
#     log_queue = Queue(-1)
#
#     file_handler = logging.FileHandler(filename=log_file)
#     stream_handler = logging.StreamHandler()
#
#     formatter = logging.Formatter(
#         '%(asctime)s | %(levelname)s | %(message)s',
#         datefmt='%Y-%m-%d,%H:%M:%S')
#
#     file_handler.setFormatter(formatter)
#     stream_handler.setFormatter(formatter)
#
#     file_handler.setLevel(level)
#     stream_handler.setLevel(level)
#
#     listener = QueueListener(log_queue, file_handler, stream_handler)
#     listener.start()
#
#     return log_queue

