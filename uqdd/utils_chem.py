import copy
import logging
from typing import Union, List, Tuple, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

import rdkit
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Draw, AllChem, MACCSkeys, rdFMCS
from rdkit.Chem import MolToSmiles, MolFromSmiles, MolFromSmarts, SanitizeMol
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem.rdchem import Mol as RdkitMol

from uqdd import DATA_DIR, FIGS_DIR
from uqdd.utils import check_nan_duplicated, custom_agg, load_npy_file, save_npy_file

from papyrus_scripts.reader import read_molecular_descriptors
from papyrus_scripts.preprocess import consume_chunks

# scipy hierarchy clustering
from scipy.cluster.hierarchy import dendrogram #, linkage
from scipy.cluster.hierarchy import fcluster, cophenet
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
import fastcluster as fc
from fastcluster import linkage
# SKlearn metrics
from sklearn.metrics import (
    silhouette_samples,
    silhouette_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
from sklearn.neighbors import NearestCentroid

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations, islice
import math

# Disable RDKit warnings
RDLogger.DisableLog("rdApp.info")
print(f"rdkit {rdkit.__version__}")

N_WORKERS = 20
BATCH_SIZE = 20000

all_models = [
    "ecfp1024",
    "ecfp2048",
    "mold2",
    "mordred",
    "cddd",
    "fingerprint",
    # "moldesc",
    # "moe",
    # "graph2d",
]

descriptors = [
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "EState_VSA1",
    "EState_VSA10",
    "EState_VSA11",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "ExactMolWt",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "FractionCSP3",
    "HallKierAlpha",
    "HeavyAtomCount",
    "HeavyAtomMolWt",
    "Ipc",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "MaxAbsEStateIndex",
    "MaxAbsPartialCharge",
    "MaxEStateIndex",
    "MaxPartialCharge",
    "MinAbsEStateIndex",
    "MinAbsPartialCharge",
    "MinEStateIndex",
    "MinPartialCharge",
    "MolLogP",
    "MolMR",
    "MolWt",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRadicalElectrons",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "NumValenceElectrons",
    "PEOE_VSA1",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA12",
    "PEOE_VSA13",
    "PEOE_VSA14",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA4",
    "PEOE_VSA5",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "RingCount",
    "SMR_VSA1",
    "SMR_VSA10",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA8",
    "SMR_VSA9",
    "SlogP_VSA1",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA7",
    "SlogP_VSA8",
    "SlogP_VSA9",
    "TPSA",
    "VSA_EState1",
    "VSA_EState10",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9",
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
    "qed",
]


def rdkit_standardize(smi, logger=None, suppress_exception=False):
    """
    Applies a standardization workflow to a SMILES string.

    Parameters:
    -----------
    smi : str
        The input SMILES string to standardize.

    logger : logging.Logger, optional
        A logger object to log error messages. Default is None.

    suppress_exception : bool, optional
        A boolean flag to suppress exceptions and return the original SMILES string if an error
        occurs during standardization. If False, an exception is raised or logged, depending on the value of logger.
        Default is True.

    Returns:
    --------
    str
        The standardized SMILES string.

    Raises
    ------
    TypeError
        If check_smiles_type is True and the input is not a string.
    StandardizationError
        If an unexpected error occurs during standardization and suppress_exception is False.
        The error message is logged or raised, depending on the value of logger.

    Notes:
    ------
    This function applies the following standardization steps to the input SMILES string:

    1. Functional Groups Normalization: The input SMILES string is converted to a molecule object,
    and any functional groups present are normalized to a standard representation.
    2. Sanitization: The molecule is sanitized, which involves performing various checks
    and corrections to ensure that it is well-formed.
    3. Neutralization: Any charges on the molecule are neutralized.
    4. Parent Tautomer: The canonical tautomer of the molecule is determined.

    This function uses the RDKit library for performing standardization.
    implementation source:
    https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    """
    if smi is None:
        return None
    og_smiles = copy.deepcopy(smi)
    try:
        # Functional Groups Normalization
        mol = MolFromSmiles(smi)
        mol.UpdatePropertyCache(strict=False)
        SanitizeMol(
            mol,
            sanitizeOps=(
                Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES
            ),
        )
        mol = rdMolStandardize.Normalize(mol)

        # Neutralization
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(rdMolStandardize.FragmentParent(mol))

    except Exception as e:
        if logger:
            logger.error(f"StandardizationError: {e} for {og_smiles}")
        if suppress_exception:
            return og_smiles
        else:
            return None

    return MolToSmiles(mol)


def remove_stereo_rdkit_molecule(
    mol: RdkitMol,
) -> Optional[RdkitMol]:
    try:
        Chem.RemoveStereochemistry(mol)
        return mol

    except Exception as e:
        raise ValueError(
            f"Removing Stereochemistry failed with the following error {e}"
        )


def neutralize_rdkit_molecule(
    mol: RdkitMol,
) -> Optional[RdkitMol]:
    try:
        pattern = MolFromSmarts(
            "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4]),-1!$([*]~[1+,2+,3+,4+])]"
        )
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]

        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                h_count = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(h_count - chg)
                atom.UpdatePropertyCache()

        return mol

    except Exception as e:
        raise ValueError(f"Neutralization failed with the following error {e}")


def remove_isotopes_rdkit_molecule(
    mol: RdkitMol,
) -> Optional[RdkitMol]:
    try:
        atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
        for atom, isotope in atom_data:
            # restore original isotope values
            if isotope:
                atom.SetIsotope(0)
        Chem.RemoveHs(mol)
        return mol

    except Exception as e:
        raise ValueError(f"Removing Isotope failed with the following error {e}")


def standardize(
    smiles: Optional[str],
    logger: Optional[logging.Logger] = None,
    suppress_exception: bool = True,
    remove_stereo: bool = False,
) -> Optional[str]:
    """
    Standardizes a given SMILES string using RDKit.

    Parameters
    ----------
    smiles : str, optional
        A SMILES string to be standardized. If None, returns None.
    remove_stereo : bool, optional
        A boolean flag to remove stereochemistry information if True. Default is False.
    logger : logging.Logger, optional
        A logger object to log error messages. Default is None.
    suppress_exception : bool, optional
        A boolean flag to suppress exceptions and return the original SMILES string if an error
        occurs during standardization. If False, an exception is raised or logged, depending on the value of logger.
        Default is True.

    Returns
    -------
    str or None
        The standardized SMILES string or None, depending on the value of suppress_exception and whether an exception
        occurs. If an exception occurs and suppress_exception is True, the original SMILES string is returned.
    """
    if smiles is None:
        return None
    og_smiles = copy.deepcopy(smiles)
    smiles_inter = None
    # if check_smiles_type and not isinstance(smiles, str):
    #     if logger:
    #         logger.error(
    #             f"smiles must be a string and not {type(smiles)}, "
    #             f"the following input is incorrect : {smiles}"
    #         )
    #     return None

    try:
        smiles = smiles.split("|")[0].split("{")[0].strip()
        mol = MolFromSmiles(smiles)
        if mol is None:
            return None

        smiles_inter = MolToSmiles(mol, canonical=True)

        if remove_stereo:
            mol = remove_stereo_rdkit_molecule(mol)
        mol = neutralize_rdkit_molecule(mol)
        mol = remove_isotopes_rdkit_molecule(mol)
        # For Sanity Double Check
        smiles = MolToSmiles(mol, canonical=True)
        mol = MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return smiles

    except Exception as e:
        if logger:
            logger.error(f"StandardizationError: {e} for {og_smiles}")
        if suppress_exception:
            return smiles_inter
        else:
            return None


# check_smiles_type : bool, optional
#         A boolean flag to check if the input is a string or None. If True and the input is not a
#         string, a TypeError is raised or logged, depending on the value of logger. Default is True.
#     Raises
#     ------
#     TypeError
#         If check_smiles_type is True and the input is not a string.
#     StandardizationError
#         If an unexpected error occurs during standardization and suppress_exception is False.
#         The error message is logged or raised, depending on the value of logger.


def standardize_wrapper(args):
    """
    Wrapper function for the standardize function to be used with the concurrent.futures.ProcessPoolExecutor.
    """
    return standardize(*args)


def rdkit_standardize_wrapper(args):
    """
    Wrapper function for the rdkit_standardize function to be used with the concurrent.futures.ProcessPoolExecutor.
    """
    return rdkit_standardize(*args)


def parallel_standardize(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    logger=None,
    suppress_exception=True,
    rd_standardize=False,
):
    # standardizing the SMILES in parallel
    standardizer = rdkit_standardize_wrapper if rd_standardize else standardize_wrapper
    unique_smiles = df[smiles_col].unique()
    args_list = [(smi, logger, suppress_exception) for smi in unique_smiles]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(
            tqdm(
                executor.map(standardizer, args_list),
                total=len(args_list),
                desc="Standardizing Unique SMILES",
            )
        )
    standardized_result = {smi: result for smi, result in zip(unique_smiles, results)}

    # Apply the standardized result to the dataframe
    df[smiles_col] = df[smiles_col].map(standardized_result)

    return df


def standardize_df(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    other_dup_col: Union[List[str], str] = None,
    sorting_col: str = "",
    drop: bool = True,
    keep: Union[bool, str] = "last",
    logger=None,
    suppress_exception=True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applies a standardization workflow to the 'smiles' column of a pandas dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe, which should contain a 'smiles' column.

    smiles_col : str, optional
        The name of the column containing the SMILES strings to standardize. Default is 'smiles'.

    other_dup_col : str or list of str, optional
        The name of the column(s) containing other information that should be kept for duplicate SMILES.
        If None, no other columns are kept. Default is None.

    sorting_col : str, optional
        The name of the column to sort the dataframe by before standardization. If None, the dataframe is not sorted.
        Default is None.

    drop : bool, optional
        A boolean flag to drop the rows with NaN SMILES before standardization. Default is True.

    keep : bool or str, optional
        A boolean flag to keep the first or last duplicate SMILES. If True, the first duplicate is kept.
        If False, the last duplicate is kept. If 'aggregate', the duplicates are aggregated into a list.
        Default is 'last'.

    logger : logging.Logger, optional
        A logger object to log error messages. Default is None.

    suppress_exception : bool, optional
        A boolean flag to suppress exceptions and return the original SMILES string if an error
        occurs during standardization. If False, an exception is raised or logged, depending on the value of logger.
        Default is True.

    Returns:
    --------
    pandas.DataFrame
        A new dataframe with the 'smiles' column replaced by the standardized versions.

    Notes:
    ------
    This function applies the `standardize` function to each SMILES string in the 'smiles' column
    of the input dataframe,
    and replaces the column with the standardized versions.
    """
    if keep == "aggregate":
        keep = False
        aggregate = True
    else:
        aggregate = False

    if other_dup_col:
        if not isinstance(other_dup_col, list):
            other_dup_col = [other_dup_col]
        cols_dup = [smiles_col, *other_dup_col]
    else:
        cols_dup = smiles_col

    # checking NaN & duplicate before standardization
    df_filtered, df_nan_before, df_dup_before = check_nan_duplicated(
        df=df,
        cols_nan=smiles_col,
        cols_dup=cols_dup,  # [smiles_col, *other_dup_col],
        nan_dup_source="smiles_before_std",
        drop=drop,
        sorting_col=sorting_col,
        keep=keep,
        logger=logger,
    )
    if logger:
        logger.info(
            f"BEFORE SMILES standardization, The number of filtered-out NaN values"
            f"is: {df_nan_before.shape[0]} NaN values"
            f"While The number of points that were found to be duplicates"
            f"is: {df_dup_before.shape[0]} duplicated rows"
        )

    df_filtered = parallel_standardize(df_filtered, smiles_col, logger, suppress_exception)
    # standardizing the SMILES in parallel
    # tqdm.pandas(desc="Standardizing SMILES")
    # unique_smiles = df_filtered[smiles_col].unique()
    # args_list = [(smi, logger, suppress_exception) for smi in unique_smiles]
    #
    # with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    #     results = list(
    #         tqdm(
    #             executor.map(standardize_wrapper, args_list),
    #             total=len(args_list),
    #             desc="Standardizing Unique SMILES",
    #         )
    #     )
    # standardized_result = {smi: result for smi, result in zip(unique_smiles, results)}
    #
    # # Apply the standardized result to the dataframe
    # df_filtered[smiles_col] = df_filtered[smiles_col].map(standardized_result)

    # # progress_apply is a wrapper around apply that uses tqdm to show a progress bar
    # start_time = time.time()
    # tqdm.pandas(desc="Standardizing SMILES")
    # # df_filtered smiles standardization
    # df_filtered[smiles_col] = df_filtered[smiles_col].progress_apply(
    #     standardize, logger=logger, suppress_exception=suppress_exception
    # )
    # # df_dup_before smiles standardization
    # df_dup_before[smiles_col] = df_dup_before[smiles_col].progress_apply(
    #     standardize, logger=logger, suppress_exception=suppress_exception
    # )
    #
    # if logger:
    #     logger.info(
    #         "SMILES standardization took --- %s seconds ---"
    #         % (time.time() - start_time)
    #     )

    # checking NaN & duplicate after standardization
    df_filtered, df_nan_after, df_dup_after = check_nan_duplicated(
        df=df_filtered,
        cols_nan=smiles_col,
        cols_dup=cols_dup,  # [smiles_col, *other_dup_col],
        nan_dup_source="smiles_after_std",
        drop=drop,
        sorting_col=sorting_col,
        keep=keep,
        logger=logger,
    )

    if logger:
        logger.info(
            f"After SMILES standardization, The number of additional NaN values (failed standardization) "
            f"is: {df_nan_after.shape[0]} NaN values"
            f"While The number of points that were found to be duplicates after standardization "
            f"is: {df_dup_after.shape[0]} duplicated rows"
        )

    # concat the nan and dup dataframes
    df_nan = pd.concat([df_nan_before, df_nan_after])
    df_dup = pd.concat([df_dup_before, df_dup_after])

    if aggregate:
        # aggregate the duplicates
        df_dup = (
            df_dup.groupby(smiles_col, as_index=False).agg(custom_agg).reset_index()
        )

    return df_filtered, df_nan, df_dup


# define function that transforms SMILES strings into ECFPs
def ecfp_from_smiles(
    smiles, radius=4, length=2**10, use_features=False, use_chirality=False
):
    """
    Generates an ECFP (Extended Connectivity Fingerprint) from a SMILES string.

    Parameters:
    -----------
    smiles : str
        The input SMILES string to generate a fingerprint from.
    radius : int, optional
        The radius of the circular substructure (in bonds) to use when generating the fingerprint.
        Default is 2.
    length : int, optional
        The length of the output fingerprint in bits. Default is 2^10.
    use_features : bool, optional
        Whether to use feature-based fingerprints instead of circular fingerprints.
        Default is False (i.e., use circular fingerprints).
    use_chirality : bool, optional
        Whether to include chirality information in the fingerprint. Default is False.

    Returns:
    --------
    numpy.ndarray
        The ECFP fingerprint as a binary numpy array.

    Notes:
    ------
    This function uses the RDKit library for generating ECFP fingerprints.
    source:
    https://www.blopig.com/blog/2022/11/how-to-turn-a-smiles-string-into-an-extended-connectivity-fingerprint-using-rdkit/
    """
    smiles = (
        smiles[0] if isinstance(smiles, list) or isinstance(smiles, tuple) else smiles
    )
    molecule = AllChem.MolFromSmiles(smiles)
    feature_list = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius=radius,
        nBits=length,
        useFeatures=use_features,
        useChirality=use_chirality,
    )
    return np.array(feature_list)


def wrapper_ecfp_from_smiles(args):
    return ecfp_from_smiles(*args)


def generate_ecfp(
    smiles,
    radius=4,
    length=2**10,
    use_features=False,
    use_chirality=False,
) -> dict[Any, Any]:
    """
    Generates ECFP fingerprints from the 'smiles' column of a pandas dataframe.

    Parameters:
    -----------
    smiles : List[str] or ndarray or Series
        The input SMILES strings to calculate ECFP fingerprints from.
    radius : int, optional
        The radius of the circular substructure (in bonds) to use when generating the fingerprint.
        Default is 2.
    length : int, optional
        The length of the output fingerprint in bits. Default is 2^10.
    use_features : bool, optional
        Whether to use feature-based fingerprints instead of circular fingerprints.
        Default is False (i.e., use circular fingerprints).
    use_chirality : bool, optional
        Whether to include chirality information in the fingerprint. Default is False.

    Returns:
    --------
    ecfp_result : dict
        A dictionary containing the ECFP fingerprints for each SMILES string.
        to be used as df[smiles_col].map(ecfp_result)
        to add the fingerprints to the dataframe

    Notes:
    ------
    This function applies the `ECFP_from_smiles` function to each
    SMILES string in the 'smiles' column of the input dataframe,
    and generates ECFP fingerprints with the specified radius, length,
    and optional parameters. The resulting fingerprints are stored in columns named 'ECFP-{length}',
    where {length} is the specified fingerprint length.
    """
    # Generate ECFP fingerprints
    # for length in [2 ** i for i in range(5, 12)]:
    args_list = [(smi, radius, length, use_features, use_chirality) for smi in smiles]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(
            tqdm(
                executor.map(wrapper_ecfp_from_smiles, args_list),
                total=len(args_list),
                desc=f"Generating ECFP {length} fingerprints",
            )
        )
    ecfp_result = {smi: result for smi, result in zip(smiles, results)}
    # This should be used as
    # df_filtered[smiles_col] = df_filtered[smiles_col].map(standardized_result)
    return ecfp_result


def get_mol_descriptors(smiles: str, chosen_descriptors: List[str] = None):
    """
    Calculates a set of molecular descriptors for a given SMILES string.

    Parameters
    ----------
    smiles : str
        The SMILES string to calculate the descriptors for.

    chosen_descriptors : list of str, optional
        The list of descriptors to calculate. If None, all 200 descriptors will be calculated.
        Default is None.

    Returns
    -------
    list of float
        The calculated descriptor values, in the order of the chosen_descriptors list.

    source:
    https://www.blopig.com/blog/2022/06/how-to-turn-a-molecule-into-a-vector-of-physicochemical-descriptors-using-rdkit/
    """

    smiles = (
        smiles[0] if isinstance(smiles, list) or isinstance(smiles, tuple) else smiles
    )

    # convert SMILES string to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)

    # choose 200 molecular descriptors
    if chosen_descriptors is None:
        chosen_descriptors = descriptors

    # create molecular descriptor calculator
    mol_descriptor_calculator = MolecularDescriptorCalculator(chosen_descriptors)

    # use molecular descriptor calculator on RDKit mol object
    desc_array = np.array(mol_descriptor_calculator.CalcDescriptors(mol))

    return desc_array


def wrapper_get_mol_descriptors(args):
    return get_mol_descriptors(*args)


def generate_mol_descriptors(
    smiles, chosen_descriptors: List[str] = None
) -> dict[Any, Any]:
    """
    Applies the `mol_descriptors` function to a pandas dataframe and returns a new dataframe
    with additional columns containing the calculated descriptor values.

    Parameters
    ----------
    smiles: List[str] or ndarray or Series
        The input SMILES strings to calculate molecular descriptors from.
    chosen_descriptors : list of str, optional
        The list of descriptors to calculate. If None, the default list of descriptors in
        `mol_descriptors` will be used.

    Returns
    -------
    mol_desc_result : dict
        A dictionary containing the molecular descriptors for each SMILES string.
        to be used as df[smiles_col].map(mol_desc_result)
        to add the descriptors to the dataframe
    """

    args_list = [(smi, chosen_descriptors) for smi in smiles]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(
            tqdm(
                executor.map(wrapper_get_mol_descriptors, args_list),
                total=len(args_list),
                desc="Generating Molecular Descriptors",
            )
        )

    mol_desc_result = {smi: result for smi, result in zip(smiles, results)}

    return mol_desc_result

    # # apply mol_descriptors() to the 'smiles' column using the .apply() method
    # calc_descriptors = new_df[smiles_col].apply(
    #     get_mol_descriptors, chosen_descriptors=chosen_descriptors
    # )

    # # convert the list of descriptor values to a DataFrame with separate columns
    # descriptor_df = pd.DataFrame(
    #     calc_descriptors, columns=chosen_descriptors
    # )  # .tolist()
    #
    # # concatenate the new DataFrame with the original DataFrame
    # # new_df = pd.concat([new_df, descriptor_df], axis=1)
    # # merge the new DataFrame with the original DataFrame
    # new_df = pd.merge(
    #     new_df,
    #     descriptor_df,
    #     left_index=True,
    #     right_index=True,
    #     how="left",
    #     on=None,
    #     validate="many_to_many",
    # )
    # return new_df

    #
    # # create a new dataframe with the same columns as the input dataframe plus the descriptor columns
    # # new_columns = [f'{desc}_mol_desc' for desc in descriptors]
    # # new_df = pd.concat([df, pd.DataFrame(columns=new_columns)])
    # new_df = pd.concat([df, pd.DataFrame(columns=chosen_descriptors)])
    #
    # # apply the mol_descriptors function to each SMILES string and fill in the new dataframe
    # new_df[[descriptors]] = new_df[smiles_col].apply(mol_descriptors, descriptors)
    #
    # for i, row in df.iterrows():
    #     smi = row[column_name]
    # descriptor_vals = mol_descriptors(smi)
    # new_df.loc[i, descriptors] = descriptor_vals
    #
    # return new_df


def get_papyrus_descriptors(connectivity_ids=None, desc_type="cddd", logger=None):
    # "mold2", "mordred", "cddd", "fingerprint", "moe", "all"
    def _merge_cols(row):
        return np.array(row[1:])

    mol_descriptors = read_molecular_descriptors(
        desc_type=desc_type,
        is3d=False,
        version="latest",
        chunksize=100000,
        source_path=DATA_DIR,
        ids=connectivity_ids,
        verbose=True,
    )
    if logger:
        logger.info(f"Loading Papyrus {desc_type} descriptors...")

    mol_descriptors = consume_chunks(mol_descriptors, progress=True, total=60)
    # # get only the connectivity ids that are from the input connectivity_ids list
    # mol_descriptors = mol_descriptors[
    #     mol_descriptors["connectivity"].isin(connectivity_ids)
    # ]

    mol_descriptors[desc_type] = mol_descriptors.apply(_merge_cols, axis=1)

    mol_descriptors = mol_descriptors[["connectivity", desc_type]]

    mol_descriptors_mapper = mol_descriptors.set_index("connectivity")[
        desc_type
    ].to_dict()

    return mol_descriptors_mapper


def get_chem_desc(
    df, desc_type: str = "ecfp1024", query_col: str = "SMILES", logger=None, **kwargs
) -> pd.DataFrame:
    desc_type = desc_type.lower()
    unique_entries = df[query_col].unique().tolist()

    if desc_type is None:
        return df
    elif desc_type in ["mold2", "mordred", "cddd", "fingerprint"]:  # , "moe"
        desc_mapper = get_papyrus_descriptors(
            connectivity_ids=unique_entries, desc_type=desc_type, logger=logger
        )
    elif desc_type.startswith("ecfp"):
        length = int(desc_type[4:])
        desc_mapper = generate_ecfp(unique_entries, radius=4, length=length, **kwargs)
    elif desc_type == "moldesc":  # errorness
        desc_mapper = generate_mol_descriptors(unique_entries, **kwargs)
    elif desc_type == "graph2d":
        raise NotImplementedError
    else:
        raise ValueError(f"desc_mol: {desc_type} is not a valid molecular descriptor")

    df[desc_type] = df[query_col].map(desc_mapper)

    return df


def mol_to_pil_image(
    molecule: Chem.rdchem.Mol, width: int = 300, height: int = 300
) -> Image:
    """
    Converts an RDKit molecule to a PIL image.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The RDKit molecule to convert.
    width : int, optional
        The width of the image in pixels. Default is 300.
    height : int, optional
        The height of the image in pixels. Default is 300.

    Returns
    -------
    PIL.Image
        The PIL image.

    source: https://www.rdkit.org/docs/Cookbook.html
    """
    Chem.AllChem.Compute2DCoords(molecule)
    Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
    pil_image = Draw.MolToImage(molecule, size=(width, height))
    return pil_image


def smi_to_pil_image(smiles: str, width: int = 300, height: int = 300) -> Image:
    """
    Converts an RDKit molecule to a PIL image.

    Parameters
    ----------
    smiles : str
        The SMILES string to convert.
    width : int, optional
        The width of the image in pixels. Default is 300.
    height : int, optional
        The height of the image in pixels. Default is 300.

    Returns
    -------
    PIL.Image
        The PIL image.

    source: https://www.rdkit.org/docs/Cookbook.html
    """
    molecule = Chem.MolFromSmiles(smiles)
    Chem.AllChem.Compute2DCoords(molecule)
    Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
    pil_image = Draw.MolToImage(molecule, size=(width, height))
    return pil_image


def generate_scaffold(smiles, include_chirality=False):
    """
    calculates the Bemis-Murcko scaffold for a SMILES string.

    Parameters
    ----------
    smiles : str
        The SMILES string.
    include_chirality : bool, optional
        Whether to include chirality in the scaffold. Default is False.

    Returns
    -------
    str
        The scaffold SMILES string.
    """
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )

    except Exception as e:
        scaffold = None
        print(f"following error {e} \n occurred while processing smiles: {smiles}")

    if scaffold is None or scaffold == "":
        scaffold = smiles

    return scaffold


def merge_scaffolds(df, smiles_col="SMILES"):
    """
    Merges the scaffold information into the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    smiles_col : str, optional
        The name of the column containing the SMILES strings. Default is 'smiles'.
    verbose : bool, optional
        If True, prints the progress of the scaffold generation. Default is False.
    output_path : str, optional
        The path to the output directory for scaffold clustering figures. Default is None.
    Returns
    -------

    """
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

    # standardize the scaffold column
    df = parallel_standardize(df, "scaffold", None, True)

    return df


### adopted from https://github.com/nina23bom/NPS-Pharmacological-profile-fingerprint-prediction-using-ML/blob/main/001.%20NPS%20unique%20compounds%20MCS%20Hierarchical%20clustering%20-%20Class%20Label.ipynb
def tanimoto_mcs(smi1, smi2):
    # reading smiles of two molecules and create molecule
    m1 = Chem.MolFromSmiles(smi1)
    m2 = Chem.MolFromSmiles(smi2)
    mols = [m1, m2]

    # number heavy atoms of both molecules
    a = m1.GetNumAtoms()
    b = m2.GetNumAtoms()
    # print(a,b)
    # find heavy atoms in MCS
    r = rdFMCS.FindMCS(
        mols,
        ringMatchesRingOnly=True,
        bondCompare=Chem.rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareAny,
        timeout=1,
    )
    c = r.numAtoms
    # print(c)
    if c < 0:
        c = 0
    mcs_tani = c / (a + b - c)
    # get MCS smart
    # mcs_sm = r.smartsString
    # print(mcs_sm)
    return mcs_tani


def tanimoto_mcs_withH(smi1, smi2):
    # reading smiles of two molecules and create molecule
    m1 = Chem.MolFromSmiles(smi1)
    m2 = Chem.MolFromSmiles(smi2)

    m1H = Chem.AddHs(m1)
    m2H = Chem.AddHs(m2)
    mols = [m1H, m2H]

    # number heavy atoms of both molecules
    a = m1H.GetNumAtoms()
    b = m2H.GetNumAtoms()
    # print(a,b)
    # find heavy atoms in MCS
    r = rdFMCS.FindMCS(
        mols,
        ringMatchesRingOnly=True,
        bondCompare=Chem.rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareAny,
        timeout=1,
    )
    c = r.numAtoms
    # print(c)
    if c < 0:
        c = 0
    mcs_tani = c / (a + b - c)
    # get MCS smart
    # mcs_sm = r.smartsString
    # print(mcs_sm)
    return mcs_tani


def tanimoto_mcs_wrapper(index_pair, cid_list):
    cid1, cid2 = index_pair
    return tanimoto_mcs(cid_list[cid1], cid_list[cid2])


def tanimoto_mcs_withH_wrapper(index_pair, cid_list):
    cid1, cid2 = index_pair
    return tanimoto_mcs_withH(cid_list[cid1], cid_list[cid2])


def chunked_iterable(n, size):
    """
    A generator to yield chunks of index pairs for all unique combinations.

    Parameters
    ----------
    n : int
        The total number of compounds.
    size : int
        The size of each chunk.

    Yields
    ------
    List of tuple
        Each yielded chunk is a list of index pairs.
    """
    iterable = combinations(range(n), 2)
    while True:
        chunk = list(islice(iterable, size))
        if not chunk:
            return
        yield chunk


def calculate_total_chunks(n_compounds, batch_size):
    """
    Calculate the total number of chunks for the tqdm progress bar.
    """
    total_pairs = n_compounds * (n_compounds - 1) / 2
    total_chunks = math.ceil(total_pairs / batch_size)
    return total_chunks


def process_chunk(chunk, similarity_matrix, cid_list, tanimoto_mcs_func):
    # tn_mcs_func = tanimoto_mcs_withH_wrapper if withH else tanimoto_mcs_wrapper
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(tanimoto_mcs_func, pair, cid_list): pair for pair in chunk
        }
        for future in futures:
            similarity = future.result()
            i, j = futures[future]
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity


def save_similarity_matrix(matrix, filename="similarity_matrix.npy"):
    """
    Save the similarity matrix to a file.

    Parameters:
    - matrix: The similarity matrix to be saved.
    - filename: The filename for the saved matrix. Default is "similarity_matrix.npy".
    """
    np.save(filename, matrix)
    print(f"Similarity matrix saved to {filename}")


def hierarchical_clustering(
    df,
    smiles_col: str = "SMILES",
    batch_size=10000,
    withH=False,
    save_path=None,
):
    """
    Perform hierarchical clustering on a DataFrame of SMILES strings.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the SMILES strings.
    smiles_col : str, optional
        The name of the column containing the SMILES strings. Default is 'SMILES'.
    batch_size : int, optional
        The size of each chunk for processing the similarity matrix. Default is 10000.
    withH : bool, optional
        A boolean flag to include hydrogens in the MCS calculation. Default is False.
    save_path : str, optional
        The path to save the similarity matrix or to check for preprocessed files. Default is None.

    Returns
    -------
    numpy.ndarray
        The similarity matrix.
    """
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        filepath = Path(save_path) / "mcs.pkl.npy"
        # now checking if file exists
        if filepath.exists():
            print(f"Similarity matrix already exists at {filepath}")
            return load_npy_file(filepath)

    print(f"Chunk Size: {BATCH_SIZE}")
    print(f"Number of Workers: {N_WORKERS}")

    cid_list = df[smiles_col].tolist()
    n_compounds = len(cid_list)
    print(f"Number of unique {smiles_col} for clustering: {df.shape[0]}")

    # Initialize the similarity matrix
    similarity_matrix = np.zeros((n_compounds, n_compounds), dtype="float16")
    # Calculate total chunks for tqdm progress bar
    total_chunks = calculate_total_chunks(n_compounds, batch_size)
    tanimoto_mcs_func = tanimoto_mcs_withH_wrapper if withH else tanimoto_mcs_wrapper
    # Process the pairs in chunks
    for chunk in tqdm(
        chunked_iterable(n_compounds, batch_size),
        desc="Calculating similarities in chunks",
        unit="chunk",
        total=total_chunks,
    ):
        process_chunk(chunk, similarity_matrix, cid_list, tanimoto_mcs_func)

    np.fill_diagonal(similarity_matrix, 1.0)

    # Save the similarity matrix to the specified path
    if save_path is not None:
        save_npy_file(similarity_matrix, filepath)
        # file_path = Path(save_path) / "mcs.npy"
        # save_similarity_matrix(similarity_matrix, save_path)

    return similarity_matrix


def form_linkage(X, save_path=None, calculate_cophenetic_coeff=False):
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        filepath = Path(save_path) / "mcs_linkage.pkl.npy"
        # now checking if file exists
        if filepath.exists():
            print(f"Linkage matrix already exists at {filepath}")
            return load_npy_file(filepath)
    Z = linkage(X, method="ward") # TODO : save and load if existing
    if save_path is not None:
        save_npy_file(Z, filepath)
    if calculate_cophenetic_coeff:
        calculate_cophenet(X, Z, save_path=save_path)
    return X, Z


def calculate_cophenet(X, Z, save_path=None):
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        Pdist_path = Path(save_path) / "mcs_pdist.pkl.npy"
        Coph_dists_path = Path(save_path) / "mcs_coph_dists.pkl.npy"
        C_path = Path(save_path) / "mcs_c.pkl"
        # now checking if file exists
        if C_path.exists():
            print(f"Cophenetic Coefficient already exists at {C_path}")
            return load_pickle_file(C_path)

    Pdist = pdist(X)
    c, coph_dists = cophenet(Z, Pdist)
    print("Cophenetic coefficient calculated: %0.4f" % c)
    if save_path is not None:
        save_npy_file(Pdist, Pdist_path)
        save_npy_file(coph_dists, Coph_dists_path)
        save_pickle_file(c, C_path)
    return c


def sil_K(X, Z, max_k=100):
    sil, n_clu = [], []
    for k in range(2, max_k):
        cluster = fcluster(Z, k, criterion="maxclust")
        silhouette_avg = silhouette_score(X, cluster)
        sil.append(silhouette_avg)
        n_clu.append(k)

    optimal_clu = n_clu[sil.index(max(sil))]
    print("Optimal number of clusters: ", optimal_clu)

    return n_clu, sil, optimal_clu


def plot_silhouette_analysis(cluster_counts, silhouette_scores, output_path=None):
    """
    Plots the silhouette analysis for determining the optimal number of clusters.

    Parameters:
    -----------
    - cluster_counts : list
        A list of integers representing the number of clusters for each silhouette score series.
    - silhouette_scores : list
        A list of floats representing the average silhouette scores for each series. Each series corresponds to a different number of clusters.
    - labels : list
        A list of strings representing the labels for each series.
    - output_path : str
        The path where the plot image will be saved.
    """
    # Initialize the plot
    fig = plt.figure(figsize=(12, 5), dpi=600)
    plt.rc("font", family="serif")

    plt.scatter(cluster_counts, silhouette_scores, label="MCS")
    # # Plot each series of cluster counts vs. silhouette scores
    # for i in range(len(cluster_counts)):
    #     plt.scatter(cluster_counts[i], silhouette_scores[i], label=labels[i])

    # Adding plot details
    plt.legend(loc="lower right", shadow=True, fontsize=16)
    plt.xlabel("Number of Clusters", fontsize=16)
    plt.ylabel("Average Silhouette Score", fontsize=16)

    # Show and save the plot
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            Path(output_path)
            / "Silhouette_analysis_for_determining_optimal_clusters_K.png",
            bbox_inches="tight",
        )

    plt.show()


def plot_cluster_heatmap(data_matrix, output_path=None):
    if hasattr(data_matrix, 'index'):
        yticklabels = data_matrix.index
    else:
        # For numpy arrays, create numeric labels for each row
        yticklabels = range(data_matrix.shape[0])
    # yticklabels = data_matrix.index
    plt.figure(figsize=(12, 30), dpi=600)
    plt.rc("font", family="serif", size=8)
    sns.set_style("white")

    # Generate the clustermap
    fig = sns.clustermap(
        data_matrix,
        method="ward",
        cmap="coolwarm",
        fmt="d",
        linewidth=0.5,
        xticklabels=False,
        yticklabels=yticklabels,
        figsize=(12, 20),
    )

    # Save the plot to the specified output path
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            Path(output_path) / "Heatmap_of_the_clustering.png",
            dpi=600,
            bbox_inches="tight",
        )

    # Show the plot
    plt.show()


def clustering(
    df,
    smiles_col: str = "scaffold",
    max_k=100,
    withH=False,
    # fig_output_path=None,
    export_mcs_path=None,
):
    # pre cleaning
    df_clean = df.copy()[[smiles_col]]
    # dropp duplicates to avoid self comparison and reset index
    df_clean.drop_duplicates(subset=smiles_col, keep="first", inplace=True)

    df_clean.dropna(subset=[smiles_col], inplace=True)
    # print(f"after nan drop: {df_clean.shape}")
    df_clean.reset_index(inplace=True, drop=True)

    # TODO : only 10 scaffolds for testing
    # TODO : checking mcs file existing then loading it instead of recalculation

    mcs_np = hierarchical_clustering(
        df_clean,
        smiles_col=smiles_col,
        batch_size=BATCH_SIZE,
        withH=withH,
        save_path=export_mcs_path,
    )
    # if export_mcs_path:
    #     Path(export_mcs_path).mkdir(parents=True, exist_ok=True)
    #     df_mcs.to_csv(Path(export_mcs_path) / "mcs_matrix.csv", index=True)
    # df_pair.to_csv(Path(export_mcs_path) / "scaffold_sim_pair.csv", index=True)
    mcs_x, mcs_z = form_linkage(mcs_np)
    mcs_k, mcs_sil, optimal_clu = sil_K(mcs_x, mcs_z, max_k=max_k)
    if export_mcs_path:
        fig_output_path = Path(export_mcs_path) / "mcs_figures"
        Path(fig_output_path).mkdir(parents=True, exist_ok=True)
        plot_silhouette_analysis(mcs_k, mcs_sil, output_path=fig_output_path)
        plot_cluster_heatmap(mcs_np, output_path=fig_output_path)

    df_clean["cluster"] = fcluster(mcs_z, optimal_clu, criterion="maxclust")

    # now we map the cluster to the original dataframe
    df = pd.merge(df, df_clean, on=smiles_col, how="left", validate="many_to_many")

    return df

#
# def (df):
#
#     fig = plt.figure(figsize=(12,5), dpi = 600)
#     plt.rc('font', family='serif')
#     #plt.scatter(MACCS_K, MACCS_sil,label="MACCS")
#     plt.scatter(MCS_K, MCS_sil,label="MCS")
#     #plt.scatter(AA_MCS_K, AA_MCS_sil,label="all-atom MCS")
#
#     plt.legend()
#     legend = plt.legend(loc='lower right', shadow=True, fontsize=16)
#     plt.xlabel("Number of clusters", fontsize=16)
#     plt.ylabel("Average Silhouette Score", fontsize=16)
#     plt.show()
#     fig.savefig(output_path+"Silhouette analysis for determining optimal clusters K.png", bbox_inches='tight')

# def hierarchical_clustering(
#     df,
#     smiles_col: str = "SMILES",
#     method: str = "average",
#     metric: str = "jaccard",
#     threshold: float = 0.7,
#     plot: bool = True,
#     figsize: Tuple[int, int] = (10, 10),
#     save_path: str = None,
#     logger=None,
# ):
#     """
#     Performs hierarchical clustering on a dataframe of SMILES strings.
#
#     Parameters
#     ----------
#     df : pandas.DataFrame
#         The input dataframe, which should contain a 'smiles' column.
#
#     smiles_col : str, optional
#         The name of the column containing the SMILES strings to cluster. Default is 'smiles'.
#
#     method : str, optional
#         The linkage method to use for hierarchical clustering. Default is 'average'.
#
#     metric : str, optional
#         The distance metric to use for hierarchical clustering. Default is 'jaccard'.
#
#     threshold : float, optional
#         The threshold value to use for cutting the dendrogram. Default is 0.7.
#
#     plot : bool, optional
#         Whether to plot the dendrogram. Default is True.
#
#     figsize : tuple of int, optional
#         The size of the plot in inches. Default is (10, 10).
#
#     save_path : str, optional
#         The path to save the plot to. Default is None.
#
#     Returns
#     -------
#     numpy.ndarray
#         The cluster labels for each SMILES string.
#
#     source: https://www.rdkit.org/docs/Cookbook.html
#     """
#     # Calculate the distance matrix
#     fps = [
#         Chem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=2048)
#         for x in df[smiles_col]
#     ]
#     dists = []
#     nfps = len(fps)
#     for i in range(1, nfps):
#         sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
#         dists.extend([1 - x for x in sims])
#
#     # Perform hierarchical clustering
#     X = np.array(dists)
#     X = X.reshape(-1, 1)
#     Z = linkage(X, method=method, metric=metric)
#
#     # Cut the dendrogram at the threshold value
#     clusters = fcluster(Z, t=threshold, criterion="distance")
#
#     # Plot the dendrogram
#     if plot:
#         plt.figure(figsize=figsize)
#         dendrogram(Z)
#         plt.title(f"Hierarchical Clustering of SMILES Strings")
#         plt.xlabel("Index")
#         plt.ylabel("Distance")
#         if save_path:
#             plt.savefig(save_path)
#         plt.show()
#
#     return clusters


# cid_idx = {cid: i for i, cid in enumerate(cid_list)}
# Generate all unique pairs for similarity calculation to avoid redundancy
# rows, cols, data = [], [], []
# total_pairs = int(len(cid_list) * (len(cid_list) - 1) / 2)
# pairs = combinations(cid_list, 2)
# index_pairs = list(combinations(range(n_compounds), 2))
# print(f"Number of unique pairs for clustering: {len(index_pairs)}")

# Split index pairs into chunks
# chunks = [
#     index_pairs[i : i + batch_size] for i in range(0, len(index_pairs), batch_size)
# ]

# Initialize a memory-mapped file to store similarities
# similarity_matrix = np.memmap(
#     filename, dtype="float32", mode="w+", shape=(total_pairs,)
# )

# Process pairs in chunks and write to memmap file
# start_idx = 0
#
# # Function to process a chunk of pairs
# def process_chunk(chunk):
#     with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
#         chunk_results = list(executor.map(tanimoto_mcs_wrapper, chunk))
#     return chunk_results
#
# start_idx = 0
# # Process the pairs in chunks
# for chunk in tqdm(
#     chunked_iterable(pairs, batch_size), total=total_pairs // batch_size
# ):
#     chunk_results = process_chunk(chunk)
#     end_idx = start_idx + len(chunk_results)
#     s
#
#     for cid1, cid2, similarity in process_chunk(chunk):
#         rows.append(cid_idx[cid1])
#         cols.append(cid_idx[cid2])
#         data.append(similarity)
#
# # Create sparse matrix and save
# matrix = coo_matrix((data, (rows, cols)), shape=(len(cid_list), len(cid_list)))
# save_sparse_matrix(filename, matrix)

# return matrix.toarray()

# df_cid.loc[cid1, cid2] = similarity
# df_cid.loc[cid2, cid1] = similarity

# for cid in names_list:
#     cidx = cid_idx.index(cid)
#     df_cid.iloc[cidx, cidx] = 1.0  # Self-similarity is 1
#     max_sim_idx = df_cid.iloc[cidx].idxmax()
#     max_sim_value = df_cid.loc[cid, max_sim_idx]
#     df_pair.loc[cid, "Pair"] = max_sim_idx
#     df_pair.loc[cid, "MaxValue"] = max_sim_value

#
# def __hierarchical_clustering(df, smiles_col: str = "SMILES"):  # , names_col=None
#     cid_idx = df.index.tolist()
#     cid_list = df[smiles_col].tolist()
#     print(f"Number of unique {smiles_col} for clustering: {df.shape[0]}")
#
#     # Generate all unique pairs for similarity calculation to avoid redundancy
#     pairs = list(combinations(cid_list, 2))
#     print(f"Number of unique pairs for clustering: {len(pairs)}")
#
#     # Initialize the DataFrame to hold Tanimoto similarities
#     df_cid = pd.DataFrame(0.0, index=cid_idx, columns=cid_idx)
#
#     print("Pooling the Tanimoto Similarity Calculation...")
#     # Parallel calculation of Tanimoto similarities
#     with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
#         results = list(
#             tqdm(
#                 executor.map(tanimoto_mcs_wrapper, pairs),
#                 total=len(pairs),
#                 desc="Calculating Tanimoto Similarity",
#             )
#         )
#
#     for cid1, cid2, similarity in results:
#         # we want to get index of the cid in the df
#         idx1 = cid_idx.index(cid1)
#         idx2 = cid_idx.index(cid2)
#         df_cid.iloc[idx1, idx2] = similarity
#         df_cid.iloc[idx2, idx1] = similarity
#         df_cid.iloc[idx1, idx1] = 1.0  # Self-similarity is 1
#         df_cid.iloc[idx2, idx2] = 1.0  # Self-similarity is 1
#
#     return df_cid  # , df_pair

# df_cid.loc[cid1, cid2] = similarity
# df_cid.loc[cid2, cid1] = similarity

# for cid in names_list:
#     cidx = cid_idx.index(cid)
#     df_cid.iloc[cidx, cidx] = 1.0  # Self-similarity is 1
#     max_sim_idx = df_cid.iloc[cidx].idxmax()
#     max_sim_value = df_cid.loc[cid, max_sim_idx]
#     df_pair.loc[cid, "Pair"] = max_sim_idx
#     df_pair.loc[cid, "MaxValue"] = max_sim_value


# def _hierarchical_clustering(df, smiles_col: str = "SMILES", names_col=None):
#     cid_list = df[smiles_col].tolist()
#     if names_col:
#         names_list = df[names_col].tolist()
#     else:
#         names_list = cid_list
#     list_len = df.shape[0]
#
#     df_cid = pd.DataFrame(0.0, index=names_list, columns=names_list)
#     df_pair = pd.DataFrame(0.0, index=names_list, columns=["Pair", "MaxValue"])
#     df_pair["Pair"] = df_pair["Pair"].astype(str)
#
#     # df_cid = pd.DataFrame(0.0, index=cid_list, columns=cid_list)
#     # df_pair = pd.DataFrame(0.0, index=cid_list, columns=["Pair", "MaxValue"])
#     # for loop with tqdm to show progress bar
#     for i in tqdm(range(list_len), desc="Calculating Tanimoto Similarity"):
#         df_cid.iloc[i, i] = 1.0
#         for j in range(i + 1, list_len):
#             df_cid.iloc[i, j] = tanimoto_mcs(cid_list[i], cid_list[j])
#             df_cid.iloc[j, i] = df_cid.iloc[i, j]
#         cid = names_list[i]
#         tmpInd = df_cid.loc[cid, cid != df_cid.columns].idxmax()
#         tmpValue = df_cid.loc[cid, tmpInd]
#         df_pair.iloc[i, 0] = tmpInd
#         df_pair.iloc[i, 1] = tmpValue
#     return df_cid, df_pair


# def chunks(pairs, batch_size):
#     """
#     Splits a list of pairs into chunks of a specified size.
#
#     Parameters
#     ----------
#     pairs : list of tuple
#         The list of pairs to split.
#     batch_size : int
#         The size of each chunk.
#
#     Yields
#     ------
#     list of tuple
#         A chunk of pairs.
#     """
#     for i in range(0, len(pairs), batch_size):
#         yield pairs[i : i + batch_size]


# def chunked_iterable(iterable, size):
#     """
#     Take an iterable and yield chunks of a given size.
#
#     Parameters
#     ----------
#     iterable : iterable
#         The iterable to be chunked.
#     size : int
#         The size of each chunk.
#
#     Yields
#     ------
#     Iterable chunks of the specified size.
#     """
#     iterator = iter(iterable)
#     for first in iterator:  # stops when iterator is depleted
#         chunk = list(islice(iterator, size - 1))
#         if not chunk:
#             # End of the iterator
#             yield [first]
#             break
#         yield [first] + chunk


#
#
# def save_sparse_matrix(filename, matrix):
#     np.savez(
#         filename,
#         data=matrix.data,
#         indices=matrix.indices,
#         indptr=matrix.indptr,
#         shape=matrix.shape,
#     )
