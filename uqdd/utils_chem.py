import copy
import logging
from multiprocessing import shared_memory
from typing import Union, List, Tuple, Any, Optional, Dict, Generator
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem, rdFMCS
from rdkit.Chem import MolToSmiles, MolFromSmiles, MolFromSmarts, SanitizeMol
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem.rdchem import Mol as RdkitMol

from uqdd import DATA_DIR
from uqdd.utils import (
    check_nan_duplicated,
    custom_agg,
    load_npy_file,
    save_npy_file,
    save_pickle,
    load_pickle,
)

from papyrus_scripts.reader import read_molecular_descriptors
from papyrus_scripts.preprocess import consume_chunks

# scipy hierarchy clustering
from scipy.cluster.hierarchy import cophenet, cut_tree
from scipy.spatial.distance import pdist
from fastcluster import linkage

# SKlearn metrics
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations, islice
import math

# Disable RDKit warnings
RDLogger.DisableLog("rdApp.info")
# print(f"rdkit {rdkit.__version__}")

N_WORKERS = 20
BATCH_SIZE = 10000

all_models = [
    "ecfp1024",
    "ecfp2048",
    "mold2",
    "mordred",
    "cddd",
    "fingerprint",
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


def rdkit_standardize(
    smi: str, logger: logging.Logger = None, suppress_exception: bool = False
):
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
    """
    Removes stereochemistry information from an RDKit molecule.

    Parameters:
    -----------
    mol : RdkitMol
        The RDKit molecule from which stereochemistry should be removed.

    Returns:
    --------
    Optional[RdkitMol]
        The modified molecule with stereochemistry removed, or None if an error occurs.
    """
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
    """
    Neutralizes charges in an RDKit molecule.

    Parameters:
    -----------
    mol : RdkitMol
        The RDKit molecule to be neutralized.

    Returns:
    --------
    Optional[RdkitMol]
        The neutralized molecule, or None if an error occurs.
    """
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
    """
    Removes isotopic labels from an RDKit molecule.

    Parameters:
    -----------
    mol : RdkitMol
        The RDKit molecule from which isotopic labels should be removed.

    Returns:
    --------
    Optional[RdkitMol]
        The modified molecule with isotopes removed, or None if an error occurs.
    """
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
    logger: Optional[logging.Logger] = None,
    suppress_exception: bool = True,
    rd_standardize: bool = False,
) -> pd.DataFrame:
    """
    Standardizes SMILES representations in a DataFrame in parallel.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing SMILES strings.
    smiles_col : str, optional
        Column containing SMILES strings (default: "SMILES").
    logger : logging.Logger, optional
        Logger instance for logging messages (default: None).
    suppress_exception : bool, optional
        If True, suppresses exceptions during standardization (default: True).
    rd_standardize : bool, optional
        Whether to use RDKit's standardization method (default: False).

    Returns:
    --------
    pd.DataFrame
        The standardized DataFrame.
    """
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
    other_dup_col: Union[List[str], str, None] = None,
    sorting_col: str = "",
    drop: bool = True,
    keep: Union[bool, str] = "last",
    logger: Optional[logging.Logger] = None,
    suppress_exception: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standardizes SMILES representations in a DataFrame and checks for duplicates and NaNs.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    smiles_col : str, optional
        Column containing SMILES strings (default: "SMILES").
    other_dup_col : str or list of str, optional
        Additional columns to consider when checking for duplicates (default: None).
    sorting_col : str, optional
        Column to sort by before standardization (default: "").
    drop : bool, optional
        Whether to drop NaN SMILES before standardization (default: True).
    keep : bool or str, optional
        Determines which duplicate SMILES to keep ('first', 'last', or 'aggregate') (default: "last").
    logger : logging.Logger, optional
        Logger instance for logging messages (default: None).
    suppress_exception : bool, optional
        If True, suppresses exceptions during standardization (default: True).

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Standardized DataFrame, DataFrame with NaNs, DataFrame with duplicates.

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

    df_filtered = parallel_standardize(
        df_filtered, smiles_col, logger, suppress_exception
    )

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
    smiles: str,
    radius: int = 2,
    length: int = 1024,
    use_features: bool = False,
    use_chirality: bool = False,
) -> np.ndarray:
    """
    Generates an ECFP (Extended Connectivity Fingerprint) from a SMILES string.

    Parameters:
    -----------
    smiles : str
        The input SMILES string.
    radius : int, optional
        Radius for circular substructure (default: 2).
    length : int, optional
        Length of fingerprint in bits (default: 1024).
    use_features : bool, optional
        Whether to use feature-based fingerprints (default: False).
    use_chirality : bool, optional
        Whether to include chirality information (default: False).

    Returns:
    --------
    np.ndarray
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


def wrapper_ecfp_from_smiles(args: Tuple[str, int, int, bool, bool]) -> np.ndarray:
    """
    Wrapper function for generating ECFP fingerprints from a SMILES string.

    Parameters:
    -----------
    args : Tuple[str, int, int, bool, bool]
        A tuple containing the SMILES string, radius, fingerprint length, feature flag, and chirality flag.

    Returns:
    --------
    np.ndarray
        The ECFP fingerprint as a binary numpy array.
    """
    return ecfp_from_smiles(*args)


def generate_ecfp(
    smiles: List[str],
    radius: int = 2,
    length: int = 1024,
    use_features: bool = False,
    use_chirality: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generates ECFP fingerprints for a list of SMILES strings.

    Parameters:
    -----------
    smiles : List[str]
        List of SMILES strings to generate fingerprints for.
    radius : int, optional
        Radius for circular substructure (default: 2).
    length : int, optional
        Length of fingerprint in bits (default: 1024).
    use_features : bool, optional
        Whether to use feature-based fingerprints (default: False).
    use_chirality : bool, optional
        Whether to include chirality information (default: False).

    Returns:
    --------
    Dict[str, np.ndarray]
        A dictionary mapping each SMILES string to its generated ECFP fingerprint.

    Notes:
    ------
    This function applies the `ECFP_from_smiles` function to each
    SMILES string in the 'smiles' column of the input dataframe,
    and generates ECFP fingerprints with the specified radius, length,
    and optional parameters. The resulting fingerprints are stored in columns named 'ECFP-{length}',
    where {length} is the specified fingerprint length.
    """
    # Generate ECFP fingerprints
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


def get_mol_descriptors(
    smiles: str, chosen_descriptors: Optional[List[str]] = None
) -> np.ndarray:
    """
    Computes molecular descriptors for a given SMILES string.

    Parameters:
    -----------
    smiles : str
        The input SMILES string.
    chosen_descriptors : List[str], optional
        List of descriptors to compute (default: all available descriptors).

    Returns:
    --------
    np.ndarray
        Array of computed molecular descriptor values.

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


def wrapper_get_mol_descriptors(args: Tuple[str, Optional[List[str]]]) -> np.ndarray:
    """
    Wrapper function for computing molecular descriptors for a SMILES string.

    Parameters:
    -----------
    args : Tuple[str, Optional[List[str]]]
        A tuple containing the SMILES string and optionally a list of descriptors.

    Returns:
    --------
    np.ndarray
        Computed molecular descriptor values.
    """
    return get_mol_descriptors(*args)


def generate_mol_descriptors(
    smiles: List[str], chosen_descriptors: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Computes molecular descriptors for a list of SMILES strings.

    Parameters:
    -----------
    smiles : List[str]
        List of SMILES strings to compute descriptors for.
    chosen_descriptors : List[str], optional
        List of descriptors to compute (default: all available descriptors).

    Returns:
    --------
    Dict[str, np.ndarray]
        A dictionary mapping each SMILES string to its computed molecular descriptors.
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


def get_papyrus_descriptors(
    connectivity_ids: Optional[List[str]] = None,
    desc_type: str = "cddd",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, np.ndarray]:
    """
    Retrieves molecular descriptors from the Papyrus dataset.

    Parameters:
    -----------
    connectivity_ids : List[str], optional
        List of connectivity IDs for which descriptors should be retrieved (default: None).
    desc_type : str, optional
        Type of descriptor to retrieve (default: "cddd").
    logger : logging.Logger, optional
        Logger instance for logging (default: None).

    Returns:
    --------
    Dict[str, np.ndarray]
        A dictionary mapping connectivity IDs to their molecular descriptors.
    """

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

    mol_descriptors[desc_type] = mol_descriptors.apply(_merge_cols, axis=1)

    mol_descriptors = mol_descriptors[["connectivity", desc_type]]

    mol_descriptors_mapper = mol_descriptors.set_index("connectivity")[
        desc_type
    ].to_dict()

    return mol_descriptors_mapper


def get_chem_desc(
    df: pd.DataFrame,
    desc_type: str = "ecfp1024",
    query_col: str = "SMILES",
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Computes and attaches molecular descriptors to a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    desc_type : str, optional
        Type of descriptor to compute (default: "ecfp1024").
    query_col : str, optional
        Column containing the input SMILES (default: "SMILES").
    logger : logging.Logger, optional
        Logger instance for logging (default: None).
    kwargs : dict
        Additional arguments for descriptor computation.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with an additional column containing molecular descriptors.
    """
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
        desc_mapper = generate_ecfp(unique_entries, radius=2, length=length, **kwargs)
    elif desc_type == "moldesc":
        desc_mapper = generate_mol_descriptors(unique_entries, **kwargs)
    elif desc_type == "graph2d":
        raise NotImplementedError
    else:
        raise ValueError(f"desc_mol: {desc_type} is not a valid molecular descriptor")

    df[desc_type] = df[query_col].map(desc_mapper)

    return df


def mol_to_pil_image(molecule: RdkitMol, width: int = 300, height: int = 300) -> Image:
    """
    Converts an RDKit molecule to a PIL image.

    Parameters
    ----------
    molecule : RdkitMol (rdkit.Chem.rdchem.Mol)
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


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a given SMILES string.

    Parameters:
    -----------
    smiles : str
        The input SMILES string.
    include_chirality : bool, optional
        Whether to retain chirality in the scaffold (default: False).

    Returns:
    --------
    str
        The computed scaffold SMILES string.
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


def merge_scaffolds(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    """
    Computes and merges scaffold information into a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    smiles_col : str, optional
        Column containing SMILES strings (default: "SMILES").

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with an additional 'scaffold' column.
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


# adopted from https://github.com/nina23bom/NPS-Pharmacological-profile-fingerprint-prediction-using-ML/blob/main/001.%20NPS%20unique%20compounds%20MCS%20Hierarchical%20clustering%20-%20Class%20Label.ipynb
def tanimoto_mcs(smi1: str, smi2: str) -> float:
    """
    Computes the Tanimoto similarity based on the Maximum Common Substructure (MCS) between two molecules.

    Parameters:
    -----------
    smi1 : str
        SMILES string of the first molecule.
    smi2 : str
        SMILES string of the second molecule.

    Returns:
    --------
    float
        Tanimoto similarity score based on MCS.
    """
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
    return mcs_tani


def tanimoto_mcs_withH(smi1: str, smi2: str) -> float:
    """
    Computes the Tanimoto similarity based on the Maximum Common Substructure (MCS) between two hydrogen-included molecules.

    Parameters:
    -----------
    smi1 : str
        SMILES string of the first molecule.
    smi2 : str
        SMILES string of the second molecule.

    Returns:
    --------
    float
        Tanimoto similarity score based on MCS with hydrogen included.
    """
    # reading smiles of two molecules and create molecule
    m1 = Chem.MolFromSmiles(smi1)
    m2 = Chem.MolFromSmiles(smi2)

    m1H = Chem.AddHs(m1)
    m2H = Chem.AddHs(m2)
    mols = [m1H, m2H]

    # number heavy atoms of both molecules
    a = m1H.GetNumAtoms()
    b = m2H.GetNumAtoms()
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
    return mcs_tani


def tanimoto_mcs_wrapper(index_pair: Tuple[int, int], cid_list: List[str]) -> float:
    """
    Wrapper function for computing Tanimoto similarity using MCS.

    Parameters:
    -----------
    index_pair : Tuple[int, int]
        Index pair representing two molecules in a list.
    cid_list : List[str]
        List of SMILES strings corresponding to compound IDs.

    Returns:
    --------
    float
        Computed Tanimoto similarity score.
    """
    cid1, cid2 = index_pair
    return tanimoto_mcs(cid_list[cid1], cid_list[cid2])


def tanimoto_mcs_withH_wrapper(
    index_pair: Tuple[int, int], cid_list: List[str]
) -> float:
    """
    Wrapper function for computing Tanimoto similarity using MCS with hydrogen included.

    Parameters:
    -----------
    index_pair : Tuple[int, int]
        Index pair representing two molecules in a list.
    cid_list : List[str]
        List of SMILES strings corresponding to compound IDs.

    Returns:
    --------
    float
        Computed Tanimoto similarity score with hydrogen included.
    """
    cid1, cid2 = index_pair
    return tanimoto_mcs_withH(cid_list[cid1], cid_list[cid2])


def chunked_iterable(n: int, size: int) -> Generator[list[tuple[int, int]], Any, None]:
    """
    Generates chunks of index pairs for all unique molecule comparisons.

    Parameters:
    -----------
    n : int
        Total number of compounds.
    size : int
        Chunk size for processing pairs.

    Yields:
    --------
    List[Tuple[int, int]]
        List of index pairs in chunks.
    """
    iterable = combinations(range(n), 2)
    while True:
        chunk = list(islice(iterable, size))
        if not chunk:
            return
        yield chunk


def calculate_total_chunks(n_compounds: int, batch_size: int) -> int:
    """
    Calculates the total number of chunks required for pairwise similarity calculations.

    Parameters:
    -----------
    n_compounds : int
        Total number of compounds.
    batch_size : int
        Batch size for chunked processing.

    Returns:
    --------
    int
        Total number of chunks needed.
    """
    total_pairs = n_compounds * (n_compounds - 1) / 2
    total_chunks = math.ceil(total_pairs / batch_size)
    return total_chunks


def process_chunk(
    chunk: List[Tuple[int, int]],
    similarity_matrix: np.ndarray,
    cid_list: List[str],
    tanimoto_mcs_func: callable,
) -> None:
    """
    Processes a chunk of index pairs and updates the similarity matrix.

    Parameters:
    -----------
    chunk : List[Tuple[int, int]]
        List of index pairs for similarity computation.
    similarity_matrix : np.ndarray
        Preallocated similarity matrix.
    cid_list : List[str]
        List of SMILES strings.
    tanimoto_mcs_func : callable
        Function to compute Tanimoto similarity.

    Returns:
    --------
    None
    """
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(tanimoto_mcs_func, pair, cid_list): pair for pair in chunk
        }
        for future in futures:
            similarity = future.result()
            i, j = futures[future]
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity


def save_similarity_matrix(
    matrix: np.ndarray, filename: str = "similarity_matrix.npy"
) -> None:
    """
    Saves the similarity matrix to a file.

    Parameters:
    -----------
    matrix : np.ndarray
        The similarity matrix to be saved.
    filename : str, optional
        Name of the file to save the matrix (default: "similarity_matrix.npy").

    Returns:
    --------
    None
    """
    np.save(filename, matrix)
    print(f"Similarity matrix saved to {filename}")


def hierarchical_clustering(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    batch_size: int = 10000,
    withH: bool = False,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Performs hierarchical clustering based on Maximum Common Substructure (MCS) similarity.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing SMILES strings.
    smiles_col : str, optional
        Column containing SMILES strings (default: "SMILES").
    batch_size : int, optional
        Size of each processing batch (default: 10000).
    withH : bool, optional
        Whether to include hydrogen in the similarity calculations (default: False).
    save_path : str, optional
        Path to save the similarity matrix (default: None).

    Returns:
    --------
    np.ndarray
        The computed similarity matrix.
    """
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        filepath = Path(save_path) / "mcs.pkl.npy"

        # now checking if file exists
        if filepath.exists():
            print(f"Similarity matrix already exists at {filepath}")
            return load_npy_file(filepath)

    print(f"Chunk Size: {batch_size}")
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


def form_linkage(
    X: np.ndarray,
    save_path: Optional[str] = None,
    calculate_cophenetic_coeff: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes hierarchical clustering linkage matrix using Ward's method.

    Parameters:
    -----------
    X : np.ndarray
        Pairwise similarity matrix.
    save_path : str, optional
        Path to save the linkage matrix (default: None).
    calculate_cophenetic_coeff : bool, optional
        Whether to calculate the cophenetic coefficient (default: True).

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Processed similarity matrix and linkage matrix.
    """
    # start = time.time()
    n_rows, n_cols = X.shape
    upper_indices = np.triu_indices(n_rows, 1)
    x_dist = 1 - X[upper_indices]
    X_ = 1 - X
    filepath = None
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        filepath = Path(save_path) / "mcs_linkage.pkl.npy"
        # now checking if file exists
        if filepath.exists():
            print(f"Linkage matrix already exists at {filepath}")
            return X_, load_npy_file(filepath)

    # check if X_ and X2 are the same
    Z = linkage(x_dist, method="ward")

    # Z = linkage(X, method="ward") # TODO : save and load if existing
    if save_path is not None:
        save_npy_file(Z, filepath)
    if calculate_cophenetic_coeff:
        calculate_cophenet(X_, Z, save_path=save_path)
    return X_, Z


def calculate_cophenet(
    X: np.ndarray, Z: np.ndarray, save_path: Optional[str] = None
) -> float:
    """
    Computes the cophenetic correlation coefficient for a given linkage matrix.

    Parameters:
    -----------
    X : np.ndarray
        Pairwise similarity matrix.
    Z : np.ndarray
        Linkage matrix from hierarchical clustering.
    save_path : str, optional
        Path to save cophenetic coefficient data (default: None).

    Returns:
    --------
    float
        Cophenetic correlation coefficient.
    """

    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        Pdist_path = Path(save_path) / "mcs_pdist.pkl.npy"
        Coph_dists_path = Path(save_path) / "mcs_coph_dists.pkl.npy"
        C_path = Path(save_path) / "mcs_c.pkl"
        # now checking if file exists
        if C_path.exists():
            print(f"Cophenetic Coefficient already exists at {C_path}")
            return load_pickle(C_path)
    else:
        Pdist_path, Coph_dists_path, C_path = None, None, None

    Pdist = pdist(X)
    c, coph_dists = cophenet(Z, Pdist)
    print("Cophenetic coefficient calculated: %0.4f" % c)
    if save_path is not None:
        save_npy_file(Pdist, Pdist_path)
        save_npy_file(coph_dists, Coph_dists_path)
        save_pickle(c, C_path)
    return c


def calculate_silhouette(
    k: int, shm_name: str, shape: Tuple[int, int], Z: np.ndarray
) -> Tuple[int, float]:
    """
    Computes the silhouette score for a given number of clusters.

    Parameters:
    -----------
    k : int
        Number of clusters.
    shm_name : str
        Shared memory name for accessing precomputed similarity matrix.
    shape : Tuple[int, int]
        Shape of the similarity matrix.
    Z : np.ndarray
        Linkage matrix for hierarchical clustering.

    Returns:
    --------
    Tuple[int, float]
        Number of clusters and corresponding silhouette score.
    """
    # Access the shared memory block
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    X_shared = np.ndarray(shape, dtype=np.float32, buffer=existing_shm.buf)

    cluster_labels = cut_tree(Z, n_clusters=k).flatten()
    silhouette_avg = silhouette_score(X_shared, cluster_labels, metric="precomputed")

    existing_shm.close()
    return k, silhouette_avg


def calculate_silhouette_helper(
    args: Tuple[int, str, Tuple[int, int], np.ndarray]
) -> Tuple[int, float]:
    """
    Wrapper function for parallel silhouette score computation.

    Parameters:
    -----------
    args : Tuple[int, str, Tuple[int, int], np.ndarray]
        Arguments including cluster count, shared memory name, similarity matrix shape, and linkage matrix.

    Returns:
    --------
    Tuple[int, float]
        Number of clusters and corresponding silhouette score.
    """
    k, shm_name, shape, Z = args
    return calculate_silhouette(k, shm_name, shape, Z)


def sil_K(
    X: np.ndarray, Z: np.ndarray, max_k: int = 500
) -> Tuple[List[int], List[float], int]:
    """
    Determines the optimal number of clusters based on silhouette scores.

    Parameters:
    -----------
    X : np.ndarray
        Pairwise similarity matrix.
    Z : np.ndarray
        Linkage matrix from hierarchical clustering.
    max_k : int, optional
        Maximum number of clusters to evaluate (default: 500).

    Returns:
    --------
    Tuple[List[int], List[float], int]
        List of cluster sizes, corresponding silhouette scores, and optimal cluster count.
    """
    # Create shared memory block for X
    X = np.array(X, dtype=np.float32)
    shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
    X_shared = np.ndarray(X.shape, dtype=np.float32, buffer=shm.buf)  # X.dtype
    np.copyto(X_shared, X)
    # Prepare arguments for the helper function
    args_list = [(k, shm.name, X.shape, Z) for k in range(2, max_k)]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        # results = pool.starmap(calculate_silhouette, [(k, shm.name, X.shape, Z) for k in range(2, max_k)])
        # Initialize the tqdm progress bar
        results = []
        for result in tqdm(
            pool.imap_unordered(calculate_silhouette_helper, args_list),
            total=max_k - 2,
            desc="Calculating silhouette scores",
        ):
            results.append(result)

    # Clean up shared memory
    shm.close()
    shm.unlink()
    results.sort(key=lambda x: x[0])  # Ensure the results are sorted by k
    n_clu, sil = zip(*results)
    optimal_clu = n_clu[sil.index(max(sil))]
    print("Optimal number of clusters: ", optimal_clu)

    return list(n_clu), list(sil), optimal_clu


def plot_silhouette_analysis(
    cluster_counts: List[int],
    silhouette_scores: List[float],
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plots the silhouette analysis for determining the optimal number of clusters.

    Parameters:
    -----------
    cluster_counts : List[int]
        List of cluster sizes.
    silhouette_scores : List[float]
        List of silhouette scores corresponding to each cluster size.
    output_path : str, optional
        Path to save the plot (default: None).

    Returns:
    --------
    None
    """
    # Initialize the plot
    fig = plt.figure(figsize=(12, 5), dpi=600)
    plt.rc("font", family="serif")
    plt.plot(cluster_counts, silhouette_scores)  # , label="MCS"
    # plt.scatter(cluster_counts, silhouette_scores, label="MCS")
    # # Plot each series of cluster counts vs. silhouette scores
    # for i in range(len(cluster_counts)):
    #     plt.scatter(cluster_counts[i], silhouette_scores[i], label=labels[i])

    # Adding plot details
    # plt.legend(loc="lower right", shadow=True, fontsize=16)
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


def plot_cluster_heatmap(
    data_matrix: np.ndarray, output_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Generates a heatmap for hierarchical clustering results.

    Parameters:
    -----------
    data_matrix : np.ndarray
        Clustering similarity matrix.
    output_path : str, optional
        Path to save the heatmap image (default: None).

    Returns:
    --------
    None
    """
    if hasattr(data_matrix, "index"):
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
    # plt.show()


def clustering(
    df: pd.DataFrame,
    smiles_col: str = "scaffold",
    max_k: int = 500,
    optimal_k: Optional[int] = None,
    withH: bool = False,
    export_mcs_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Performs hierarchical clustering based on molecular similarity.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing molecular data.
    smiles_col : str, optional
        Column containing molecular representations (default: "scaffold").
    max_k : int, optional
        Maximum number of clusters to evaluate (default: 500).
    optimal_k : int, optional
        Predefined optimal number of clusters (default: None).
    withH : bool, optional
        Whether to consider hydrogen atoms in similarity calculations (default: False).
    export_mcs_path : str, optional
        Path to save processed clustering data (default: None).

    Returns:
    --------
    pd.DataFrame
        Clustered DataFrame with assigned cluster labels.
    """
    clustered_df_path = None
    if export_mcs_path:
        clustered_df_path = Path(export_mcs_path) / "clustered_df.pkl"
        if clustered_df_path.exists():
            print(f"Clustered DataFrame already exists at {clustered_df_path}")
            return load_pickle(clustered_df_path)
    # pre cleaning
    df_clean = df.copy()[[smiles_col]]
    # dropp duplicates to avoid self comparison and reset index
    df_clean.drop_duplicates(subset=smiles_col, keep="first", inplace=True)

    df_clean.dropna(subset=[smiles_col], inplace=True)
    # print(f"after nan drop: {df_clean.shape}")
    df_clean.reset_index(inplace=True, drop=True)

    # TODO : checking mcs file existing then loading it instead of recalculation
    mcs_np = hierarchical_clustering(
        df_clean,
        smiles_col=smiles_col,
        batch_size=BATCH_SIZE,
        withH=withH,
        save_path=export_mcs_path,
    )
    # Just for debugging
    # df_clean = df_clean[:1000]
    # mcs_np = mcs_np[:1000, :1000]
    # if export_mcs_path:
    #     Path(export_mcs_path).mkdir(parents=True, exist_ok=True)
    #     df_mcs.to_csv(Path(export_mcs_path) / "mcs_matrix.csv", index=True)
    # df_pair.to_csv(Path(export_mcs_path) / "scaffold_sim_pair.csv", index=True)
    mcs_x, mcs_z = form_linkage(
        mcs_np, save_path=export_mcs_path, calculate_cophenetic_coeff=True
    )
    max_k = min(max_k, df_clean[smiles_col].nunique())
    # max_k = df_clean[smiles_col].nunique ()
    print(f"Max number of clusters: {max_k}")
    if optimal_k is None:
        mcs_k, mcs_sil, optimal_k = sil_K(mcs_x, mcs_z, max_k=max_k)  # , max_k=max_k
        if export_mcs_path:
            fig_output_path = Path(export_mcs_path) / "mcs_figures"
            Path(fig_output_path).mkdir(parents=True, exist_ok=True)
            plot_silhouette_analysis(mcs_k, mcs_sil, output_path=fig_output_path)

            optimal_k_path = Path(export_mcs_path) / f"mcs_optimal_k.pkl"
            save_pickle(optimal_k, optimal_k_path)

            # saving the silhouette scores
            sil_scores_path = Path(export_mcs_path) / f"mcs_sil_scores.pkl"
            save_pickle(zip(mcs_k, mcs_sil), sil_scores_path)

        # plot_cluster_heatmap(mcs_np, output_path=fig_output_path) # TAKES SO MUCH TIME
    # optimal_clu = 11974
    # df_clean["cluster"] = fcluster(mcs_z, optimal_clu, criterion="maxclust")
    print(f"Optimal number of clusters: {optimal_k}")
    df_clean["cluster"] = cut_tree(mcs_z, n_clusters=optimal_k).flatten()

    # now we map the cluster to the original dataframe
    df = pd.merge(df, df_clean, on=smiles_col, how="left", validate="many_to_many")
    df["cluster"] = df["cluster"].astype("Int64")
    if export_mcs_path:
        save_pickle(df, clustered_df_path)
    return df
