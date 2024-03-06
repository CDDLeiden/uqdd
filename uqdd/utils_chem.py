import copy
from typing import Union, List, Tuple, Dict, Any

from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from uqdd import DATA_DIR
from uqdd.utils import check_nan_duplicated, custom_agg

from papyrus_scripts.reader import read_molecular_descriptors
from papyrus_scripts.preprocess import consume_chunks

RDLogger.DisableLog("rdApp.info")
print(f"rdkit {rdkit.__version__}")

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


def standardize(smi, logger=None, suppress_exception=False):
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
        mol = Chem.MolFromSmiles(smi)
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
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

    return Chem.MolToSmiles(mol)


def standardize_wrapper(args):
    """
    Wrapper function for the standardize function to be used with the concurrent.futures.ProcessPoolExecutor.
    """
    return standardize(*args)


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

    # standardizing the SMILES in parallel
    # tqdm.pandas(desc="Standardizing SMILES")
    unique_smiles = df_filtered[smiles_col].unique()
    args_list = [(smi, logger, suppress_exception) for smi in unique_smiles]

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(standardize_wrapper, args_list),
                total=len(args_list),
                desc="Standardizing Unique SMILES",
            )
        )
    standardized_result = {smi: result for smi, result in zip(unique_smiles, results)}

    # Apply the standardized result to the dataframe
    df_filtered[smiles_col] = df_filtered[smiles_col].map(standardized_result)

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

    with ProcessPoolExecutor() as executor:
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

    with ProcessPoolExecutor() as executor:
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
    return scaffold
