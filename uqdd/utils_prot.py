"""
Protein utilities for computing and loading sequence embeddings.

This module provides thin wrappers around BioTransformers and Ankh to compute
protein embeddings, plus helpers to read Papyrus descriptors and merge results
back into pandas DataFrames.
"""

import logging
import os
from typing import List, Dict, Union, Generator, Optional

import ankh
import numpy as np
import pandas as pd
import ray
import torch
from biotransformers import BioTransformers
from papyrus_scripts.reader import read_protein_descriptors
from tqdm.auto import tqdm

from uqdd import DATA_DIR, DEVICE

torch.cuda.empty_cache()
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

all_models = [
    "esm1_t34",
    "esm1_t12",
    "esm1_t6",
    "esm1b",
    "esm_msa1",
    "esm_msa1b",
    "esm1v",
    "protbert",
    "protbert_bfd",
    "ankh-base",
    "ankh-large",
    "unirep",  # TODO to calculate it from scratch for TDC
]
num_gpus = torch.cuda.device_count()


def create_results_dict(
        entries: List[str], embeddings: List
) -> Dict[str, Union[List, torch.Tensor]]:
    """
    Create a mapping from entries to their embeddings.

    Parameters
    ----------
    entries : list of str
        Sequence identifiers or raw sequences.
    embeddings : list
        Embedding vectors aligned with `entries`.

    Returns
    -------
    dict[str, list or torch.Tensor]
        Mapping from entry to its embedding.

    Notes
    -----
    - The order of `embeddings` must match the order of `entries`.
    """
    results = {ent: emb for ent, emb in zip(entries, embeddings)}
    return results


def get_num_params(model: torch.nn.Module) -> int:
    """
    Count total parameters in a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to inspect.

    Returns
    -------
    int
        Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


def batch_generator(seq_list: List[str], size: int) -> Generator[List[str], None, None]:
    """
    Yield successive batches from a list.

    Parameters
    ----------
    seq_list : list of str
        List of sequences to split.
    size : int
        Batch size.

    Yields
    ------
    list of str
        Next batch of sequences.
    """
    for i in range(0, len(seq_list), size):
        yield seq_list[i: i + size]


def compute_biotransformer_embeddings(
        protein_sequences: List[str],
        embedding_type: str,
        batch_size: int = 8,
        num_gpus: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Compute sequence embeddings using BioTransformers.

    Parameters
    ----------
    protein_sequences : list of str
        Protein sequences (amino acid strings).
    embedding_type : str
        Model backend identifier (e.g., "esm1b", "protbert").
    batch_size : int, optional
        Batch size for embedding computation. Default is 8.
    num_gpus : int, optional
        Number of GPUs to allocate to the backend. Default is 1.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from sequence to pooled embedding tensor of shape (d,).

    Raises
    ------
    ValueError
        If `embedding_type` is not supported by BioTransformers.
    RuntimeError
        If Ray or backend initialization fails.
    """
    embtype_keys = {
        "esm1_t34": "esm1_t34_670M_UR100",
        "esm1_t12": "esm1_t12_85M_UR50S",
        "esm1_t6": "esm1_t6_43M_UR50S",
        "esm1b": "esm1b_t33_650M_UR50S",
        "esm_msa1": "esm_msa1_t12_100M_UR50S",
        "esm_msa1b": "esm_msa1b_t12_100M_UR50S",
        "esm1v": "esm1v_t33_650M_UR90S_1",
        "protbert": "protbert",
        "protbert_bfd": "protbert_bfd",
    }
    if embedding_type not in embtype_keys:
        raise ValueError(f"Unsupported BioTransformers model type: {embedding_type}")
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    biotrans = BioTransformers(backend=embtype_keys[embedding_type], num_gpus=num_gpus)
    embeddings = biotrans.compute_embeddings(
        protein_sequences, batch_size=batch_size, pool_mode=("mean",)
    )
    ray.shutdown()
    return create_results_dict(protein_sequences, embeddings["mean"])


def compute_ankh_embeddings(
        protein_sequences: List[str], embedding_type: str, batch_size: int = 32
) -> Dict[str, torch.Tensor]:
    """
    Compute sequence embeddings using Ankh.

    Parameters
    ----------
    protein_sequences : list of str
        Protein sequences (amino acid strings).
    embedding_type : str
        Either "ankh-base" or "ankh-large".
    batch_size : int, optional
        Batch size for embedding computation. Default is 32.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from sequence to pooled embedding tensor of shape (d,).

    Raises
    ------
    ValueError
        If `embedding_type` is not one of {"ankh-base", "ankh-large"}.
    RuntimeError
        If tokenizer or model evaluation fails.
    """
    if embedding_type == "ankh-base":
        logging.info("Loading Ankh base model...")
        model, tokenizer = ankh.load_base_model()
    elif embedding_type == "ankh-large":
        logging.info("Loading Ankh large model...")
        model, tokenizer = ankh.load_large_model()
    else:
        raise ValueError(f"Unsupported Ankh model type: {embedding_type}")

    model.to(DEVICE)
    model.eval()
    print(f"Number of parameters: {get_num_params(model)}")

    embeddings = []
    for batch_seqs in tqdm(
            batch_generator(protein_sequences, batch_size),
            desc=f"Processing Protein Embedding batches {embedding_type}",
            total=max(1, len(protein_sequences) // batch_size),
    ):
        outputs = tokenizer.batch_encode_plus(
            batch_seqs,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            batch_embeddings = model(
                input_ids=outputs["input_ids"], attention_mask=outputs["attention_mask"]
            )
            averaged_batch_embeddings = (
                batch_embeddings.last_hidden_state.mean(1).cpu().numpy()
            )
            embeddings.extend(averaged_batch_embeddings)

    return create_results_dict(protein_sequences, embeddings)


def get_papyrus_embeddings(
        target_ids: Optional[List[str]] = None, desc_type: str = "unirep"
) -> Dict[str, np.ndarray]:
    """
    Load protein embeddings from the Papyrus dataset.

    Parameters
    ----------
    target_ids : list of str or None, optional
        Target IDs for which embeddings are requested. If None, returns all available.
    desc_type : str, optional
        Descriptor type to extract (e.g., "unirep"). Default is "unirep".

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping from target_id to embedding vector of shape (d,).

    Raises
    ------
    RuntimeError
        If Papyrus reader fails to load descriptors.
    """

    def _merge_cols(row):
        return np.array(row[1:])

    protein_descriptors = read_protein_descriptors(
        desc_type=desc_type,
        version="latest",
        chunksize=100000,
        source_path=DATA_DIR,
        ids=target_ids,
        verbose=True,
    )

    protein_descriptors[desc_type] = protein_descriptors.apply(_merge_cols, axis=1)

    protein_descriptors = protein_descriptors[["target_id", desc_type]]
    protein_descriptors_mapper = protein_descriptors.set_index("target_id")[
        desc_type
    ].to_dict()

    return protein_descriptors_mapper


def merge_embeddings(
        df: pd.DataFrame,
        results_mapper: Dict[str, np.ndarray],
        embedding_type: str,
        query_col: str = "Sequence",
) -> pd.DataFrame:
    """
    Merge embeddings into a DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing sequences or IDs.
    results_mapper : dict[str, numpy.ndarray]
        Mapping from sequence or ID to embedding vectors.
    embedding_type : str
        Name of the target column to create (e.g., "esm1b", "ankh-base").
    query_col : str, optional
        Column in `df` used to look up embeddings. Default is "Sequence".

    Returns
    -------
    pd.DataFrame
        DataFrame with a new column `{embedding_type}` containing embedding arrays of shape (d,).

    Raises
    ------
    KeyError
        If `query_col` is not present in `df`.
    """
    df[embedding_type] = df[query_col].map(results_mapper)
    return df


def get_embeddings(
        df: pd.DataFrame,
        embedding_type: str,
        query_col: str = "Sequence",
        batch_size: int = 4,
) -> pd.DataFrame:
    """
    Compute or retrieve protein embeddings and add them to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with protein sequences or target IDs.
    embedding_type : str
        Embedding family: one of BioTransformers (esm*, protbert*), Ankh (ankh-*), or "unirep" via Papyrus.
    query_col : str, optional
        Column name containing sequences or target IDs. Default is "Sequence".
    batch_size : int, optional
        Batch size for computation when applicable. Default is 4.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional `{embedding_type}` column containing numpy arrays or tensors of shape (d,).

    Notes
    -----
    - Clears CUDA cache if available prior to computing embeddings.
    - Computes embeddings for unique values in `query_col` and maps them back to `df`.

    Raises
    ------
    ValueError
        If `embedding_type` is unsupported.
    RuntimeError
        If a backend fails to initialize or compute.
    """
    # Clear the CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embedding_type = embedding_type.lower()
    protein_sequences = df[query_col].unique().tolist()
    if embedding_type is None:
        return df
    elif embedding_type in [
        "esm1_t34",
        "esm1_t12",
        "esm1_t6",
        "esm1b",
        "esm_msa1",
        "esm_msa1b",
        "esm1v",
        "protbert",
        "protbert_bfd",
    ]:
        embeddings = compute_biotransformer_embeddings(
            protein_sequences, embedding_type, batch_size=batch_size
        )
    elif embedding_type in ["ankh-base", "ankh-large"]:
        embeddings = compute_ankh_embeddings(
            protein_sequences, embedding_type, batch_size=batch_size
        )
    elif embedding_type == "unirep":
        logging.warning(
            "UniRep can only be extracted from Papyrus and not computed on the fly for new input."
        )
        embeddings = get_papyrus_embeddings(protein_sequences, embedding_type)

    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

    df = merge_embeddings(
        df=df,
        results_mapper=embeddings,
        embedding_type=embedding_type,
        query_col=query_col,
    )

    return df
