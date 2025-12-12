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
    Creates a dictionary mapping each entry to its corresponding embedding.

    Parameters:
    -----------
    entries : List[str]
        A list of protein sequence identifiers or names.
    embeddings : List
        A list of corresponding embeddings.

    Returns:
    --------
    Dict[str, Union[List, torch.Tensor]]
        A dictionary where keys are entries and values are their respective embeddings.
    """
    results = {ent: emb for ent, emb in zip(entries, embeddings)}
    return results


def get_num_params(model: torch.nn.Module) -> int:
    """
    Computes the total number of trainable parameters in a model.

    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model.

    Returns:
    --------
    int
        Total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())


def batch_generator(seq_list: List[str], size: int) -> Generator[List[str], None, None]:
    """
    Yields successive n-sized chunks from a list.

    Parameters:
    -----------
    seq_list : List[str]
        The list to split into batches.
    size : int
        The batch size.

    Yields:
    -------
    Generator[List[str], None, None]
        A generator yielding batches of the input list.
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
    Computes embeddings using BioTransformers.

    Parameters:
    -----------
    protein_sequences : List[str]
        A list of protein sequences.
    embedding_type : str
        The type of embedding model to use.
    batch_size : int, optional
        The batch size for processing. Default is 8.
    num_gpus : int, optional
        Number of GPUs to use. Default is 1.

    Returns:
    --------
    Dict[str, torch.Tensor]
        A dictionary mapping sequences to their embeddings.
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
    Computes embeddings using the Ankh model.

    Parameters:
    -----------
    protein_sequences : List[str]
        A list of protein sequences.
    embedding_type : str
        The type of Ankh model to use.
    batch_size : int, optional
        The batch size for processing. Default is 32.

    Returns:
    --------
    Dict[str, torch.Tensor]
        A dictionary mapping sequences to their embeddings.
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
            total=len(protein_sequences) // batch_size,
    ):
        # Tokenize the current batch of sequences
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
    Retrieves protein embeddings from the Papyrus dataset.

    Parameters:
    -----------
    target_ids : Optional[List[str]], default=None
        A list of target IDs for which embeddings are needed. If None, all available embeddings are retrieved.
    desc_type : str, default="unirep"
        The type of descriptor to extract (e.g., "unirep", "cddd", etc.).

    Returns:
    --------
    Dict[str, np.ndarray]
        A dictionary where keys are target IDs and values are the corresponding protein embeddings.
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

    # Creating the mapper
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
    Merges computed embeddings into the original DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing protein sequences or target IDs.
    results_mapper : Dict[str, np.ndarray]
        A dictionary mapping sequences or target IDs to their embeddings.
    embedding_type : str
        The type of embeddings being merged (e.g., "esm1b", "ankh-base").
    query_col : str, default="Sequence"
        The column in the DataFrame containing the protein sequences or target IDs.

    Returns:
    --------
    pd.DataFrame
        The updated DataFrame with an additional column containing the embeddings.
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
    Computes or retrieves protein sequence embeddings and adds them to the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing protein sequences or target IDs.
    embedding_type : str
        The type of embeddings to compute or retrieve (e.g., "esm1b", "ankh-large", "unirep").
    query_col : str, default="Sequence"
        The column in the DataFrame containing the protein sequences or target IDs.
    batch_size : int, default=4
        The batch size for computing embeddings, applicable for models that require batch processing.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with an additional column containing the computed or retrieved embeddings.
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
