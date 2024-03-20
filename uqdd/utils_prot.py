import os
import logging
import torch
import numpy as np
import pandas as pd
from biotransformers import BioTransformers
import ankh
import ray
from uqdd import DATA_DIR, DEVICE
from papyrus_scripts.reader import read_protein_descriptors
from tqdm.auto import tqdm

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
    "unirep",  # TODO to calculate it
]
num_gpus = torch.cuda.device_count()


def create_results_dict(entries, embeddings):
    results = {ent: emb for ent, emb in zip(entries, embeddings)}
    return results


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def batch_generator(seq_list, size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(seq_list), size):
        yield seq_list[i : i + size]


def compute_biotransformer_embeddings(
    protein_sequences: list, embedding_type: str, batch_size=8
):
    """Compute embeddings using BioTransformers."""
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
    protein_sequences: list, embedding_type: str, batch_size: int = 32
):
    """Compute embeddings using Ankh."""
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
    #
    # with torch.no_grad():
    #     for batch in tqdm(dataloader, desc="Computing embeddings"):
    #         input_ids, attention_mask = batch
    #         input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
    #         batch_embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
    #         averaged_batch_embeddings = (
    #             batch_embeddings.last_hidden_state.mean(1).cpu().numpy()
    #         )
    #         embeddings.extend(averaged_batch_embeddings)
    # embeddings = model(
    #     input_ids=outputs["input_ids"], attention_mask=outputs["attention_mask"]
    # )
    # Average the embeddings over dim 1
    # averaged_embeddings = embeddings.last_hidden_state.mean(1).numpy()

    return create_results_dict(protein_sequences, embeddings)


def get_papyrus_embeddings(target_ids=None, desc_type="unirep"):
    # df, target_id_col=None, desc_type="unirep"):
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


def merge_embeddings(df, results_mapper, embedding_type, query_col="Sequence"):
    """
    Merge the computed embeddings with the original DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to merge the embeddings with.
    results_mapper : dict
        A dictionary where the keys are the protein sequences or target_ids and the values are the computed embeddings.
    embedding_type : str
        The type of embeddings being merged. This will be used to create a column name.
    query_col : str
        The name of the column in the DataFrame that contains the protein sequences or target_ids

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the merged embeddings.
    """

    df[embedding_type] = df[query_col].map(results_mapper)

    return df


def get_embeddings(
    df: pd.DataFrame,
    embedding_type: str,
    query_col: str = "Sequence",
    batch_size: int = 4,
) -> pd.DataFrame:
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


# class ProteinEmbedder:
#     def __init__(
#         self,
#         df: pd.DataFrame = None,
#         sequence_col: str = "sequence",
#         embedding_type: str = "esm1b",
#         logger=None,
#     ):
#         self.all_models = [
#             "esm1_t34",
#             "esm1_t12",
#             "esm1_t6",
#             "esm1b",
#             "esm_msa1",
#             "esm_msa1b",
#             "esm1v",
#             "protbert",
#             "protbert_bfd",
#             "ankh-base",
#             "ankh-large",
#         ]
#         assert embedding_type in self.all_models, "Unsupported embedding type."
#
#         self.log = create_logger("ProteinEmbedder") if logger is None else logger
#         self.df = df
#         self.sequence_col = sequence_col
#         self.embedding_type = embedding_type
#
#     def compute_embeddings(self):
#         """Compute embeddings based on the specified type."""
#         protein_sequences = self.df[self.sequence_col].unique().tolist()
#         if self.embedding_type in [
#             "esm1_t34",
#             "esm1_t12",
#             "esm1_t6",
#             "esm1b",
#             "esm_msa1",
#             "esm_msa1b",
#             "esm1v",
#             "protbert",
#             "protbert_bfd",
#         ]:
#             embeddings = self.compute_biotransformer_embeddings(protein_sequences)
#         elif self.embedding_type in ["ankh-base", "ankh-large"]:
#             embeddings = self.compute_ankh_embeddings(protein_sequences)
#         else:
#             raise ValueError(f"Unsupported embedding type: {self.embedding_type}")
#
#         self.df = self.merge_embeddings(protein_sequences, embeddings)
#         results = {}
#         for prot, emb in zip(protein_sequences, embeddings):
#             results[prot] = emb
#         self.df[self.embedding_type] = self.df[self.sequence_col].map(results)
#         return self.df
#
#         # elif self.embedding_type == "all":
#         #     # Warn if extensive embedding calculation is selected
#         #     self.log.warning(
#         #         "Extensive embedding calculation selected. This will require significant time and memory."
#         #     )
#         #     self.compute_all_embeddings()
#
#     def compute_biotransformer_embeddings(self, protein_sequences: list):
#         """Compute embeddings using BioTransformers."""
#         # we need to calculate only for unique sequences with
#         # concurrent futures parallelism then merge the results
#         # sequences = self.df[self.sequence_col].tolist()
#         ray.init()
#         bio_trans = BioTransformers(
#             backend=self.embedding_type, num_gpus=torch.cuda.device_count()
#         )
#         embeddings = bio_trans.compute_embeddings(
#             protein_sequences, pool_mode=("cls", "mean"), batch_size=64
#         )
#         return embeddings["mean"]
#         # self.merge_embeddings(embeddings["mean"])
#
#     def compute_ankh_embeddings(self, protein_sequences: list):
#         """Compute embeddings using Ankh."""
#         if self.embedding_type == "ankh-base":
#             model, tokenizer = ankh.load_base_model()
#         elif self.embedding_type == "ankh-large":
#             model, tokenizer = ankh.load_large_model()
#         else:
#             self.log.error("Invalid Ankh model type.")
#             return
#
#         model.eval()
#         # protein_sequences = self.df[self.sequence_col].tolist()
#         outputs = tokenizer.batch_encode_plus(
#             protein_sequences,
#             add_special_tokens=True,
#             padding=True,
#             return_tensors="pt",
#         )
#         with torch.no_grad():
#             embeddings = model(
#                 input_ids=outputs["input_ids"], attention_mask=outputs["attention_mask"]
#             )
#         # Average the embeddings over dim 1
#         averaged_embeddings = embeddings.last_hidden_state.mean(1).numpy()
#         return averaged_embeddings
#
#         # self.merge_embeddings(averaged_embeddings)
#
#     def merge_embeddings(self, protein_sequences, embeddings):
#         """Merge the computed embeddings with the original DataFrame."""
#         # Convert embeddings to a list of lists, where each sublist represents the embedding vector for a sequence
#         results = {}
#         for prot, emb in zip(protein_sequences, embeddings):
#             results[prot] = emb
#
#         column_name = f"{self.embedding_type}_embeddings"
#         self.df[column_name] = self.df[self.sequence_col].map(results)
#
#         return self.df
#
#         # embeddings_list = embeddings.tolist()
#         #
#         # # Create a column name that is representative of the type of embedding
#         # column_name = f"{self.embedding_type}_embeddings"
#         #
#         # # Assign the list of embeddings to the DataFrame in a single column
#         # self.df[column_name] = embeddings_list
#         #
#         # self.log.info(f"{column_name} merged with DataFrame.")
#
#     def initialize_data(self, **kwargs):
#         """
#         Initialize data by first trying to load existing processed data.
#         If no processed data is found, looks for and processes unprocessed data.
#         """
#         possible_exts = ["csv", "pkl", "parquet"]
#
#         # Attempt to load processed data
#         if self.try_loading_data(
#             self.base_path,
#             f"{self.data_name}_proteins_{self.embedding_type}",
#             possible_exts,
#         ):
#             self.log.info(f"Loaded processed data for {self.embedding_type}.")
#             return True
#
#         ## in case all embeddings are to be computed
#         if self.embedding_type == "all":
#             for model in self.all_models:
#                 self.embedding_type = model
#                 if not self.try_loading_data(
#                     self.base_path,
#                     f"{self.data_name}_proteins_{self.embedding_type}",
#                     ["csv", "pkl", "parquet"],
#                 ):
#                     self.log.info(f"Computing embeddings for {model}")
#                     try:
#                         if model in ["ankh-base", "ankh-large"]:
#                             self.compute_ankh_embeddings()
#                         else:
#                             self.compute_biotransformer_embeddings()
#                     except Exception as e:
#                         self.log.error(f"Error computing embeddings for {model}: {e}")
#
#         # Attempt to load processed data
#         if not self.try_loading_data(
#             self.base_path,
#             f"{self.data_name}_proteins_{self.embedding_type}",
#             possible_exts,
#             **kwargs,
#         ):
#             self.log.info(
#                 f"No existing processed data found for {self.embedding_type}."
#             )
#             # If no processed data, attempt to load unprocessed data
#             if not self.try_loading_data(
#                 self.base_path, f"{self.data_name}_proteins", possible_exts, **kwargs
#             ):
#                 self.log.info("No existing unprocessed data found.")
#                 # Fallback to filepath if no data found using data_name
#                 if self.filepath:
#                     self.df = load_df(self.filepath, **kwargs)
#                     self.log.info(f"Loaded data from filepath {self.filepath}")
#                 else:
#                     raise FileNotFoundError(
#                         "No existing data found. Please check the data_name or filepath."
#                     )
#
#     def try_loading_data(self, embedding_type, **kwargs):
#         """
#         Attempts to load data from a file matching the given pattern and extensions.
#         Returns True if data was successfully loaded, False otherwise.
#         """
#         # TODO this logic to be moved to utils_data or the data itself
#         ## in case all embeddings are to be computed
#         if embedding_type == "all":
#             for model in self.all_models:
#                 if not self.try_loading_data(model, **kwargs):
#                     return False
#
#         for ext in kwargs.get("possible_exts", ["csv", "pkl", "parquet"]):
#             file_path = os.path.join(
#                 self.base_path, f"{self.data_name}_proteins_{embedding_type}.{ext}"
#             )
#             if os.path.exists(file_path):
#                 self.df = load_df(file_path, **kwargs)
#                 self.log.info(f"Loaded data from {file_path}")
#                 return True
#         return False
#
#     def export_dataframe(self, ext="pkl", **kwargs):
#         """Export the updated DataFrame to the same directory as the input file with an updated filename."""
#
#         if self.df is not None:
#             self.log.info("Exporting DataFrame...")
#             filename = f"{self.data_name}_proteins_{self.embedding_type}"
#             export_df(self.df, self.base_path, filename, ext, **kwargs)
#
#         # # Derive directory and original filename
#         # dir_path, original_filename = os.path.split(self.filepath)
#         # filename_without_extension = os.path.splitext(original_filename)[0]
#         #
#         # # Create an updated filename that reflects the embedding type
#         # updated_filename = (
#         #     f"{filename_without_extension}_{self.embedding_type}_embeddings.parquet"
#         # )
#         #
#         # # Construct the full path for the output file
#         # output_file_path = os.path.join(dir_path, updated_filename)
#         #
#         # # Export DataFrame as Parquet
#         # self.df.to_parquet(output_file_path, index=False)
#         # self.log.info(f"DataFrame exported to {output_file_path}")
#
#
# # Note: The implementation of compute_popular_embeddings, compute_all_embeddings, and merge_embeddings
# # would require further detail based on the specifications and the intended output format for the embeddings.
#
# # import numpy as np
# # import pandas as pd
# # import torch
# #
# # #
# # # from bio_embeddings.embed import BeplerEmbedder, CPCProtEmbedder, EmbedderInterface, FastTextEmbedder, GloveEmbedder
# # # from bio_embeddings.embed import OneHotEncodingEmbedder, PLUSRNNEmbedder, ProtTransAlbertBFDEmbedder
# # # from bio_embeddings.embed import ProtTransT5BFDEmbedder, ProtTransT5UniRef50Embedder, ProtTransT5XLU50Embedder
# # # from bio_embeddings.embed import ProtTransXLNetUniRef100Embedder, SeqVecEmbedder, UniRepEmbedder, Word2VecEmbedder
# # #
# #
# #
# # from biotransformers import BioTransformers
# # import ray
# #
# # ray.init()
# # sequences = [
# #     "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
# #     "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
# # ]
# #
# # bio_trans = BioTransformers(
# #     backend="here we should write the type of embedding",
# #     num_gpus=torch.cuda.device_count(),
# # )
# # embeddings = bio_trans.compute_embeddings(sequences, pool_mode=("mean"), batch_size=8)
# #
# #
# # import ankh
# #
# # model, tokenizer = ankh.load_base_model()
# # model.eval()
# #
# # protein_sequences = [
# #     "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
# #     "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
# # ]
# # # protein_sequences = [list(seq) for seq in protein_sequences]
# # outputs = tokenizer.batch_encode_plus(
# #     protein_sequences,
# #     add_special_tokens=True,
# #     padding=True,
# #     # max_length=1024,  # hyperparameter
# #     # is_split_into_words=True,
# #     return_tensors="pt",
# # )
# #
# # with torch.no_grad():
# #     embeddings = model(
# #         input_ids=outputs["input_ids"],
# #         attention_mask=outputs["attention_mask"],
# #     )
#
# #
# # # Search for processed data across all extensions
# # for ext in possible_exts:
# #     processed_path = (
# #         os.path.join(
# #             base_path, f"{self.data_name}_proteins_{self.embedding_type}.{ext}"
# #         )
# #         if self.data_name
# #         else ""
# #     )
# #     if os.path.exists(processed_path):
# #         self.df = load_df(processed_path)
# #         self.log.info(f"Loaded existing processed data from {processed_path}")
# #         processed_found = True
# #         break
# #
# # # If processed data not found, search for unprocessed data
# # if not processed_found:
# #     for ext in possible_exts:
# #         unprocessed_path = (
# #             os.path.join(base_path, f"{self.data_name}_proteins.{ext}")
# #             if self.data_name
# #             else ""
# #         )
# #         if os.path.exists(unprocessed_path):
# #             self.filepath = unprocessed_path
# #             self.df = load_df(self.filepath)
# #             self.log.info(f"Loaded unprocessed data from {self.filepath}")
# #             break
# #
# #     if self.df is None and self.filepath:
# #         # Fallback to filepath if no unprocessed data found in data_name directory
# #         self.df = load_df(self.filepath)
# #         self.log.info(f"Loaded data from filepath {self.filepath}")
# #
# #     if self.df is None:  # No data found, raise error
# #         raise FileNotFoundError(
# #             "No existing data found. Please check the data_name or filepath."
# #         )
#
# # def load_data(self):
# #     """Load protein sequences from the specified CSV file."""
# #     try:
# #         self.df = pd.read_csv(self.filepath)
# #         assert (
# #             self.sequence_col in self.df.columns
# #         ), f"{self.sequence_col} not found in CSV."
# #     except Exception as e:
# #         self.log.error(f"Failed to load data: {e}")
#
# # def compute_popular_embeddings(self):
# #     """Compute embeddings for the popular models."""
# #     # Recursion is not the best here for simplicity, memory efficiency, and performance considerations.
# #     popular_models = ["esm1b", "esm_msa1b", "protbert", "ankh-base", "ankh-large"]
# #     for model in popular_models:
# #         self.log.info(f"Computing embeddings for {model}")
# #         if model in ["ankh-base", "ankh-large"]:
# #             self.embedding_type = model
# #             self.compute_ankh_embeddings()
# #         else:
# #             self.embedding_type = model
# #             self.compute_biotransformer_embeddings()
#
#
# #
# # ### Here we should implement the embeddings and protein featurizers functionalities
# #
# # # Prottrans -> bio-embeddings
# # import torch
# # from transformers import AutoTokenizer, AutoModel
# # import ankh
# # from uqdd import DEVICE
# #
# #
# # def transformer_featurizer(
# #         input_str,
# #         model_name="dmis-lab/biobert-v1.1"
# # ):
# #     tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=1024)
# #     model = AutoModel.from_pretrained(model_name)
# #     tokens = tokenizer(input_str, return_tensors="pt")
# #     outputs = model(**tokens)
# #     embeddings = outputs.last_hidden_state.mean(dim=1)
# #     return embeddings.detach().cpu().numpy()
# #
# #
# # # ESM -> bio-embeddings or Otto-KG
# # class ESMProtein(torch.nn.Module):
# #     def __init__(self, repo_or_dir, model, repr_layer, device='cpu'):
# #         super(ESMProtein, self).__init__()
# #         self.device = device
# #         self.repo_or_dir = repo_or_dir
# #         self.model = model
# #         self.repr_layer = repr_layer
# #         self._model = None
# #         self._batch_converter = None
# #         self.to(self.device)
# #
# #     def load_model(self):
# #         self._model, alphabet = torch.hub.load(self.repo_or_dir, self.model)
# #         self._batch_converter = alphabet.get_batch_converter(truncation_seq_length=1022)
# #         self._model.eval()
# #         self._model.forward_original = self._model.forward
# #
# #     def get_embeddings(self, sequences):
# #         if not self._model:
# #             self.load_model()
# #         ids = ['ids_' + str(i) for i in range(len(sequences))]
# #         _, _, tensors = self._batch_converter(list(zip(ids, sequences)))
# #         results = self._model.forward_original(tensors, repr_layers=[self.repr_layer], return_contacts=True)
# #         token_representations = results["representations"][self.repr_layer]
# #
# #         # Generate per-sequence representations via averaging
# #         # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
# #         sequence_representations = []
# #         for i, seq in enumerate(results['contacts']):
# #             seq_len = (tensors[i] == 2).nonzero()
# #             sequence_representations.append(token_representations[i, 1: seq_len].mean(0))
# #
# #         return torch.stack(sequence_representations)
# #
# # # ProteinBERT -> bio-embeddings as part of protTrans
# # def get_protbert(df, sequences_col='sequence'):
# #     raise NotImplementedError
# #
# # # MSA ->
# # def get_msa(df, sequences_col='sequence'):
# #     raise NotImplementedError
# #
# #
# # # ProteinLM ->
# # def get_proteinlm(df, sequences_col='sequence'):
# #     raise NotImplementedError
# #
# #
# # # TAPE ->
# # def get_tape(df, sequences_col='sequence'):
# #     raise NotImplementedError
# #
# # def get_prot_desc(
# #         df,
# #         sequences_col='sequence',
# #         desc_prot='esm1b',
# #         **kwargs
# # ):
# #     desc_prot = desc_prot.lower()
# #
# #     if desc_prot.startswith('ankh'):
# #         if desc_prot == 'ankh-base':
# #             model, tokenizer = ankh.load_base_model()
# #             model.eval()
# #             model.to(DEVICE)
# #             outputs = tokenizer.batch_encode_plus(
# #                 df[sequences_col].tolist(),
# #                 add_special_tokens=True,
# #                 padding=True,
# #                 is_split_into_words=True,
# #                 return_tensors="pt"
# #             )
# #
# #             with torch.no_grad():
# #                 embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])
# #
# #         elif desc_prot == 'ankh-large':
# #             model, tokenizer = ankh.load_large_model()
# #             model.eval()
# #             model.to(DEVICE)
# #             outputs = tokenizer.batch_encode_plus(
# #                 df[sequences_col].tolist(),
# #                 add_special_tokens=True,
# #                 padding=True,
# #                 is_split_into_words=True,
# #                 return_tensors="pt"
# #             )
# #             with torch.no_grad():
# #                 embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])
# #
# #     elif desc_prot == 'esm1b':
# #         raise NotImplementedError
# #
# #     elif desc_prot == 'protbert':
# #         raise NotImplementedError
# #
# #     elif desc_prot == 'msa':
# #         raise NotImplementedError
# #
# #     elif desc_prot == 'proteinlm':
# #         raise NotImplementedError
# #
# #     elif desc_prot == 'tape':
# #         raise NotImplementedError
# #
# #     elif desc_prot == 'ankh-base':
# #         model, tokenizer = ankh.load_base_model()
# #         model.eval()
# #         model.to(DEVICE)
# #         outputs = tokenizer.batch_encode_plus(
# #             df[sequences_col].tolist(),
# #             add_special_tokens=True,
# #             padding=True,
# #             is_split_into_words=True,
# #             return_tensors="pt"
# #         )
# #
# #         with torch.no_grad():
# #             embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])
# #
# #     elif desc_prot == 'ankh-large':
# #         model, tokenizer = ankh.load_large_model()
# #         model.eval()
# #         model.to(DEVICE)
# #         outputs = tokenizer.batch_encode_plus(
# #             df[sequences_col].tolist(),
# #             add_special_tokens=True,
# #             padding=True,
# #             is_split_into_words=True,
# #             return_tensors="pt"
# #         )
# #         with torch.no_grad():
# #             embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])
# #
# #     return embeddings
