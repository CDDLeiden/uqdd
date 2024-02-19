### Here we should implement the embeddings and protein featurizers functionalities

# Prottrans -> bio-embeddings
import torch
from transformers import AutoTokenizer, AutoModel
import ankh
from uqdd import DEVICE


def transformer_featurizer(
        input_str,
        model_name="dmis-lab/biobert-v1.1"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=1024)
    model = AutoModel.from_pretrained(model_name)
    tokens = tokenizer(input_str, return_tensors="pt")
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().cpu().numpy()


# ESM -> bio-embeddings or Otto-KG
class ESMProtein(torch.nn.Module):
    def __init__(self, repo_or_dir, model, repr_layer, device='cpu'):
        super(ESMProtein, self).__init__()
        self.device = device
        self.repo_or_dir = repo_or_dir
        self.model = model
        self.repr_layer = repr_layer
        self._model = None
        self._batch_converter = None
        self.to(self.device)

    def load_model(self):
        self._model, alphabet = torch.hub.load(self.repo_or_dir, self.model)
        self._batch_converter = alphabet.get_batch_converter(truncation_seq_length=1022)
        self._model.eval()
        self._model.forward_original = self._model.forward

    def get_embeddings(self, sequences):
        if not self._model:
            self.load_model()
        ids = ['ids_' + str(i) for i in range(len(sequences))]
        _, _, tensors = self._batch_converter(list(zip(ids, sequences)))
        results = self._model.forward_original(tensors, repr_layers=[self.repr_layer], return_contacts=True)
        token_representations = results["representations"][self.repr_layer]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, seq in enumerate(results['contacts']):
            seq_len = (tensors[i] == 2).nonzero()
            sequence_representations.append(token_representations[i, 1: seq_len].mean(0))

        return torch.stack(sequence_representations)

# ProteinBERT -> bio-embeddings as part of protTrans
def get_protbert(df, sequences_col='sequence'):
    raise NotImplementedError

# MSA ->
def get_msa(df, sequences_col='sequence'):
    raise NotImplementedError


# ProteinLM ->
def get_proteinlm(df, sequences_col='sequence'):
    raise NotImplementedError


# TAPE ->
def get_tape(df, sequences_col='sequence'):
    raise NotImplementedError

def get_prot_desc(
        df,
        sequences_col='sequence',
        desc_prot='esm1b',
        **kwargs
):
    desc_prot = desc_prot.lower()

    if desc_prot.startswith('ankh'):
        if desc_prot == 'ankh-base':
            model, tokenizer = ankh.load_base_model()
            model.eval()
            model.to(DEVICE)
            outputs = tokenizer.batch_encode_plus(
                df[sequences_col].tolist(),
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])

        elif desc_prot == 'ankh-large':
            model, tokenizer = ankh.load_large_model()
            model.eval()
            model.to(DEVICE)
            outputs = tokenizer.batch_encode_plus(
                df[sequences_col].tolist(),
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])

    elif desc_prot == 'esm1b':
        raise NotImplementedError

    elif desc_prot == 'protbert':
        raise NotImplementedError

    elif desc_prot == 'msa':
        raise NotImplementedError

    elif desc_prot == 'proteinlm':
        raise NotImplementedError

    elif desc_prot == 'tape':
        raise NotImplementedError

    elif desc_prot == 'ankh-base':
        model, tokenizer = ankh.load_base_model()
        model.eval()
        model.to(DEVICE)
        outputs = tokenizer.batch_encode_plus(
            df[sequences_col].tolist(),
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])

    elif desc_prot == 'ankh-large':
        model, tokenizer = ankh.load_large_model()
        model.eval()
        model.to(DEVICE)
        outputs = tokenizer.batch_encode_plus(
            df[sequences_col].tolist(),
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])

    return embeddings