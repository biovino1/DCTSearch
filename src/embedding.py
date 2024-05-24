"""Defines the Model, Embedding, and Batch classes, which are used to embed protein sequences.

__author__ = "Ben Iovino"
__date__ = "2/19/24"
"""

from dataclasses import dataclass, field
import logging
import os
import esm
import re
import torch
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np

logging.getLogger("transformers").setLevel(logging.ERROR)


class Model:
    """Stores encoders and tokenizers for embedding sequences.

    Attributes:
        esm_encoder (esm.model.ProteinBertModel): ESM-2 model.
        esm_tokenizer (esm.model.Alphabet): ESM-2 tokenizer.
        pt5_encoder (T5EncoderModel): ProtT5-XL-U50 model.
        pt5_tokenizer (T5Tokenizer): ProtT5-XL-U50 tokenizer.
    """

    def __init__(self):
        """Model contains encoder and tokenizer for both ESM-2 and ProtT5. Checkpoints are set
        after testing and are not meant to be changed, but can be done manually if desired.
        """

        self.load_esm2('t30')
        self.load_pt5('xl_u50')


    def load_esm2(self, checkpoint: str):
        """Loads ESM-2 model.

        Args:
            checkpoint (str): Model checkpoint to load.
        """

        if checkpoint == 't33':
            self.esm_encoder, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        if checkpoint == 't30':
            self.esm_encoder, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()

        self.esm_tokenizer = self.alphabet.get_batch_converter()
        self.esm_encoder.eval()

    
    def load_pt5(self, checkpoint: str):
        """Loads ProtT5-XL-U50 model.

        Args:
            checkpoint (str): Model checkpoint to load.
        """

        if checkpoint == 'xl_u50':
            self.pt5_encoder = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50', output_hidden_states=True)
        self.pt5_tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', legacy=True)


    def to_device(self, device: str):
        """Moves model to device.

        Args:
            device (str): gpu/cpu
        """

        self.esm_encoder.to(device)
        self.pt5_encoder.to(device)


@dataclass
class Embedding:
    """This class stores the necessary information for a protein sequence to be embedded.
    An Embedding is meant to be used in the Fingerprint class for quantization.

    Attributes:
        pid (str): Protein ID.
        seq (str): Protein sequence.
        embed (dict): Dictionary of embeddings from each layer.
        contacts (np.array): Contact map.
    """
    pid: str = field(default_factory=str)
    seq: str = field(default_factory=str)
    embed: dict = field(default_factory=dict)
    contacts: np.array = field(default_factory=list)


    def __post_init__(self):
        """Initialize contacts to array.
        """

        self.contacts = np.array([])


    def split_seq(self, maxlen: int, overlap: int) -> list:
        """Splits a sequence into smaller sequences of length maxlen with overlap.

        Args:
            maxlen (int): Maximum length of each sub-sequence
            overlap (int): Overlap between sub-sequences

        Returns:
            list: List of sub-sequences
        """

        subseqs = []
        for i in range(0, len(self.seq), maxlen-overlap):
            subseq = self.seq[i:i+maxlen]
            if len(subseq) > overlap:  # skip if subseq is too short to be unique
                subseqs.append(subseq)

        return subseqs


    def extract_esm2(self, seq: str, model: Model, device: str) -> np.ndarray:
        """Returns a dictionary containing contact map from ESM-2.

        Args:
            seq (str): Protein sequence.
            model (Model): Model class with encoder and tokenizer.
            device (str): gpu/cpu

        Returns:
            np.ndarray: Contact map (nxn) of protein sequence.
        """

        _, _, batch_tokens = model.esm_tokenizer([(self.pid, seq)])
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model.esm_encoder(batch_tokens, return_contacts=True)
        
        return results["contacts"][0]


    def extract_pt5(self, seq: str, model: Model, device: str, layers: list) -> dict:
        """Returns a dictionary containing embeddings from ProtT5-XL-U50.

        Args:
            seq (str): Protein sequence.
            model (Model): Model class with encoder and tokenizer.
            device (str): gpu/cpu
            layers (list): List of layers to extract embeddings from.

        Returns:
            dict: Dictionary of embeddings from ProtT5-XL-U50.
        """

        seq = re.sub(r"[UZOB]", "X", seq)
        seq = [' '.join([*seq])]

        # Tokenize, encode, and load sequence
        ids = model.pt5_tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)  # pylint: disable=E1101
        attention_mask = torch.tensor(ids['attention_mask']).to(device)  # pylint: disable=E1101

        # Extract final layer of model
        with torch.no_grad():
            outputs = model.pt5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        embs = {}
        for layer in layers:
            embs[layer] = outputs.hidden_states[layer][0]

        return embs


    def combine_contacts(self, mat1: torch.Tensor, mat2: torch.Tensor, inc: int, times: int) -> torch.Tensor:
        """Returns a larger square matrix combining two smaller square matrices.

        Mat1 and mat2 are both square matrices. New matrix (mat3) is of size (n+m, n+m) where
        n and m are the dimensions of mat1 and mat2. Overlapping indices are averaged.

        Args:
            mat1 (torch.Tensor): Running matrix of contacts (n x n).
            mat2 (torch.Tensor): Matrix to be added to mat1 (m x m).
            inc (int): Number of indices to increase the size of the matrix by (inc x inc).
            times (int): Running total of times the function has been called.

        Returns:
            torch.Tensor: Matrix of combined contacts (n+inc x n+inc).
        """

        # Add space to mat1
        olp = inc * times
        mlen1, mlen2 = mat1.size(0), mat2.size(0)
        mlen3 = olp + mlen2
        new_mat = torch.zeros((mlen3, mlen3), device=mat1.device)
        new_mat[:mlen1, :mlen1] = mat1

        # Add mat2 to new_mat
        new_mat[olp:mlen3, olp:mlen3] = new_mat[olp:mlen3, olp:mlen3] + mat2
        new_mat[olp:mlen1, olp:mlen1] = new_mat[olp:mlen1, olp:mlen1] / 2

        return new_mat


    def embed_seq(self, model: Model, device: str, layers: list, maxlen: int):
        """Returns ESM-2 embedding and contact map of a protein sequence.

        Args:
            pt_model (Model): Model class containing loaded ESM2 and ProtT5 models.
            device (str): gpu/cpu
            layer (list: List of layers to extract embeddings from.
            maxlen (int): Maximum length of sequence to embed
        """

        olp = 200  # overlap between sub-sequences
        if len(self.seq) > maxlen:
            subseqs = self.split_seq(maxlen, olp)
        else:  # still need to make it a list for the loop
            subseqs = [self.seq]

        # Extract embeddings and contact maps for each subsequence
        edata = {}
        for i, seq in enumerate(subseqs):
            embs = self.extract_pt5(seq, model, device, layers)
            ct = self.extract_esm2(seq, model, device)

            # If first subsequence, initialize edata
            if not edata:
                edata = embs
                edata['ct'] = ct
                continue

            # Average overlapping positions between previous and current embeddings
            for lay, emb in embs.items():
                if lay == 'ct':
                    edata[lay] = self.combine_contacts(edata[lay], emb, maxlen-olp, i)
                    continue
                edata[lay][-olp:] = (edata[lay][-olp:] + emb[:olp]) / 2
                edata[lay] = torch.cat((edata[lay], emb[olp:]), axis=0)

        self.embed = {k: v.cpu().numpy() for k, v in edata.items() if k in layers}
        self.contacts = edata['ct'].cpu().numpy()


@dataclass
class Batch:
    """This class embeds a batch of protein sequences. Sequences can be embedded one at a time, but
    parallel embedding of multiple sequences can be faster if using a GPU. The embedding of single
    sequences is handled by methods in the Embedding class.

    Attributes:
        seqs (list): List of tuples containing (protein id, sequence)
        model (Model): Model object containing encoder, alphabet, and tokenizer
        device (str): gpu/cpu
        embeds (list): List of Embedding objects
    """
    seqs: list = field(default_factory=list)
    model: Model = field(default_factory=Model)
    device: str = field(default_factory=str)
    embeds: list = field(default_factory=list)


    def embed_batch(self, layers: list, maxlen: int):
        """Embeds a batch of protein sequences.

        Args:
            layers (list): List of layers to extract embeddings from.
            maxlen (int): Maximum length of sequence to embed.
        """

        if len(self.seqs) == 1:
            self.embed_single(layers, maxlen)
        else:
            self.embed_parallel(layers)


    def embed_single(self, layers: list, maxlen: int):
        """Embeds a single protein sequence.

        Args:
            layers (list): List of layers to extract embeddings from.
            maxlen (int): Maximum length of sequence to embed.
        """

        for seq in self.seqs:
            emb = Embedding(pid=seq[0], seq=seq[1])
            emb.embed_seq(self.model, self.device, layers, maxlen)
            self.embeds.append(emb)


    def embed_parallel(self, layers: list):
        """Embeds a batch of protein sequences in parallel.

        Args:
            layers (list): List of layers to extract embeddings from.
        """

        _, _, batch_tokens = self.model.tokenizer(self.seqs)
        batch_lens = (batch_tokens != self.model.alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(self.device)

        # Embed sequences and parse results into individual Embedding objects
        with torch.no_grad():
            results = self.model.encoder(batch_tokens, repr_layers=layers, return_contacts=True)
        for i, seq in enumerate(self.seqs):
            emb = Embedding(pid=seq[0], seq=seq[1])
            for layer, embed in results["representations"].items():
                emb.embed[layer] = embed[i][1:batch_lens[i]-1].cpu().numpy()
            emb.contacts = results["contacts"][i][:batch_lens[i]-2, :batch_lens[i]-2].cpu().numpy()
            self.embeds.append(emb)
        

    def save_batch(self, direc: str):
        """Saves individual embeddings to the given directory

        Args:
            filename (str): Name of file to save embeddings to.
        """

        if not self.embeds:
            print('No embeddings to save')
            return
        if not os.path.exists(direc):
            os.makedirs(direc)
        for emb in self.embeds:
            np.savez_compressed(f'{direc}/{emb.pid}', pid=emb.pid,
                                 seq=emb.seq, embed=emb.embed, contacts=emb.contacts)
