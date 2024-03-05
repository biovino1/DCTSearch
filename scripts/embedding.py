"""Defines the Model and Embedding classes, which are used to embed protein sequences using ESM-2
protein language models.

__author__ = "Ben Iovino"
__date__ = "2/19/24"
"""

from dataclasses import dataclass, field
import esm
import torch
import numpy as np


class Model:
    """Stores model and tokenizer for embedding sequences.
    """

    def __init__(self, model: str, checkpoint: str):
        """Model contains encoder and tokenizer.

        Args:
            model (str): Model to use for embedding. Currently only supports ESM-2.
            checkpoint (str): Model checkpoint to load.
        """

        if model == 'esm2':
            self.load_esm2(checkpoint)


    def load_esm2(self, checkpoint: str):
        """Loads ESM-2 model.

        Args:
            checkpoint (str): Model checkpoint to load.
        """

        if checkpoint == 't33':
            self.encoder, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        if checkpoint == 't30':
            self.encoder, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()

        self.tokenizer = self.alphabet.get_batch_converter()
        self.encoder.eval()


    def to_device(self, device: str):
        """Moves model to device.

        Args:
            device (str): gpu/cpu
        """

        self.encoder.to(device)


@dataclass
class Embedding:
    """This class stores the necessary information for a protein sequence to be embedded.
    An Embedding is meant to be used in the Fingerprint class for quantization. Embeddings
    can also be saved and loaded from file (npz).

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


    def extract_embeds(self, seq: str, model: Model, device: str, layers: list) -> dict:
        """Returns a dictionary containing embeddings from each layer and the contact map.

        Args:
            seq (str): Protein sequence.
            model (Model): Model class with encoder and tokenizer.
            device (str): gpu/cpu
            layer (list: List of layers to extract embeddings from.

        Returns:
            dict: Dictionary containing embeddings from each layer and the contact map.
        """

        _, _, batch_tokens = model.tokenizer([(self.pid, seq)])
        batch_tokens = batch_tokens.to(device)
        try:
            with torch.no_grad():
                results = model.encoder(batch_tokens, repr_layers=layers, return_contacts=True)
        except RuntimeError:
            print(f'Error embedding {self.pid}, length {len(self.seq)}')
            return {}

        # Store results from each layer
        embs = {}
        for layer in layers:
            emb = results["representations"][layer]
            embs[layer] = emb[0][1:-1] # remove <cls> and <eos>
        embs['ct'] = results["contacts"][0]

        return embs


    def combine_contacts(self, mat1: torch.Tensor, mat2: torch.Tensor, inc: int, times: int) -> torch.Tensor:
        """Returns a larger square matrix combining two smaller square matrices.

        mat1 has values starting from the top left corner, mat2 has values starting from the bottom
        right corner, and the overlapping indices are averaged. The number of overlapping indices is
        determine by the inc argument.

        Args:
            mat1 (torch.Tensor): Running matrix of contacts (n x n).
            mat2 (torch.Tensor): Matrix to be added to mat1 (m x m).
            inc (int): Number of indices to increase the size of the matrix by (inc x inc).
            times (int): Running total of times the function has been called.

        Returns:
            torch.Tensor: Matrix of combined contacts (n+inc x n+inc).
        """

        # Create new matrices to store combined contacts
        mlen = len(mat1)
        size = mlen + inc
        zeros1 = np.zeros((size, size))
        zeros2 = np.zeros((size, size))

        # Add input matrices to new matrices
        olp = inc*times
        zeros1[:mlen, :mlen] = mat1
        zeros2[:len(mat2), :len(mat2)] = mat2
        zeros2 = np.roll(zeros2, olp, axis = 0)
        zeros2 = np.roll(zeros2, olp, axis = 1)

        # Average the overlapping indices
        zeros1[olp:mlen, olp:mlen] = (zeros1[olp:mlen, olp:mlen] + zeros2[olp:mlen, olp:mlen]) / 2

        # Add rest of zeros2 to zeros1
        zeros1[olp:size, mlen:size] = zeros2[olp:size, mlen:size]
        zeros1[mlen:size, olp:mlen] = zeros2[mlen:size, olp:mlen]

        # If any row or column is all 0's, remove it
        zeros1 = zeros1[~np.all(zeros1 == 0, axis=1)]
        zeros1 = zeros1[:, ~np.all(zeros1 == 0, axis=0)]

        return zeros1


    def embed_seq(self, model: Model, device: str, layers: list, maxlen: int):
        """Returns ESM-2 embedding and contact map of a protein sequence.

        Args:
            model (Model): Model class with encoder and tokenizer.
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
            embs = self.extract_embeds(seq, model, device, layers)
            if not edata:  # dynamic initialization in case different layers are used
                edata = embs
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


    def save(self, filename: str):
        """Saves Embedding object to npz file.

        Args:
            filename (str): File to save embeddings to.
        """

        np.savez_compressed(filename, pid=self.pid, seq=self.seq,
                             embeds=self.embed, contacts=self.contacts)


    def load(self, filename: str):
        """Loads Embedding object from npz file.

        Args:
            filename (str): File to load embeddings from.
        """

        data = np.load(filename, allow_pickle=True)
        self.pid = data['pid']
        self.seq = data['seq']
        self.embed = data['embeds']
        self.contacts = data['contacts']
        data.close()


@dataclass
class Batch:
    """This class embeds a batch of protein sequences. Sequences can be embedded one at a time, but
    parallel embedding of multiple sequences can be faster.

    Attributes:
        seqs (list): List of tuples containing (protein id, sequence)
        model (Model): Model object containing encoder, alphabet, and tokenizer
        device (str): gpu/cpu
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
        