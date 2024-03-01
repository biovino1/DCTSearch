"""Defines the Model and Embedding classes, which are used to embed protein sequences using ESM-2
protein language models.

__author__ = "Ben Iovino"
__date__ = "2/19/24"
"""

from dataclasses import dataclass, field
import os
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
        """

        if checkpoint == 't33':
            self.encoder, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        if checkpoint == 't30':
            self.encoder, alphabet = esm.pretrained.esm2_t30_150M_UR50D()

        self.tokenizer = alphabet.get_batch_converter()
        self.encoder.eval()


    def to_device(self, device: str):
        """Moves model to device.

        Args:
            device (str): gpu/cpu
        """

        self.encoder.to(device)


@dataclass
class Embedding:
    """This class creates and stores the necessary information for protein sequences to be
    embedded. An Embedding object is meant to be used in the Fingerprint class for quantization.
    Embeddings can also be saved and loaded from file (npz).

    Any number of sequences can be embedded at once to maximize GPU usage, however, make sure
    you do not overload the GPU memory with too many sequences.

    Attributes:
        pid (list): Protein IDs.
        seq (list): Protein sequences.
        embed (list): Dictionaries of embeddings from each layer.
        contacts (list): Numpy arrays of contact maps.
        model (Model): Model class with encoder and tokenizer.
        device (str): gpu/cpu
        layers (list): List of layers to extract embeddings from.
    """
    pid: list = field(default_factory=list)
    seq: list = field(default_factory=list)
    embed: list = field(default_factory=list)
    contacts: list = field(default_factory=list)
    model: Model = field(default=None)
    device: str = field(default='cpu')
    layers: list = field(default_factory=list)


    def split_seq(self, maxlen: int, olp: int):
        """Splits a sequence into smaller sequences of length maxlen with overlap and embeds
        each subsequence. Overlapping regions in the embeddings and contact maps are averaged.

        Args:
            maxlen (int): Maximum length of each sub-sequence
            olp (int): Overlap between sub-sequences
        """

        subseqs = []
        for i in range(0, len(self.seq[0]), maxlen-olp):  # only one seq in list
            subseq = self.seq[0][i:i+maxlen]
            if len(subseq) > olp:  # skip if subseq is too short to be unique
                subseqs.append(subseq)

         # Extract embeddings and contact maps for each subsequence
        edata = {}
        for i, seq in enumerate(subseqs):
            embs = [(self.pid, seq)]
            embs = self.extract_embeds(embs)
            if not edata:  # dynamic initialization in case different layers are used
                edata = embs[0]  # only one seq in dictionary
                continue

            # Average overlapping positions between previous and current embeddings
            for lay, emb in embs[0].items():
                if lay == 'ct':
                    edata[lay] = self.combine_contacts(edata[lay], emb, maxlen-olp, i)
                    continue
                edata[lay][-olp:] = (edata[lay][-olp:] + emb[:olp]) / 2
                edata[lay] = np.concatenate((edata[lay], emb[olp:]), axis=0)

        # Store embeddings and contact maps
        self.embed = [{k: v for k, v in edata.items() if k in self.layers}]
        self.contacts = [edata['ct']]


    def extract_embeds(self, embs: list) -> dict:
        """Returns a dictionary containing embeddings for every sequence in the object. Each
        sequence will contain a dictionary of embeddings from each layer and the contact map.

        Args:
            embs (list): List of tuples of (protein ID, sequence) to embed.
                ex. [(pid1, seq1), (pid2, seq2), ...]

        Returns:
            dict: Nested dictionary of embeddings and contact maps.
                ex. {0: {'ct': np.array, 'layer1': np.array, 'layer2': np.array, ...},
                        1: {'ct': np.array, 'layer1': np.array, 'layer2': np.array, ...},
                        ...}
        """

        # Tokenize batch and count length of each sequence
        _, _, batch_tokens = self.model.tokenizer(embs)
        batch_tokens = batch_tokens.to(self.device)
        lengths = []
        for batch in batch_tokens:
            lengths.append(batch.ne(1).sum()-2)  # don't include padding, <bos> and <eos> tokens
        try:
            with torch.no_grad():
                results = self.model.encoder(batch_tokens,
                                              repr_layers=self.layers, return_contacts=True)
        except RuntimeError:
            print(f'Error embedding {self.pid}')
            return {}

        # Store results for each sequence
        embs = {}
        for i in (range(len(self.seq))):  #pylint: disable=C0200
            embs[i] = {'ct': results["contacts"].cpu().numpy()[i][:lengths[i], :lengths[i]]}
        for layer in self.layers:
            for scount, emb in enumerate(results["representations"][layer].cpu().numpy()):
                embs[scount][layer] = emb[1:lengths[scount]+1] # remove <bos> token

        return embs


    def combine_contacts(self, mat1: np.ndarray, mat2: np.ndarray, inc: int, times: int) -> np.ndarray:
        """Returns a larger square matrix combining two smaller square matrices.

        mat1 has values starting from the top left corner, mat2 has values starting from the bottom
        right corner, and the overlapping indices are averaged. The number of overlapping indices is
        determine by the inc argument.

        Args:
            mat1 (numpy array): Running matrix of contacts (n x n).
            mat2 (numpy array): Matrix to be added to mat1 (m x m).
            inc (int): Number of indices to increase the size of the matrix by (inc x inc).
            times (int): Running total of times the function has been called.

        Returns:
            numpy array: Matrix of combined contacts (n+inc x n+inc).
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


    def embed_seq(self, maxlen: int):
        """Returns ESM-2 embedding and contact map of a protein sequence.

        Args:
            layers (list: List of layers to extract embeddings from.
            maxlen (int): Maximum length of sequence to embed
        """

        if len(self.seq) == 1 and len(self.seq[0]) > maxlen:
            olp = 200  # overlap between sub-sequences
            self.split_seq(maxlen, olp)
        else:
            embs = list(zip(self.pid, self.seq))
            embs = self.extract_embeds(embs)
            for _, emb in embs.items():
                self.embed.append({k: v for k, v in emb.items() if k in self.layers})
                self.contacts.append(emb['ct'])


    def save(self, direc: str):
        """Saves Embedding object to npz file.

        Args:
            direc (str): Directory to save embeddings to.
        """

        if not os.path.exists(direc):
            os.makedirs(direc)
        for i, pid in enumerate(self.pid):
            filename = f'{direc}/{pid}.npz'
            np.savez_compressed(filename, pid=pid, seq=self.seq[i],
                                embeds=self.embed[i], contacts=self.contacts[i])


    def load(self, filename: str):
        """Loads Embedding object from npz file.

        Args:
            filename (str): File to load embeddings from.
        """

        data = np.load(filename, allow_pickle=True)
        self.pid.append(data['pid'])
        self.seq.append(data['seq'])
        self.embed.append(data['embeds'].item())
        self.contacts.append(data['contacts'])
        data.close()
