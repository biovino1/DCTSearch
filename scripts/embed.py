"""Defines the Transform class, which is used to embed protein sequences using the
ESM-2_t36_3B protein language model and then perform iDCT quantization.

__author__ = "Ben Iovino"
__date__ = "12/18/23"
"""

from dataclasses import dataclass, field
import esm
import torch
import numpy as np
from scipy.fft import dct, idct


class Model:
    """Stores model and tokenize for embedding sequences.
    """

    def __init__(self, model: str):
        """Model contains encoder and tokenizer.

        Args:
            model (str): Model to use for embedding. Currently only supports ESM-2.
        """

        if model == 'esm2':
            self.load_esm2()


    def load_esm2(self):
        """Loads ESM-2 model.
        """

        self.encoder, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.tokenizer = alphabet.get_batch_converter()
        self.encoder.eval()


    def to_device(self, device: str):
        """Moves model to device.

        Args:
            device (str): gpu/cpu
        """

        self.encoder.to(device)


@dataclass
class Transform:
    """This class stores the inverse discrete cosine transform (iDCT) quantization of
    an embedded protein sequence.
    """
    pid: str = field(default_factory=str)
    seq: str = field(default_factory=str)
    embed: dict = field(default_factory=dict)
    quant: np.array = field(default_factory=list)


    def __post_init__(self):
        """Initialize quant to empty array.
        """

        self.quant = np.array([])


    def esm2_embed(self, model: Model, device: str, layers: list):
        """Returns embedding of a protein sequence. Each vector represents a single amino
        acid using Facebook's ESM2 model.

        Args:
            model (Model): Model class with encoder and tokenizer.
            device (str): gpu/cpu
            layer (list: List of layers to embed with.
        """

        # Embed sequences
        self.seq = self.seq.upper()  # tok does not convert to uppercase
        embed = [np.array([self.pid, self.seq], dtype=object)]  # for tokenizer
        _, _, batch_tokens = model.tokenizer(embed)
        batch_tokens = batch_tokens.to(device)  # send tokens to gpu
        with torch.no_grad():
            results = model.encoder(batch_tokens, repr_layers=layers)

        # Store embedding from each layer
        embed = {}
        for layer in layers:
            emb = results["representations"][layer].cpu().numpy()
            embed[layer] = emb[0][1:-1]
        self.embed = embed


    def scale(self, vec: np.ndarray) -> np.ndarray:
        """Scale from protsttools. Takes a vector and returns it scaled between 0 and 1.

        Args:
            vec (np.ndarray): Vector to be scaled.

        Returns:
            np.ndarray: Scaled vector.
        """

        maxi = np.max(vec)
        mini = np.min(vec)

        return (vec - mini) / float(maxi - mini)


    def idct_quant(self, vec: np.ndarray, num: int) -> np.ndarray:
        """iDCTquant from protsttools. Takes a vector and returns the iDCT of the DCT.

        Args:
            vec (np.ndarray): Vector to be transformed.
            num (int): Number of coefficients to keep.

        Returns:
            np.ndarray: Transformed vector.
        """

        f = dct(vec.T, type=2, norm='ortho')
        trans = idct(f[:,:num], type=2, norm='ortho')  #pylint: disable=E1126
        for i in range(len(trans)):  #pylint: disable=C0200
            trans[i] = self.scale(trans[i])  #pylint: disable=E1137

        return trans.T  #pylint: disable=E1101


    def quantize(self, qdim: list):
        """quant2D from protsttools. Takes an embedding(s) and returns the flattened iDCT
        quantization on both axes.

        Args:
            qdim (list): List of quantization dimensions. Even indices are for the first
                        axis and odd indices are for the second axis.
        """

        # Perform iDCT quantization on each layer
        for i, embed in enumerate(self.embed.values()):
            n_dim, m_dim = qdim[i*2], qdim[i*2+1]
            dct = self.idct_quant(embed[1:len(embed)-1], n_dim)  #pylint: disable=W0621
            ddct = self.idct_quant(dct.T, m_dim).T
            ddct = ddct.reshape(n_dim * m_dim)
            self.quant = np.append(self.quant, ddct)
