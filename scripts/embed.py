"""Defines the Transform class, which is used to embed protein sequences using the
ESM-2_t36_3B protein language model and then perform iDCT quantization.

__author__ = "Ben Iovino"
__date__ = "12/18/23"
"""

from dataclasses import dataclass
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

        self.encoder, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
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
    id: str = ''
    seq: str = ''
    embed: np.ndarray = None
    quant: np.ndarray = None


    def esm2_embed(self, model: Model, device: str, layer: int):
        """Returns embedding of a protein sequence. Each vector represents a single amino
        acid using Facebook's ESM2 model.

        Args:
            model (Model): Model class with encoder and tokenizer
            device (str): gpu/cpu
            layer (int): layer of ESM2 to use for embedding
        """

        # Embed sequences
        self.seq = self.seq.upper()  # tok does not convert to uppercase
        embed = [np.array([self.id, self.seq], dtype=object)]  # for tokenizer
        _, _, batch_tokens = model.tokenizer(embed)
        batch_tokens = batch_tokens.to(device)  # send tokens to gpu

        with torch.no_grad():
            results = model.encoder(batch_tokens, repr_layers=[layer])
        embed = results["representations"][layer].cpu().numpy()
        self.embed = embed[0][1:-1]  # remove beginning and end tokens


    def scale(self, vec: np.ndarray) -> np.ndarray:
        """Scale from protsttools. Takes a vector and returns it scaled between 0 and 1.

        Args:
            vec (np.ndarray): vector to be scaled

        Returns:
            np.ndarray: scaled vector 
        """

        maxi = np.max(vec)
        mini = np.min(vec)

        return (vec - mini) / float(maxi - mini)


    def idct_quant(self, vec: np.ndarray, num: int) -> np.ndarray:
        """iDCTquant from protsttools. Takes a vector and returns the iDCT of the DCT.

        :param vec: vector to be transformed
        :param num: number of coefficients to keep
        :return: transformed vector
        """

        f = dct(vec.T, type=2, norm='ortho')
        trans = idct(f[:,:num], type=2, norm='ortho')
        for i in range(len(trans)):  #pylint: disable=C0200
            trans[i] = self.scale(trans[i])

        return trans.T


    def quant_2d(self, n_dim: int, m_dim: int):
        """quant2D from protsttools. Takes an embedding and returns the iDCT quantization
        on both axes.

        :param emb: embedding to be transformed (n x m array)
        :param n_dim: number of coefficients to keep on first axis
        :param m_dim: number of coefficients to keep on second axis
        """

        dct = self.idct_quant(self.embed[1][1:len(self.embed[1])-1], n_dim)  #pylint: disable=W0621
        ddct = self.idct_quant(dct.T, m_dim).T
        try:
            ddct = ddct.reshape(n_dim * m_dim)
            self.quant = (ddct*127).astype('int8')
        except ValueError:  # If embedding is too small to transform
            self.quant = None
