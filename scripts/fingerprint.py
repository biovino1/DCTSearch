"""Defines the Fingerprint class, which is used to embed protein sequences using the
ESM-2_t36_3B protein language model, predict domains, and then perform iDCT quantization
on each one.

__author__ = "Ben Iovino"
__date__ = "12/18/23"
"""

from dataclasses import dataclass, field
from operator import itemgetter
import subprocess as sp
import esm
import torch
import numpy as np
from scipy.fft import dct, idct


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

        if checkpoint == 't36':
            self.encoder, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        if checkpoint == 't33':
            self.encoder, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        if checkpoint == 't30':
            self.encoder, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        if checkpoint == 't12':
            self.encoder, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        if checkpoint == 't6':
            self.encoder, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

        self.tokenizer = alphabet.get_batch_converter()
        self.encoder.eval()


    def to_device(self, device: str):
        """Moves model to device.

        Args:
            device (str): gpu/cpu
        """

        self.encoder.to(device)


@dataclass
class Fingerprint:
    """This class creates and stores the necessary information for a protein sequence to be
    embedded and quantized.

    Attributes:
        pid (str): Protein ID.
        seq (str): Protein sequence.
        embed (dict): Dictionary of embeddings from each layer.
        contacts (np.array): Contact map.
        domains (list): List of domain boundaries.
        quants (dict): Dictionary of quantizations from each domain.
    """
    pid: str = field(default_factory=str)
    seq: str = field(default_factory=str)
    embed: dict = field(default_factory=dict)
    contacts: np.array = field(default_factory=list)
    domains: list = field(default_factory=list)
    quants: dict = field(default_factory=dict)


    def __post_init__(self):
        """Initialize quant to empty array.
        """

        self.contacts = np.array([])


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
            results = model.encoder(batch_tokens, repr_layers=layers, return_contacts=True)

        # Store embedding from each layer
        embeds = {}
        for layer in layers:
            emb = results["representations"][layer].cpu().numpy()
            embeds[layer] = emb[0][1:-1]  # remove <cls> and <eos>
        self.embed = embeds
        self.contacts = results["contacts"].cpu().numpy()


    def writece(self, outfile: str, t: float):
        """write in CE for domain segmentation
        the top t * L contact pairs, L is the length (t is the alpha parameter in FUpred paper)

        Args:
            outfile (str): path to output file
            t (float): threshold for contact map (0.5 to 5)
        """

        # Sort contacts by confidence
        slen = len(self.seq)
        cta = self.contacts.reshape(slen, slen)
        data = []
        for i in range(slen - 5):
            for j in range(i + 5, slen):
                data.append([cta[i][j], i, j])
        ct_sorted = sorted(data, key=itemgetter(0), reverse=True)

        # Get top t * L contacts
        sout = ""
        tot = int(t * slen)
        for s in range(tot):
            i, j = ct_sorted[s][1], ct_sorted[s][2]
            if not sout:
                sout = f"CON   {i} {j} {cta[i][j]:.6f}"
            else:
                sout += f",{i} {j} {cta[i][j]:.6f}"

        # Write sequence info and top contacts to file
        with open(outfile, "w", encoding='utf8') as out_f:
            out_f.write(f"INF   {self.pid} {slen}\n")
            out_f.write(f"SEQ   {self.seq}\n")
            out_f.write(f"SS    {'C' * slen}\n")
            out_f.write(sout + "\n")


    def reccut(self, file: str):
        """Runs RecCut on a CE file to predict domains.

        Args:
            file (str): Path to CE file.
        """

        command = ['scripts/RecCut', '--input', file, '--name', f'{self.pid}']
        result = sp.run(command, stdout=sp.PIPE, text=True, check=True)
        self.domains = result.stdout.strip().split()[2].split(';')[:-1]


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

            # Quantize each domain
            for dom in self.domains:

                # Split embedding into domain
                try:
                    beg, end = dom.split('-')
                    dom_emb = embed[int(beg):int(end)+1, :]
                except ValueError:  # discontinuous domain
                    dom_emb = np.empty((0, embed.shape[1]))
                    ddom = dom.split(',')
                    for do in ddom:
                        beg, end = do.split('-')
                        dom_emb = np.append(dom_emb, embed[int(beg):int(end)+1, :], axis=0)

                # Quantize domain
                dct = self.idct_quant(dom_emb[1:len(dom_emb)-1], n_dim)  #pylint: disable=W0621
                ddct = self.idct_quant(dct.T, m_dim).T
                ddct = ddct.reshape(n_dim * m_dim)
                ddct = (ddct*127).astype('int8')
                self.quants.setdefault(beg, []).extend(ddct.tolist())

        # Set all lists to numpy arrays
        for key, value in self.quants.items():
            self.quants[key] = np.array(value)
