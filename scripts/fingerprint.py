"""Defines the Fingerprint class, which is used to take a protein sequence, it's protein language
model embedding, and contact map to predict domains and perform iDCT vector quantization on each
domain to generate a set of fingerprints.

__author__ = "Ben Iovino"
__date__ = "12/18/23"
"""

from dataclasses import dataclass, field
from operator import itemgetter
import os
import subprocess as sp
import numpy as np
from scipy.fft import dct, idct


@dataclass
class Fingerprint:
    """This class creates and stores the necessary information for a protein sequence to be
    quantized into one or more fingerprints for each predicted domain of the sequence.

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
        """Initialize contacts to array.
        """

        self.contacts = np.array(self.contacts)


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
        if tot > len(ct_sorted):  # If sequence is too small for t * L contacts, use all contacts
            tot = len(ct_sorted)
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


    def reccut(self, threshold: float):
        """Runs RecCut on a contact map to predict domains and appends the domain boundaries to the
        domains list.

        Args:
            threshold (float): Threshold for contact maps (0.5 to 5).
        """

        # Get top contacts then predict domains
        filename = f'{self.pid}.ce'
        self.writece(filename, threshold)
        command = ['scripts/RecCut', '--input', filename, '--name', f'{self.pid}']
        result = sp.run(command, stdout=sp.PIPE, text=True, check=True)
        os.remove(filename)

        # Append domain boundaries to list
        domains = result.stdout.strip().split()[2].split(';')[:-1]  # remove last empty string
        if len(domains) > 1:  # If there are multiple domains, add length of full sequence
            self.domains.append('1-' + str(len(self.seq)))
        for dom in domains:
            self.domains.append(dom)


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

                # Split embedding (0-indexed) into domain (1-indexed)
                try:
                    beg, end = dom.split('-')
                    dom_emb = embed[int(beg)-1:int(end), :]
                except (TypeError, IndexError, ValueError):  # discontinuous domain
                    dom_emb = np.empty((0, embed.shape[1]))
                    dom = dom.split(',')
                    for do in dom:
                        beg, end = do.split('-')
                        dom_emb = np.append(dom_emb, embed[int(beg)-1:int(end), :], axis=0)
                    dom = ','.join(dom)  # convert back to string for dict key

                # Quantize domain
                dct = self.idct_quant(dom_emb[1:len(dom_emb)-1], n_dim)  #pylint: disable=W0621
                ddct = self.idct_quant(dct.T, m_dim).T
                ddct = ddct.reshape(n_dim * m_dim)
                ddct = (ddct*127).astype('int8')
                self.quants.setdefault(dom, []).extend(ddct.tolist())

        # Set all lists to numpy arrays
        for key, value in self.quants.items():
            self.quants[key] = np.array(value)


    def save(self, filename: str, save_embed: bool = False):
        """Saves Fingerprint object to npz file.

        Args:
            filename (str): File to save fingerprints to.
            save_embed (bool): Save embeddings to file (default: False).
        """

        if save_embed:
            np.savez_compressed(filename, pid=self.pid, seq=self.seq, embed=self.embed,
                                contacts=self.contacts, domains=self.domains, quants=self.quants)
        else:
            np.savez_compressed(filename, pid=self.pid, seq=self.seq,
                                domains=self.domains, quants=self.quants)


    def load(self, filename: str):
        """Loads Fingerprint object from npz file.

        Args:
            filename (str): File to load fingerprints from.
        """

        with np.load(filename, allow_pickle=True) as data:
            self.pid = data['pid']
            self.seq = data['seq']
            self.domains = data['domains']
            self.quants = data['quants']
            if 'embed' in data.files:
                self.embed = data['embed']
            if 'contacts' in data.files:
                self.contacts = data['contacts']
