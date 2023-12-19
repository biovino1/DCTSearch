"""Embeds protein sequences using protein language models.

__author__ = "Ben Iovino"
__date__ = "12/18/23"
"""

import argparse
import datetime
import logging
import os
import torch
import numpy as np
from Bio import SeqIO
from embed import Model, Transform

log_filename = 'data/logs/embed_seqs.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def load_seqs(filename: str) -> dict:
    """Returns a dictionary of protein sequences from a fasta file.

    Args:
        filename (str): Name of fasta file to parse.
    
    Returns:
        dict: dictionary where key is protein ID and value is the sequence
    """

    seqs = {}
    for record in SeqIO.parse(filename, 'fasta'):
        seqs[record.id] = str(record.seq)

    return seqs


def embed_seqs(seqs: list, efile: str):
    """Embeds a list of sequences and writes them to a file.

    :param seqs: list of sequences
    :param efile: path to embeddings file to save
    """

    model = Model('esm2')  # pLM encoder and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    model.to_device(device)

    # Embed each sequence and write to file
    quants = []
    for pid, seq in seqs.items():

        # Initialize object and embed
        logging.info('%s: Embedding %s', datetime.datetime.now(), pid)
        quant = Transform(pid, seq)
        quant.esm2_embed(model, device, layer=17)
        quant.quantize(8, 75)
        quants.append(quant)

    with open(efile, 'wb') as file:
        np.save(file, quants)


def main():
    """Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='data/scop_seqs.fa', help='fasta file to embed')
    parser.add_argument('-e', type=str, help='file to save embeddings to')
    args = parser.parse_args()

    # Load sequences from file and embed
    seqs = load_seqs(args.f)
    embed_seqs(seqs, 'data/scop_quants.np')


if __name__ == '__main__':
    main()
