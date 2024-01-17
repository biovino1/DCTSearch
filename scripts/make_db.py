"""Embeds protein sequences using protein language models.

__author__ = "Ben Iovino"
__date__ = "12/18/23"
"""

import argparse
import datetime
import logging
import os
import numpy as np
import torch
from embed import Model, Transform

log_filename = 'data/logs/make_db.log'  #pylint: disable=C0103
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
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            if line.startswith('>'):
                pid = line[1:].strip().split()[0]  #ex. '>16vpA00    110129010'
                seqs[pid] = ''
            else:
                seqs[pid] += line.strip()

    return seqs


def embed_seqs(seqs: dict, efile: str, layers: list, qdim: list, ch):
    """Creates DCT fingerprints for a set of protein sequences and saves them as a npz file with
    two arrays, one for protein IDs and one for fingerprints.

    Args:
        seqs (dict): Dictionary of protein sequences
        efile (str): File to write embeddings to.
        layers (list): List of layers to use for embedding.
        qdim (list): List of quantization dimensions.
    """

    model = Model('esm2', ch)  # pLM encoder and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    model.to_device(device)

    # Embed and quantize each sequence
    pids, quants = [], []
    for pid, seq in seqs.items():

        # Initialize object and embed
        logging.info('%s: Embedding %s', datetime.datetime.now(), pid)
        trans = Transform(pid=pid, seq=seq)
        trans.esm2_embed(model, device, layers=layers)
        trans.quantize(qdim)

        # Add protein ID and it's fingerprint to lists for later storage
        pids.append(pid)
        quants.append(trans.quant)

    # Make lists into numpy arrays and save as npz
    pids = np.array(pids)
    quants = np.array(quants)
    np.savez_compressed(efile, pids=pids, quants=quants)


def main():
    """Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='t6', help='esm checkpoint')
    parser.add_argument('-d', type=str, default='data/scop_quants', help='db file to write to')
    parser.add_argument('-f', type=str, default='data/scop_seqs.fa', help='fasta file to embed')
    parser.add_argument('-l', type=int, nargs='+', default=[5], help='embedding layers')
    parser.add_argument('-q', type=int, nargs='+', default=[5, 50],
                         help='quantization dimensions, each pair of dimensions quantizes a layer')
    args = parser.parse_args()

    # Load sequences from file and embed
    seqs = load_seqs(args.f)
    embed_seqs(seqs, args.d, args.l, args.q, args.c)


if __name__ == '__main__':
    main()
