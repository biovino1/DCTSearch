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
from fingerprint import Model, Fingerprint

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


def get_fprints(seqs, dbfile, layers, qdim, ch):
    """Creates DCT fingerprints for a fasta file of protein sequences and saves them as a npz file
    with three arrays, one for protein IDs, one for domain boundaries, and one for fingerprints.

    Args:
        seqs (dict): Dictionary of protein sequences
        dbfile (str): File to write fingerprints to.
        layers (list): List of layers to use for embedding.
        qdim (list): List of quantization dimensions.
    """

    model = Model('esm2', ch)  # pLM encoder and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    model.to_device(device)

    pids, idx, doms, quants = [], [], [], []
    idx_count = 0  # index of domains in npz file
    for pid, seq in seqs.items():

        # Initialize object and get embeddings for each layer + contact map
        logging.info('%s: Embedding %s', datetime.datetime.now(), pid)
        fprint = Fingerprint(pid=pid, seq=seq)
        fprint.esm2_embed(model, 'cpu', layers=layers)
        fprint.writece('tmp.ce', 2.6)
        fprint.reccut('tmp.ce')
        fprint.quantize(qdim)

        # Save protein ID, domain boundaries, and fingerprints
        pids.append(pid)
        idx.append(idx_count)
        idx_count += len(fprint.domains)
        for item in zip(fprint.domains, fprint.quants.values()):
            doms.append(item[0])
            quants.append(item[1])

    # Save as npz
    np.savez_compressed(dbfile, pids=pids, idx=idx, doms=doms, quants=quants)


def main():
    """Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='t33', help='esm checkpoint')
    parser.add_argument('-d', type=str, default='test', help='db file to write to')
    parser.add_argument('-f', type=str, default='test.fa', help='fasta file to embed')
    parser.add_argument('-l', type=int, nargs='+', default=[13, 25], help='embedding layers')
    parser.add_argument('-q', type=int, nargs='+', default=[3, 85, 5, 44],
                         help='quantization dimensions, each pair of dimensions quantizes a layer')
    args = parser.parse_args()

    # Load sequences from file and embed
    seqs = load_seqs(args.f)
    get_fprints(seqs, args.d, args.l, args.q, args.c)


if __name__ == '__main__':
    main()
