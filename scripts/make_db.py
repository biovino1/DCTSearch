"""Makes a database of DCT fingerprints from a fasta file of protein sequences.

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
from util import load_seqs

log_filename = 'data/logs/make_db.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def get_fprints(seqs: dict, device: str, dbfile: str, layers: list, qdim: list):
    """Creates DCT fingerprints for a fasta file of protein sequences and saves them as a npz file
    with three arrays, one for protein IDs, one for domain boundaries, and one for fingerprints.

    Args:
        seqs (dict): Dictionary of protein sequences
        device (str): gpu/cpu
        dbfile (str): File to write fingerprints to.
        layers (list): List of layers to use for embedding.
        qdim (list): List of quantization dimensions.
    """

    model = Model('esm2', 't33')  # pLM encoder and tokenizer
    model.to_device(device)

    pids, idx, doms, quants = [], [], [], []
    idx_count = 0  # index of domains in npz file
    for pid, seq in seqs.items():

        # Skip if too big to embed
        if len(seq) > 1400:
            logging.info('%s: Skipping %s, sequence too long', datetime.datetime.now(), pid)
            continue

        # Initialize object and get embeddings for each layer + contact map
        logging.info('%s: Embedding %s', datetime.datetime.now(), pid)
        fprint = Fingerprint(pid=pid, seq=seq)
        fprint.esm2_embed(model, device, layers=layers)
        if not fprint.embed:  # Skip if embedding failed
            continue
        fprint.reccut(2.6)
        fprint.quantize(qdim)

        # Save protein ID, domain boundaries, and fingerprints
        pids.append(pid)
        idx.append(idx_count)
        idx_count += len(fprint.domains)
        for item in zip(fprint.domains, fprint.quants.values()):
            doms.append(item[0])
            quants.append(item[1])

    np.savez_compressed(dbfile, pids=pids, idx=idx, doms=doms, quants=quants)


def main():
    """Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--fafile', type=str, required=True, help='fasta file to embed')
    parser.add_argument('--dbfile', type=str, required=True, help='db file to write to')
    parser.add_argument('--gpu', type=bool, default=False, help='gpu (True) or cpu (False)')
    parser.add_argument('--layers', type=int, nargs='+', default=[13, 25], help='embedding layers')
    parser.add_argument('--quantdims', type=int, nargs='+', default=[3, 85, 5, 44],
                         help='quantization dimensions, each pair of dimensions quantizes a layer')
    args = parser.parse_args()

    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Load sequences from file and embed
    seqs = load_seqs(args.fafile)
    get_fprints(seqs, device, args.dbfile, args.layers, args.quantdims)


if __name__ == '__main__':
    main()
