"""Embeds protein sequences using protein language models.

__author__ = "Ben Iovino"
__date__ = "12/18/23"
"""

import argparse
import datetime
import logging
import os
import pickle
import torch
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


def embed_seqs(seqs: list, efile: str, layers: list, qdim: list, ch):
    """Embeds a list of sequences and writes them to a file.

    Args:
        seqs (list): List of sequences to embed.
        efile (str): File to write embeddings to.
        layers (list): List of layers to use for embedding.
        qdim (list): List of quantization dimensions.
    """

    model = Model('esm2', ch)  # pLM encoder and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    model.to_device(device)

    # Embed each sequence and write to file
    quants = {}
    for pid, seq in seqs.items():

        # Initialize object and embed
        logging.info('%s: Embedding %s', datetime.datetime.now(), pid)
        quant = Transform(pid=pid, seq=seq)
        quant.esm2_embed(model, device, layers=layers)
        quant.quantize(qdim)
        quants[quant.pid] = quant.quant

    with open(efile, 'wb') as file:
        pickle.dump(quants, file)


def main():
    """Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='t30', help='model checkpoint')
    parser.add_argument('-e', type=str, help='file to save embeddings to')
    parser.add_argument('-f', type=str, default='data/scop_seqs.fa', help='fasta file to embed')
    parser.add_argument('-l', type=int, nargs='+', default=[17, 25], help='embedding layers')
    parser.add_argument('-q', type=int, nargs='+', default=[3, 85, 5, 44],
                         help='quantization dimensions, each pair of dimensions quantizes a layer')
    args = parser.parse_args()

    # Load sequences from file and embed
    seqs = load_seqs(args.f)
    embed_seqs(seqs, 'data/scop_quants.pkl', args.l, args.q, args.c)


if __name__ == '__main__':
    main()
