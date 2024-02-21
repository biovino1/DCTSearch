"""Makes a database of DCT fingerprints from a fasta file of protein sequences.

__author__ = "Ben Iovino"
__date__ = "12/18/23"
"""

import argparse
import datetime
import logging
import os
import torch
from embedding import Model, Embedding
from fingerprint import Fingerprint
from database import Database
from util import load_seqs

log_filename = 'data/logs/make_db.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def get_fprints(seqs: dict, device: str, args: argparse.Namespace):
    """Creates DCT fingerprints for a fasta file of protein sequences, adds them to a database, and
    saves the database to a file.

    Args:
        seqs (dict): Dictionary of protein sequences
        device (str): gpu/cpu
        args (argparse.Namespace): Command line arguments
    """

    model = Model('esm2', 't30')  # pLM encoder and tokenizer
    model.to_device(device)

    db = Database()
    for pid, seq in seqs.items():

        # Initialize object and get embeddings for each layer + contact map
        logging.info('%s: Fingerprinting %s %s', datetime.datetime.now(), pid, len(seq))
        emb = Embedding(pid=pid, seq=seq)
        emb.embed_seq(model, device, args.layers, args.maxlen)
        fprint = Fingerprint(pid=pid, seq=seq, embed=emb.embed, contacts=emb.contacts)
        fprint.reccut(2.6)
        fprint.quantize(args.quantdims)
        db.add_fprint(fprint)

    db.save_db(args.dbfile)


def main():
    """Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--fafile', type=str, required=True, help='fasta file to embed')
    parser.add_argument('--dbfile', type=str, required=True, help='db file to write to')
    parser.add_argument('--maxlen', type=int, default=1000, help='max sequence length to embed')
    parser.add_argument('--gpu', type=bool, default=False, help='gpu (True) or cpu (False)')
    parser.add_argument('--layers', type=int, nargs='+', default=[15, 21], help='embedding layers')
    parser.add_argument('--quantdims', type=int, nargs='+', default=[3, 80, 3, 80],
                         help='quantization dimensions, each pair of dimensions quantizes a layer')
    args = parser.parse_args()

    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Load sequences from file and embed
    seqs = load_seqs(args.fafile)
    get_fprints(seqs, device, args)


if __name__ == '__main__':
    main()
