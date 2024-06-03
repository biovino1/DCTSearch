"""Embeds and transforms sequences for the purposes of testing different pLM embeddings.

__author__ = "Ben Iovino"
__date__ = "05/29/24"
"""

import argparse
import os
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
from src.database import Database
from src.embedding import Model, Embedding
from src.fingerprint import Fingerprint
from utils.utils import read_fasta
from multiprocessing import Lock, Value


def init_dbs(layers: list[str], dbdir: str, fafile: str):
    """Initializes a database for each layer of the model to store that layer's fingerprints.
    Redundant in space, but allows for simpler db operations.

    Args:
        layers (list[str]): List of layer names.
        dbdir (str): Path to directory for databases.
        fafile (str): Path to fasta file.
    """

    if not os.path.exists(dbdir):
        os.makedirs(dbdir)
    for layer in layers:
        Database(f'{dbdir}/layer{layer}.db', fafile)


def embed_seqs(fafile: str, dbdir: str):
    """Embeds each sequence in dictionary and stores fingerprints in respective layer's db.

    Args:
        fafile (str): Path to fasta file.
        dbdir (str): Path to directory for databases.
    """

    model, lock, counter = Model(), Lock(), Value('i', 0)
    for pid, seq in read_fasta(fafile).items():
        emb = Embedding(pid, seq)
        outputs = emb.extract_pt5(seq, model, 'cpu')
        for i, out in enumerate(outputs.hidden_states):
            db = Database(f'{dbdir}/layer{i}.db')
            out = out.cpu().numpy()
            fp = Fingerprint(pid, seq, embed={0: out[0]}, domains=[f'1-{len(seq)}'])
            fp.quantize([5, 85])
            db.add_fprint(fp, lock, counter)


def main():
    """Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=str, required=True, help='scop or cath')
    parser.add_argument('--layers', type=int, default=25, help='Number of layers in model')
    args = parser.parse_args()

    path = 'bench/models/data'
    layers = [str(i) for i in range(args.layers)]
    if args.bench == 'scop':
        dbdir = f'{path}/scop/db'
        fafile = f'{path}/scop/seqs_test.fa'
    elif args.bench == 'cath':
        dbdir = f'{path}/cath/db'
        fafile = f'{path}/cath/seqs_test.fa'

    init_dbs(layers, dbdir, fafile)
    embed_seqs(fafile, dbdir)
    for layer in layers:
        db = Database(f'{dbdir}/layer{layer}.db')
        db.create_index()


if __name__ == '__main__':
    main()
