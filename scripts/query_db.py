"""Queries a database of DCT fingerprints for most similar protein

__author__ = "Ben Iovino"
__date__ = "12/18/23"
"""

import argparse
import torch
from fingerprint import Model, Fingerprint
from util import load_seqs, load_fdb


def fprint_query(query: dict, device: str) -> dict:
    """Returns a dictionary of DCT fingerprints for a given collection of protein sequences.

    Args:
        query (dict): Dictionary where key is protein ID and value is the sequence.
        device (str): gpu/cpu

    Returns:
        dict: Nested dictionary where key is protein ID and value is a dictionary of DCT
        fingerprints for each predicted domain.
    """

    model = Model('esm2', 't33')
    model.to_device(device)

    fprints = {}
    for pid, seq in query.items():
        fprint = Fingerprint(pid=pid, seq=seq)
        fprint.esm2_embed(model, device, layers=[15, 23])
        if not fprint.embed:
            continue
        fprint.reccut(2.6)
        fprint.quantize([3, 85, 5, 44])
        fprints[pid] = fprint.quants

    return fprints


def search_db(fprints: dict, fdb: dict):
    """Searches a database of DCT fingerprints for the most similar protein to each query sequence.

    Args:
        fprint (dict): Dictionary of DCT fingerprints for query sequence(s).
        fdb (dict): Dictionary of DCT fingerprints for database.
    """

    for pid, quants in fprints.items():
        max_sim, doms = 0, []
        for db_pid, db_quants in fdb.items():
            for qdom, quant in quants.items():
                for dbdom, db_quant in db_quants.items():
                    sim = 1-abs(quant-db_quant).sum()/17000
                    if sim > max_sim:
                        max_sim = sim
                        max_pid = db_pid
                        doms = [qdom, dbdom]

        print(f'Query: {pid}')
        print(f'Top Hit: {max_pid}')
        print(f'Similarity Score: {max_sim}')
        print(f'Query Region: {doms[0]}')
        print(f'Top Hit Region: {doms[1]}\n')

def main():
    """Main
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True, help='Query sequence (fasta)')
    parser.add_argument('--dbfile', required=True, help='Database of fingerprints (npz)')
    parser.add_argument('--gpu', default=False, help='gpu (True) or cpu (False)')
    args = parser.parse_args()

    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Load query sequence and get fingerprints
    query = load_seqs(args.query)
    fprints = fprint_query(query, device)

    # Load database and search query sequence against db
    fdb = load_fdb(args.dbfile)
    search_db(fprints, fdb)


if __name__ == '__main__':
    main()
