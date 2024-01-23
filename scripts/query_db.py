"""Queries a database of DCT fingerprints for most similar protein

__author__ = "Ben Iovino"
__date__ = "12/18/23"
"""

import argparse
from fingerprint import Model, Fingerprint
from util import load_seqs, load_fdb


def fprint_query(query: dict, device: str) -> dict:
    """Returns a dictionary of DCT fingerprints for a given protein sequence.

    Args:
        query (dict): Dictionary where key is protein ID and value is the sequence.
        device (str): gpu/cpu
    """

    model = Model('esm2', 't33')
    model.to_device(device)

    for pid, seq in query.items():
        fprint = Fingerprint(pid=pid, seq=seq)
        fprint.esm2_embed(model, device, layers=[15, 23])
        if not fprint.embed:
            continue
        fprint.reccut(2.6)
        fprint.quantize([3, 85, 5, 44])

    return fprint.quants


def search_db(fprint: dict, fdb: dict):
    """Searches a database of DCT fingerprints for most similar protein.

    Args:
        fprint (dict): Dictionary of DCT fingerprints for query sequence.
        fdb (dict): Dictionary of DCT fingerprints for database.
    """

    max_sim = 0
    for pid, quants in fdb.items():
        for quant in quants:
            for fp in fprint.values():
                sim = 1-abs(quant-fp).sum()/17000
                if sim > max_sim:
                    max_sim = sim
                    max_pid = pid

    print(max_sim, max_pid)


def main():
    """Main
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--query', default='example_q.fasta', help='Query sequence (fa format)')
    parser.add_argument('--dbfile', default='example.npz', help='Database of fingerprints')
    args = parser.parse_args()

    # Load query sequence and get fingerprints
    query = load_seqs(args.query)
    fprint = fprint_query(query, 'cpu')

    # Load database and search query sequence against db
    fdb = load_fdb(args.dbfile)
    search_db(fprint, fdb)


if __name__ == '__main__':
    main()
