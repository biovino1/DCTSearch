"""Queries a database of DCT fingerprints for most similar protein to each query sequence.

__author__ = "Ben Iovino"
__date__ = "3/19/23"
"""

import argparse
import os
from embedding import Model, Embedding
from fingerprint import Fingerprint
from database import Database
from make_db import embed_cpu, embed_gpu


def compare_fprints(qfp: list, dfp: list) -> float:
    """Compares two sets of fingerprints and returns the highest similarity score between
    all pairs.

    Args:
        qfp (list): List of query fingerprints
        dfp (list): List of database fingerprints

    Returns:
        float: Similarity score
    """

    max_sim = 0
    for q in qfp:
        for d in dfp:
            sim = 1-abs(q-d).sum()/17000
            if sim > max_sim:
                max_sim = sim
    return max_sim


def search_db(query_db: str, fp_db: str):
    """Searches a database of DCT fingerprints for the most similar protein to each query sequence.

    Args:
        query_db (str): Name of query database
        fp_db (str): Name of fingerprint database
    """

    # Connect to databases
    query_db = Database(query_db)
    query_db.db_info()
    fp_db = Database(fp_db)
    fp_db.db_info()

    # Load fingerprints
    query_fps = query_db.load_fprints()
    db_fps = fp_db.load_fprints()

    # Compare fingerprints
    for query, qfp in query_fps.items():
        max_sim, max_pid = 0, ''
        for db, dfp in db_fps.items():
            sim = compare_fprints(qfp, dfp)
            if sim > max_sim:
                max_sim = sim
                max_pid = db

        print(f'{query} is most similar to {max_pid} with a similarity score of {max_sim}')
    print()

    query_db.close()
    fp_db.close()


def main():
    """Processes sequences same as make_db.py and queries --dbfile for most similar sequence for
    each sequence in the query database.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--fafile', type=str, required=True, help='query file (.fa)')
    parser.add_argument('--dbfile', type=str, required=True, help='fingerprint database (.db)')
    parser.add_argument('--maxlen', type=int, default=1000, help='max sequence length to embed')
    parser.add_argument('--cpu', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--gpu', type=int, default=False, help='number of gpus to use')
    args = parser.parse_args()

    # Embed query sequences
    query_db = os.path.splitext(args.fafile)[0]
    db = Database(query_db, args.fafile)
    if args.gpu:
        embed_gpu(args, db)
    else:
        embed_cpu(args, db)

    # Query database for most similar sequence
    search_db(query_db, args.dbfile)
   

if __name__ == '__main__':
    main()
