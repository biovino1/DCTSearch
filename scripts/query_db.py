"""Queries a database of DCT fingerprints for most similar protein to each query sequence.

__author__ = "Ben Iovino"
__date__ = "3/19/23"
"""

import argparse
import faiss
import logging
import os
import multiprocessing as mp
import numpy as np
from database import Database
from make_db import embed_cpu, embed_gpu


def get_top_hits(dm: np.ndarray, im: np.ndarray, top: int, fp_db: Database, query: str) -> list:
    """Returns a list of protein ID, domain, and distance for the top hits. Searches distance
    matrix for lowest values, and with corresponding values in index matrix, queries database
    for corresponding protein ID and domain.

    Args:
        D (np.ndarray): Distance matrix
        I (np.ndarray): Index matrix
        top (int): Number of top hits to return
        fp_db (Database): Database object for fingerprint database
        query (str): Query protein ID

    Returns:
        list: List of protein ID's and distance for top hits
    """

    # Store all hits and sort by distance
    top_hits = {}
    for i, khits in enumerate(dm):
        for j, dist in enumerate(khits):
            top_hits[i, j] = dist
    top_hits = dict(sorted(top_hits.items(), key=lambda x: x[1]))

    # Get protein ID and domain for each hit
    for i, index in enumerate(list(top_hits.keys())[:top]):
        vid = int(im[index[0], index[1]])
        if vid == -1:  # No more hits
            break
        select = """ SELECT pid, domain FROM fingerprints WHERE vid = ? """
        pid, domain = fp_db.cur.execute(select, (vid+1,)).fetchone()  # vid is 0-indexed
        logging.info('Query: %s, Result %s: %s-%s, Distance: %s',
                      f'{query}-{index[0]}', i, pid, domain, top_hits[index])
    print()


def search_db(args: argparse.Namespace, query_db: str, fp_db: str):
    """Searches a database of DCT fingerprints for the most similar protein to each query sequence.

    Args:
        args (argparse.Namespace): Command line arguments
        query_db (str): Name of query database
        fp_db (str): Name of fingerprint database
    """

    # Connect to databases
    query_db = Database(query_db)
    query_db.db_info()
    index = faiss.read_index(fp_db.replace('.db', '.index'))
    fp_db = Database(fp_db)

    # Get each sequence from query db and compare to db
    select = """ SELECT pid FROM sequences """
    query_fps = query_db.cur.execute(select).fetchall()
    print('Querying database...\n')
    for query in query_fps:
        qfps = query_db.load_fprints(pid=query[0])
        que = np.array([fp[1] for fp in qfps])
        dm, im = index.search(que, args.khits)  # distance, index matrices
        get_top_hits(dm, im, args.khits, fp_db, query[0])

    query_db.close()
    fp_db.close()


def main():
    """Processes sequences same as make_db.py and queries --dbfile for most similar sequence for
    each sequence in the query database.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True, help='can be .fa or .db file')
    parser.add_argument('--db', type=str, required=True, help='fingerprint database (.db)')
    parser.add_argument('--out', type=str, default=False, help='output file')
    parser.add_argument('--maxlen', type=int, default=1000, help='max sequence length to embed')
    parser.add_argument('--khits', type=int, default=100, help='number of hits to return')
    parser.add_argument('--cpu', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--gpu', type=int, default=False, help='number of gpus to use')
    args = parser.parse_args()

    # Logging for either stdout or file
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if args.out:
        logging.basicConfig(filename=args.out, filemode='w')

    # Embed query sequences
    query_db = os.path.splitext(args.query)[0]
    db = Database(query_db, args.query)
    vid = db.get_last_vid()
    lock, counter = mp.Lock(), mp.Value('i', vid)
    if args.gpu:
        embed_gpu(args, db, lock, counter)
    else:
        embed_cpu(args, db, lock, counter)

    # Query database for most similar sequence
    os.environ['OMP_NUM_THREADS'] = str(args.cpu)
    search_db(args, query_db, args.db)
   

if __name__ == '__main__':
    main()
