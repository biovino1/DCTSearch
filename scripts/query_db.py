"""Queries a database of DCT fingerprints for most similar protein to each query sequence.

__author__ = "Ben Iovino"
__date__ = "3/19/23"
"""

import argparse
from datetime import datetime
import faiss
import logging
import os
import multiprocessing as mp
import numpy as np
from database import Database
from make_db import embed_cpu, embed_gpu

log_filename = 'data/logs/query_db.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def search_db(query_db: str, fp_db: str):
    """Searches a database of DCT fingerprints for the most similar protein to each query sequence.

    Args:
        query_db (str): Name of query database
        fp_db (str): Name of fingerprint database
    """

    # Connect to databases
    query_db = Database(query_db)
    query_db.db_info()
    index = faiss.read_index(fp_db.replace('.db', '.index'))
    fp_db = Database(fp_db)
    fp_db.db_info()

    # Get each sequence from query db and compare to db
    select = """ SELECT pid FROM sequences """
    query_fps = query_db.cur.execute(select).fetchall()
    for query in query_fps:
        qfps = query_db.load_fprints(pid=query[0])
        que = np.array([fp[1] for fp in qfps], dtype=np.uint8)
        D, I = index.search(que, 100)
    
        # For each result, get vid from fingerprints db
        for i in range(100):
            ind = int(I[0][i])
            select = 'SELECT pid, domain FROM fingerprints WHERE vid = ?'
            db_match = fp_db.cur.execute(select, (ind,)).fetchone()
            logging.info('%s, %s, %s, %s', datetime.now(), query[0], db_match, D[0][i]) 

    query_db.close()
    fp_db.close()


def main():
    """Processes sequences same as make_db.py and queries --dbfile for most similar sequence for
    each sequence in the query database.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True, help='can be .fa or .db file')
    parser.add_argument('--dbfile', type=str, required=True, help='fingerprint database (.db)')
    parser.add_argument('--maxlen', type=int, default=1000, help='max sequence length to embed')
    parser.add_argument('--cpu', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--gpu', type=int, default=False, help='number of gpus to use')
    args = parser.parse_args()

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
    search_db(query_db, args.dbfile)
   

if __name__ == '__main__':
    main()
