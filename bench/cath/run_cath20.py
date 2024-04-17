"""Queries CATH20 database with sequences from cath20_queries.txt.

__author__ = "Ben Iovino"
__date__ = "4/16/24"
"""

import argparse
import faiss
import logging
import os
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
import numpy as np
from src.database import Database
from src.query_db import get_top_hits


def get_queries(path: str) -> dict[str, str]:
    """Returns a dictionary of query PID's and their homologous superfamily classifcation.

    Args:
        path (str): Path to cath20_queries.txt.
    
    Returns:
        dict[str, str]: key: PID, value: classification
    """

    queries = {}
    with open(f'{path}/cath20_queries.txt', 'r', encoding='utf8') as file:
        for line in file:
            line = line.split()
            queries[line[0]] = line[1]

    return queries


def search_cath20(path: str, queries: dict[str, str], khits: int):
    """Similar to search_db in query_db.py, but queries CATH20 db with sequences from
    cath20_queries.txt.

    Args:
        path (str): Path to database and index files
        queries (dict[str, str]): Dictionary of query PID's and their homologous superfamily
        classification
        khits (int): Number of hits to return
    """

    db = Database(f'{path}/cath20.db')
    index = faiss.read_index(f'{path}/cath20.db'.replace('.db', '.index'))
    for pid, fam in queries.items():
        qfps = db.load_fprints(pid=f'{pid}|{fam}')
        que_arrs = np.array([fp[1] for fp in qfps])
        que_ind = np.array([fp[0] for fp in qfps])
        dm, im = index.search(que_arrs, khits)
        get_top_hits(dm, im, khits, db, db, que_ind)
    
    db.close()


def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--khits', type=int, default=14433, help='number of hits to return')
    args = parser.parse_args()

    # Read queries sequences
    path = 'bench/cath/data'
    queries = get_queries(path)
    logging.basicConfig(level=logging.INFO, filename=f'{path}/cath20_results.txt',
                         filemode='w', format='%(message)s')
    
    # Get queries from CATH20 database
    os.environ['OMP_NUM_THREADS'] = str(args.cpu)
    search_cath20(path, queries, args.khits)
        

if __name__ == '__main__':
    main()
