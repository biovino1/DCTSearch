"""Queries CATH20 database with sequences from cath20_queries.txt using DCTSearch.

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
            if line.startswith('>'):
                line = line.split()
                queries[line[0][1:]] = line[1]  # key: PID, value: classification

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


def dct_results(path: str) -> dict[str, set]:
    """Returns dictionary of query PID's and a set of TP's up to first FP.

    Args:
        path (str): Path to results file

    Returns:
        dict[str, set]: key: query PID, value: set of TP sequences
    """

    with open(f'{path}/results_dct.txt', 'r', encoding='utf8') as file:
        results: dict[str, set] = {}
        query, fp = '', False 
        for line in file:
            line = line.split()
            if line == []:  # Skip empty lines
                continue
            if line[1] != query:  # New query
                query = line[1]
                fp = False
            result = line[5]

            # Ignore if already reached a false positive or self hit
            if (fp or query == result):
                continue
            
            # Check homologous superfamily classification
            if query.split('|')[1] == result.split('|')[1]:  # True positive
                results[query] = results.get(query, set()) | set([result])
            else:  # False positive
                fp = True

    return results


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
    logging.basicConfig(level=logging.INFO, filename=f'{path}/results_dct.txt',
                         filemode='w', format='%(message)s')
    
    # Get queries from CATH20 database and get top1/AUC1
    os.environ['OMP_NUM_THREADS'] = str(args.cpu)
    search_cath20(path, queries, args.khits)
        

if __name__ == '__main__':
    main()
