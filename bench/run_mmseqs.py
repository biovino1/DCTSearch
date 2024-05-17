"""Queries protein sequence database with sequences from a query file using MMseqs2.

__author__ = "Ben Iovino"
__date__ = "4/23/24"
"""

import argparse
import os
import subprocess as sp
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path


def main():
    """
    Args:
        --bench (str): benchmark to query
            Can be 'cath', 'pfam', or 'scop'
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=str, required=True, help='benchmark to test')
    args = parser.parse_args()

    # Determine query and db files
    if args.bench == 'cath':
        path = 'bench/cath/data'
        query = 'cath20_queries.fa'
        db = 'cath20.fa'
    elif args.bench == 'pfam':
        path = 'bench/pfam/data'
        query = 'pfam20.fa'
        db = 'pfam20.fa'
    elif args.bench == 'scop':
        path = 'bench/scop/data'
        query = 'query.fa'
        db = 'target.fa'

    sp.run(['mmseqs', 'easy-search', f'{path}/{query}', f'{path}/{db}',
            f'{path}/results_mmseqs.txt', f'{path}/tmp', '-e', '10000', '-s', '7.5'])


if __name__ == '__main__':
    main()
