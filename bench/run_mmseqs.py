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
    """MMseqs ran like in their benchmark.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=str, required=True, help='benchmark to test')
    args = parser.parse_args()

    # Determine query and db files
    if args.bench == 'cath':
        path = 'bench/cath/data'
        query = 'cath20_queries'
        db = 'cath20'
    elif args.bench == 'pfam':
        path = 'bench/pfam/data'
        query = 'pfam20'
        db = 'pfam20'
    elif args.bench == 'scop':
        path = 'bench/scop/data'
        query = 'query'
        db = 'target'

    # Create MMseqs2 databases
    sp.run(['mmseqs', 'createdb', f'{path}/{query}.fa', f'{path}/mmseqs/{query}DB'])
    sp.run(['mmseqs', 'createdb', f'{path}/{db}.fa', f'{path}/mmseqs/{db}DB'])
    sp.run(['mmseqs', 'createindex', f'{path}/mmseqs/{db}DB', '-k', '7', '--split', '1'])

    # Search and convert results
    os.makedirs(f'{path}/mmseqs', exist_ok=True)
    sp.run(['mmseqs', 'search', f'{path}/mmseqs/{query}DB', f'{path}/mmseqs/{db}DB',
            f'{path}/mmseqs/results_sense_aln', f'{path}/tmp', '--min-ungapped-score', '0',
            '-e', '10000', '-k', '7', '--k-score', '80', '--max-seqs', '4000', '-s', '7.5'])
    sp.run(['mmseqs', 'convertalis', f'{path}/mmseqs/{query}DB', f'{path}/mmseqs/{db}DB',
            f'{path}/mmseqs/results_sense_aln', f'{path}/results_mmseqs.txt'])


if __name__ == '__main__':
    main()
