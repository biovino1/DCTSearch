"""Queries CATH20 database with sequences from cath20_queries.txt using MMseqs2.

__author__ = "Ben Iovino"
__date__ = "4/23/24"
"""

import os
import subprocess as sp
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path


def main():
    """
    """

    path = 'bench/cath/data'
    sp.run(['mmseqs', 'easy-search', f'{path}/cath20_queries.fa', f'{path}/cath20.fa',
            f'{path}/results_mmseqs.txt', f'{path}/tmp', '-e', '10000', '-s', '7.5'])


if __name__ == '__main__':
    main()
