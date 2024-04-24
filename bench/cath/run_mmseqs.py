"""Queries CATH20 database with sequences from cath20_queries.txt using MMseqs2.

__author__ = "Ben Iovino"
__date__ = "4/23/24"
"""

import os
import subprocess as sp
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path


def mmseqs_results(path: str) -> dict[str, set]:
    """Returns dictionary of query PID's and a set of TP's up to first FP.

    Args:
        path (str): Path to results file

    Returns:
        dict[str, set]: key: query PID, value: set of TP sequences
    """

    with open(f'{path}/results_mmseqs.txt', 'r', encoding='utf8') as file:
        results: dict[str, set] = {}
        query, fp = '', False 
        for line in file:
            line = line.split()
            if line == []:  # Skip empty lines
                continue
            if line[0] != query:  # New query
                query = line[0]
                fp = False
            result = line[1]

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

    path = 'bench/cath/data'
    sp.run(['mmseqs', 'easy-search', f'{path}/cath20_queries.fa', f'{path}/cath20.fa',
            f'{path}/results_mmseqs.txt', f'{path}/tmp', '-s', '7.5', '--max-seqs', '1000'])


if __name__ == '__main__':
    main()
