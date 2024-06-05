"""Queries SCOP20 database with sequences from scop_seqs.fa using DCTSearch.

__author__ = "Ben Iovino"
__date__ = "6/04/24"
"""

import argparse
import logging
import os
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
from bench.run_dct import get_queries, search_db
from bench.plot_results import get_fams, read_results, eval_scores

logging.getLogger()


def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=str, required=True, help='benchmark to test')
    parser.add_argument('--layers', type=int, default=25, help='number of layers in model')
    parser.add_argument('--cpu', type=int, default=1, help='number of cpus to use for search')
    parser.add_argument('--khits', type=int, default=300, help='number of hits to return')
    args = parser.parse_args()

    if args.bench == 'scop':
        path = 'bench/models/data/scop'
        query = 'seqs_test.fa'
        fams = get_fams(f'{path}/{query}')

    logging.basicConfig(level=logging.INFO, filename=f'{path}/results_layers.txt',
                         filemode='w', format='%(message)s')
    
    # Get queries from SCOP20 database and search against database
    queries = get_queries(f'{path}/{query}')
    os.environ['OMP_NUM_THREADS'] = str(args.cpu)
    for lay in range(0, args.layers):
        db = f'db/layer{lay}.db'
        open(f'{path}/results_layers.txt', 'w').close()
        search_db(f'{path}/{db}', queries, args.khits)
        results = read_results(f'{path}/results_layers.txt', 1, 5)
        eval_scores(fams, results, f'Layer {lay}')


if __name__ == '__main__':
    main()
