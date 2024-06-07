"""Queries SCOP20 database with sequences from scop_seqs.fa using DCTSearch.

__author__ = "Ben Iovino"
__date__ = "6/04/24"
"""

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
from bench.run_dct import get_queries, search_db
from bench.plot_results import get_classes, read_results, eval_scores


def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=str, required=True, help='benchmark to test')
    parser.add_argument('--cid', type=str, default='fam', help='classification of sequences')
    parser.add_argument('--layers', type=int, default=25, help='number of layers in model')
    parser.add_argument('--cpu', type=int, default=1, help='number of cpus to use for search')
    parser.add_argument('--khits', type=int, default=300, help='number of hits to return')
    args = parser.parse_args()

    if args.bench == 'scop':
        path = 'bench/models/data/scop'
        query = 'seqs_test.fa'
        fams = get_classes(f'{path}/{query}', args.cid)

    logging.basicConfig(level=logging.INFO, filename=f'{path}/results_layers.txt',
                         filemode='w', format='%(message)s')
    
    # Get queries from SCOP20 database and search against database
    queries = get_queries(f'{path}/{query}')
    os.environ['OMP_NUM_THREADS'] = str(args.cpu)
    mean_auc1 = []
    for lay in range(0, args.layers):
        db = f'db/layer{lay}.db'
        open(f'{path}/results_layers.txt', 'w').close()
        search_db(f'{path}/{db}', queries, args.khits)
        results = read_results(f'{path}/results_layers.txt', 1, 5)
        auc = eval_scores(fams, results, f'Layer {lay}', args.cid)
        mean_auc1.append(np.mean(list(auc.values())))
    
    # Graph results
    _, ax = plt.subplots()
    ax.plot(range(0, args.layers), mean_auc1)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean AUC1')
    ax.set_title('Mean AUC1 by Layer')
    plt.show()


if __name__ == '__main__':
    main()
