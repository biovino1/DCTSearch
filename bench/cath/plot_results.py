"""Plots AUC1 values of all benchmarked methods for each query in the CATH dataset.

__author__ = "Ben Iovino"
__date__ = "4/23/24"
"""

import matplotlib.pyplot as plt
from bench.cath.run_dct import dct_results
from bench.cath.run_mmseqs import mmseqs_results


def eval_results(path: str, results: dict[str, set]):
    """Returns dict of AUC1 scores for each query. AUC1 is calculated as the number of TP's
    up to the 1st FP divided by the number of sequences in the family.

    Args:
        path (str): Path to results file
        results (dict[str, set]): Dictionary of query PID's and a set of TP's up to the 1st FP.

    Returns:
        dict[str, float]: key: query PID, value: AUC1 score
    """

    # Read number of sequences in each family
    with open(f'{path}/cath20_queries.fa', 'r', encoding='utf8') as file:
        fams = {}
        for line in file:
            if line.startswith('>'):
                fam = line.split('|')[1].strip()
                fams[fam] = fams.get(fam, 0) + 1
    
    # Calculate AUC1
    scores: dict[str, float] = {}
    for query, tps in results.items():
        fam = query.split('|')[1]
        scores[query] = scores.get(query, 0) + len(tps) / (fams[fam]-1)  # ignore self hit
    
    return scores


def graph_results(scores: list[dict[str, float]]):
    """Graphs AUC1 scores for each query.

    Args:
        scores (list[dict[str, float]]): List of dictionaries containing AUC1 scores.
    """

    methods = ['DCTSearch', 'MMseqs2']
    colors = ['blue', 'red']
    _, ax = plt.subplots()
    for i, sco in enumerate(scores):
        y = range(len(sco))
        x = sorted(list(sco.values()), reverse=True)
        ax.plot(x, y, label=methods[i], color=colors[i])
    ax.set_xlabel('AUC1')
    ax.set_ylabel('Query')
    ax.legend(title='Search Tool', labels=methods)
    ax.set_title('AUC1 Scores for CATH20 Queries')
    plt.show()


def main():
    """
    """

    path = 'bench/cath/data'
    dct_res = dct_results(path)
    mmseqs_res = mmseqs_results(path)
    dct_scores = eval_results(path, dct_res)
    mmseqs_scores = eval_results(path, mmseqs_res)
    graph_results([dct_scores, mmseqs_scores])


if __name__ == '__main__':
    main()
