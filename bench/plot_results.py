"""Plots AUC1 values of all benchmarked methods for each query in the CATH dataset.

__author__ = "Ben Iovino"
__date__ = "4/23/24"
"""

import argparse
import matplotlib.pyplot as plt


def get_fams(file: str) -> dict[str, list]:
    """Returns a dictionary of family names and the sequences that belong to them.

    Args:
        file (str): Path to queries file

    Returns:
        dict[str, list]: key: family name, value: list of sequences
    """

    with open(file, 'r', encoding='utf8') as file:
        fams: dict[str, list] = {}
        for line in file:
            if line.startswith('>'):
                line = line.split('|')
                dom, fam = line[0], line[1].strip()
                fams[fam] = fams.get(fam, []) + [dom]

    return fams


def read_results(path: str, query_ind: int, result_ind: int) -> dict[str, set]:
    """Returns a dictionary of query PID's and their top hits until the first FP. Only one hit per
    unique sequence is counted, hence the use of a set. This allows for the calculation of both
    AUC1 and top1 scores.

    Args:
        path (str): Path to results file
        query (int): Column index of query PID
        result (int): Column index of result PID

    Returns:
        dict[str, set]: key: query PID, value: set of top hits until the first FP
    """

    with open(f'{path}', 'r', encoding='utf8') as file:
        results, curr_query = {}, ''
        for line in file:
            if line == '\n':
                continue
            line = line.split()

            # If query had FP, continue until new query
            query = line[query_ind]
            result = line[result_ind]
            if query == curr_query:
                continue
            results[query] = results.get(query, set())

            # Ignore self-hits
            if query == result:
                continue

            # Stop counting hits for current query if domains are different
            query_dom = query.split('|')[1]
            result_dom = result.split('|')[1]
            if query_dom != result_dom:
                curr_query = query
                continue

            # Add hit to results
            results[query].add(result)

    return results


def eval_scores(fams: dict[str, list], results: dict[str, set], method: str):
    """Returns dict of AUC1 scores for each query. AUC1 is calculated as the number of TP's
    up to the 1st FP divided by the number of sequences in the family.

    Args:
        path (str): Path to results file
        results (dict[str, set]): Dictionary of query PID's and a set of TP's up to the 1st FP.
        method (str): Method being evaluated

    Returns:
        dict[str, float]: key: query PID, value: AUC1 score
    """
    
    # Calculate scores
    auc_scores: dict[str, float] = {}
    fam_scores: dict[str, list[float, float]] = {}  # keep track of tp/total for each family
    top1, total = 0, 0
    for query, tps in results.items():

        # Get family and update scores
        fam = query.split('|')[1]  # i.e. 16vpA00|3.30.930.10
        fam_scores[fam] = fam_scores.get(fam, [0, 0])

        # top1 - Add 1 if top hit belongs to same family
        # fam_scores - Add 1 to total and top1 if top hit belongs to same family
        if len(tps) > 0:  # If there are hits, that means top hit is TP
            fam_scores[fam][0] += 1
            top1 += 1
        fam_scores[fam][1] += 1
        total += 1

        # Calculate AUC1 score
        try:
            score = len(tps) / (len(fams[fam])-1)  # ignore self hit
        except ZeroDivisionError:
            score = 0
        auc_scores[query] = auc_scores.get(query, 0) + score  # ignore self hit

    # For each family, find number of true positives over total family size
    fam_total = 0
    for fam, (tp, tot) in fam_scores.items():
        fam_total += (1/(tot)) * tp

    # Print results
    print(f'{method}:')
    print(f'QNormTop1: {(fam_total):.2f}/{len(fam_scores)}, {fam_total/len(fam_scores)*100:.2f}%')
    print(f'QRawTop1: {top1}/{total}, {(top1/total * 100):.2f}%\n')
    
    return auc_scores


def graph_results(scores: list[dict[str, float]], bench: str):
    """Graphs AUC1 scores for each query.

    Args:
        scores (list[dict[str, float]]): List of dictionaries containing AUC1 scores.
        bench (str): Benchmark being evaluated
    """

    methods = ['DCTSearch', 'ProtT5-Mean', 'MMseqs2']
    averages = [sum(sco.values()) / len(sco) for sco in scores]
    labels = [f'{m} (mean: {a:.2f})' for m, a in zip(methods, averages)]
    colors = ['blue', 'orange', 'red']
    _, ax = plt.subplots()
    for i, sco in enumerate(scores):
        y = range(len(sco))
        x = sorted(list(sco.values()), reverse=True)
        ax.plot(x, y, label=methods[i], color=colors[i])
    ax.set_xlabel('AUC1')
    ax.set_ylabel('Query')
    ax.legend(title='Search Tool', labels=labels)
    ax.set_title(f'AUC1 Scores for {bench.upper()}')
    plt.show()


def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=str, required=True, help='benchmark to query')
    args = parser.parse_args()

    # Determine query and db files
    if args.bench == 'cath':
        path = 'bench/cath/data'
        fams = get_fams(f'{path}/cath20_queries.fa')
    elif args.bench == 'pfam':
        path = 'bench/pfam/data'
        fams = get_fams(f'{path}/pfam20.fa')
    elif args.bench == 'scop':
        path = 'bench/scop/data'
        fams = get_fams(f'{path}/query.fa')

    # Read results after running DCTSearch and MMseqs2
    dct_res = read_results(f'{path}/results_dct.txt', 1, 5)
    mean_res = read_results(f'{path}/results_mean.txt', 1, 5)
    mmseqs_res = read_results(f'{path}/results_mmseqs.txt', 0, 1)

    # Plot AUC1 scores for each query
    dct_scores = eval_scores(fams, dct_res, 'DCTSearch')
    mean_scores = eval_scores(fams, mean_res, 'ProtT5-Mean')
    mmseqs_scores = eval_scores(fams, mmseqs_res, 'MMseqs2')
    graph_results([dct_scores, mean_scores, mmseqs_scores], args.bench)


if __name__ == '__main__':
    main()
