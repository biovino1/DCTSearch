"""Plots AUC1 values of all benchmarked methods for each query in the CATH dataset.

__author__ = "Ben Iovino"
__date__ = "4/23/24"
"""

import argparse
import matplotlib.pyplot as plt


def get_classes(file: str, claisf: str) -> dict[str, list]:
    """Returns a dictionary of family names and the sequences that belong to them.

    Args:
        file (str): Path to queries file
        clasif (str): Desired classification ID of seqs (SCOP ONLY)
            fam - Family ID
            supfam - Superfamily ID
            fold - Fold ID

    Returns:
        dict[str, list]: key: classification ID, value: list of sequences
    """

    with open(file, 'r', encoding='utf8') as file:
        cids: dict[str, list] = {}  # key: classification ID, value: list of sequences
        for line in file:
            if line.startswith('>'):
                line = line.split('|')
                seq = line[0]
                for cid in line[1].strip().split(';'):  # For multi-domain seqs (i.e. scop benchmark)
                    if cid not in cids:
                        cids[cid] = set()
                    cids[cid].add(seq)

    return cids


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
            query_dom = query.split('|')[1].split(';')
            result_dom = result.split('|')[1].split(';')
            if not any([dom in query_dom for dom in result_dom]):
                curr_query = query
                continue

            # Add hit to results
            results[query].add(result)

    return results


def eval_scores(cids: dict[str, list], results: dict[str, set], method: str):
    """Returns dict of AUC1 scores for each query. AUC1 is calculated as the number of TP's
    up to the 1st FP divided by the number of sequences in the family.

    Args:
        cids (dict[str, list]): Dictionary of classification ID's and a list of sequences.
        results (dict[str, set]): Dictionary of query PID's and a set of TP's up to the 1st FP.
        method (str): Method being evaluated.

    Returns:
        dict[str, float]: key: query PID, value: AUC1 score
    """

    # Calculate scores
    auc_scores: dict[str, float] = {}
    cid_scores: dict[str, list[float, float]] = {}  # keep track of tp/total for each class
    top1, total = 0, 0
    for query, tps in results.items():

        # Get family and update scores
        qcids = query.split('|')[1]  # i.e. 16vpA00|3.30.930.10
        cid_scores[qcids] = cid_scores.get(qcids, [0, 0])

        # top1 - Add 1 if top hit belongs to same family
        # fam_scores - Add 1 to total and top1 if top hit belongs to same family
        if len(tps) > 0:  # If there are hits, that means top hit is TP
            cid_scores[qcids][0] += 1
            top1 += 1
        cid_scores[qcids][1] += 1
        total += 1

        # Calculate AUC1 score
        try:
            possible_hits = 0
            for cid in qcids.split(';'):
                possible_hits += len(cids[cid])
            score = len(tps) / (possible_hits-1)  # ignore self hit
        except ZeroDivisionError:
            score = 0
        auc_scores[query] = auc_scores.get(query, 0) + score  # ignore self hit

    # For each family, find number of true positives over total family size
    fam_total = 0
    for fam, (tp, tot) in cid_scores.items():
        fam_total += (1/(tot)) * tp

    # Print results
    print(f'{method}:')
    print(f'QNormTop1: {(fam_total):.2f}/{len(cid_scores)}, {fam_total/len(cid_scores)*100:.2f}%')
    print(f'QRawTop1: {top1}/{total}, {(top1/total * 100):.2f}%\n')
    
    return auc_scores


def graph_results(scores: list[dict[str, float]], bench: str, methods: list[str]):
    """Graphs AUC1 scores for each query.

    Args:
        scores (list[dict[str, float]]): List of dictionaries containing AUC1 scores.
        bench (str): Benchmark being evaluated
        methods (list[str]): List of methods being evaluated
    """

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
    parser.add_argument('--cid', type=str, default='fam', help='classification of sequences')
    args = parser.parse_args()

    # Determine query and db files
    if args.bench == 'cath':
        path = 'bench/cath/data'
        cids = get_classes(f'{path}/cath20_queries.fa', args.cid)
    elif args.bench == 'pfam':
        path = 'bench/pfam/data'
        cids = get_classes(f'{path}/pfam20.fa', args.cid)
    elif args.bench == 'scop':
        path = 'bench/scop/data'
        cids = get_classes(f'{path}/target.fa', args.cid)

    # Read results after running DCTSearch and MMseqs2
    dct_res = read_results(f'{path}/results_dct.txt', 1, 5)
    mean_res = read_results(f'{path}/results_mean.txt', 1, 5)
    mmseqs_res = read_results(f'{path}/results_mmseqs.txt', 0, 1)

    # Plot AUC1 scores for each query
    scores = []
    scores.append(eval_scores(cids, dct_res, 'DCTSearch'))
    scores.append(eval_scores(cids, mean_res, 'ProtT5-Mean'))
    scores.append(eval_scores(cids, mmseqs_res, 'MMseqs2'))
    methods = ['DCTSearch', 'ProtT5-Mean', 'MMseqs2']
    graph_results(scores, args.bench, methods)


if __name__ == '__main__':
    main()
