"""Testing performance of each layer in every ESM-2 model checkpoint.

__author__ = "Ben Iovino"
__date__ = "1/9/24"
"""

import os
from embed import Model
from sklearn import metrics


def calc_auc():
    """Calculates AUC for pair evaluation results.
    """

    # Load eval_pairs.log
    labels, scores = [], []
    with open('data/logs/eval_pairs.log', 'r', encoding='utf8') as file:
        for line in file:
            line = line.split()
            labels.append(int(line[4]))
            scores.append(float(line[6]))

    # Calculate AUC
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)

    return auc


def main():
    """Main
    """

    checkpoints = ['t6', 't12', 't30', 't33']
    for ch in checkpoints:

        # Load checkpoint and get number of layers
        model = Model('esm2', ch)
        layers = model.encoder.num_layers

        # For each layer, embed sequences and evaluate performance
        for lay in range(1, layers+1):
            os.system(f'python scripts/embed_seqs.py -c {ch} -l {lay} -q 5 50')
            os.system('python scripts/eval_pairs.py -m dct')
            auc = calc_auc()

            # Log AUC
            with open('data/logs/testing.log', 'a', encoding='utf8') as file:
                file.write(f'{ch}\t{lay}\t{auc}\n')


if __name__ == '__main__':
    main()
