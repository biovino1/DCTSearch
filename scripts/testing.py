"""Testing performance of each layer in every ESM-2 model checkpoint.

__author__ = "Ben Iovino"
__date__ = "1/9/24"
"""

import os
from datetime import datetime
import numpy as np
from embed import Model
from sklearn import metrics


def weighted_conf_mat(labels: list, weights: list, scores: list, threshold: float) -> tuple:
    """Returns confusion matrix for a given list of scores and labels with weighted samples.
    
    :param labels: list of labels
    :param weights: list of weights for each label
    :param scores: list of scores
    :param threshold: threshold for classification
    :return tuple: confusion matrix (tp, fp, tn, fn)
    """

    tn, fp, fn, tp = 0, 0, 0, 0
    for i, label in enumerate(labels):
        if scores[i] > threshold and label == 0:
            tn += 1*weights[i]
        if scores[i] > threshold and label == 1:
            fn += 1*weights[i]
        if scores[i] < threshold and label == 0:
            fp += 1*weights[i]
        if scores[i] < threshold and label == 1:
            tp += 1*weights[i]

    return (tp, fp, tn, fn)


def calc_metrics(cfm: tuple, samples: int) -> tuple:
    """Returns accuracy, precision, recall, and F1 from a confusion matrix.

    :param confusion_matrix: confusion matrix ([TN, FP], [FN, TP])
    :param samples: number of samples
    :return tuple: accuracy, precision, recall, and F1
    """

    tn, fp, fn, tp = cfm[0][0], cfm[0][1], cfm[1][0], cfm[1][1]
    acc = (tp + tn) / samples
    prec = tp / (tp + fp) if tp + fp != 0 else 0
    rec = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec != 0 else 0

    return acc, prec, rec, f1


def get_metrics() -> tuple:
    """Returns a tuple of metrics for a given log file.

    Returns:
        tuple: AUC, weighted TP and FP, accuracy, precision, recall, and F1.
    """

    # Get all values from log file
    labels, scores, weights = [], [], []
    with open('data/logs/eval_pairs.log', 'r', encoding='utf8') as file:
        for line in file:
            line = line.split()
            labels.append(int(line[4]))
            weights.append(float(line[5]))
            scores.append(float(line[6]))

    # Calculate AUC
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)

    # Weighted and non-weighted confusion matrices
    auc_thresh = thresholds[np.argmax(tpr - fpr)]
    wcm = weighted_conf_mat(labels, weights, scores, auc_thresh)  # weighted
    ncm = metrics.confusion_matrix(labels, scores < auc_thresh)  # non-weighted
    mets = calc_metrics(ncm, len(labels))  # acc, prec, rec, f1

    return (auc, wcm[0], wcm[1], mets[0], mets[1], mets[2], mets[3])


def test_layers():
    """Tests performance of each layer in every ESM-2 model checkpoint.
    """

    checkpoints = ['t30', 't33', 't36']
    for ch in checkpoints:

        # Load checkpoint and get number of layers
        model = Model('esm2', ch)
        layers = model.encoder.num_layers

        # For each layer, embed sequences and evaluate performance
        for lay in range(1, layers+1):
            os.system(f'python scripts/embed_seqs.py -c {ch} -l {lay} -q 5 50')
            os.system('python scripts/eval_pairs.py -m dct')
            mets = get_metrics()

            # Unpack and log metrics
            with open('data/logs/test_layers.log', 'a', encoding='utf8') as file:
                file.write(f'{datetime.now()}: {ch}-{lay}\t')
                file.write('\t'.join([str(round(m, 2)) for m in mets]) + '\n')


def main():
    """Main
    """

    test_layers()


if __name__ == '__main__':
    main()
