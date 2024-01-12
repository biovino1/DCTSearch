"""Testing performance of each layer in every ESM-2 model checkpoint.

__author__ = "Ben Iovino"
__date__ = "1/9/24"
"""

import os
from datetime import datetime
import torch
import numpy as np
from embed import Model, Transform
from sklearn import metrics
from make_db import load_seqs


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
        tuple: AUC, TP/FP, accuracy, precision, recall, and F1.
    """

    # Get all values from log file
    labels, scores = [], []
    with open('data/logs/eval_pairs.log', 'r', encoding='utf8') as file:
        for line in file:
            line = line.split()
            labels.append(int(line[4]))
            scores.append(float(line[5]))

    # Calculate AUC
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)

    # Get confusion matrix and metrics
    auc_thresh = thresholds[np.argmax(tpr - fpr)]
    ncm = metrics.confusion_matrix(labels, scores < auc_thresh)
    mets = calc_metrics(ncm, len(labels))  # acc, prec, rec, f1

    return (auc, ncm[1][1], ncm[0][1], mets[0], mets[1], mets[2], mets[3])


def test_layers():
    """Tests performance of each layer in every ESM-2 model checkpoint.
    """

    checkpoints = ['t6', 't12', 't30', 't33', 't36']
    for ch in checkpoints:

        # Load checkpoint and get number of layers
        model = Model('esm2', ch)
        layers = model.encoder.num_layers

        # For each layer, embed sequences and evaluate performance
        for lay in range(1, layers+1):
            os.system(f'python scripts/make_db.py -c {ch} -l {lay} -q 5 50')
            os.system('python scripts/eval_pairs.py -m dct')
            mets = get_metrics()

            # Unpack and log metrics
            with open('data/logs/test_layers.log', 'a', encoding='utf8') as file:
                file.write(f'{datetime.now()}: {ch}-{lay}\t')
                file.write('\t'.join([str(round(m, 2)) for m in mets]) + '\n')


def embed_seqs(seqs: dict, layers: list, ch: str) -> list:
    """Returns a tuple of protein ID's and their corresponding embeddings.

    Args:
        seqs (dict): Dictionary of protein sequences
        layers (list): Layers to use for embedding.
        ch (str): ESM-2 checkpoint.

    Returns:
        tuple: List of Transform objects
    """

    model = Model('esm2', ch)  # pLM encoder and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    model.to_device(device)

    # Just embed seqs
    transforms = []
    for pid, seq in seqs.items():
        trans = Transform(pid=pid, seq=seq)
        trans.esm2_embed(model, device, layers=layers)
        transforms.append(trans)

    return transforms


def test_dims():
    """Tests performance of different iDCT quantization parameters.
    """

    q1 = [3, 4, 5, 6, 7, 8, 9, 10]
    q2 = [30, 40, 50, 60, 70, 80, 90, 100]
    db = "cath"

    # Embed sequences
    seqs = load_seqs(f'data/{db}_seqs.fa')
    transforms = embed_seqs(seqs, [15], 't30')

    # Grid search
    for q_1 in q1:
        for q_2 in q2:

            # Quantize embeddings
            pids, quants = [], []
            for trans in transforms:
                trans.quantize([q_1, q_2])
                pids.append(trans.pid)
                quants.append(trans.quant)

            # Make lists into numpy arrays and save as npz
            quants = np.array(quants)
            np.savez_compressed('data/test_db.npz', pids=pids, quants=quants)

            # Evaluate performance
            os.system(f'python scripts/eval_pairs.py -d {db} -m dct')
            mets = get_metrics()

            # Unpack and log metrics
            with open('data/logs/test_dims.log', 'a', encoding='utf8') as file:
                file.write(f'{datetime.now()}: {q_1}-{q_2}\t')
                file.write('\t'.join([str(round(m, 2)) for m in mets]) + '\n')


def main():
    """Main
    """

    test_dims()


if __name__ == '__main__':
    main()
