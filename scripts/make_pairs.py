"""Creates a file of homologous/non-homologous protein pairs given their classifications.

__author__ = "Ben Iovino"
__date__ = "12/19/23"
"""

import argparse
from random import sample
import regex as re
from Bio import SeqIO


def get_folds(filename: str) -> dict:
    """Returns a dictionary of folds and their proteins.

    Args:
        filename (str): Name of file to parse.

    Returns:
        dict: dictionary where key is fold and value is a list of proteins.
    """

    folds = {}
    for record in SeqIO.parse(filename, 'fasta'):
        fold = re.findall(r'=[0-9]*', record.description)[0][1:]  # 0-fold, 1-sf, 2-fam
        folds[fold] = folds.get(fold, []) + [record.id]

    return folds


def write_pairs(cla: str, pairs: list):
    """Writes protein pairs to file.

    Args:
        fold (str): Hom/nonhom designation.
        pairs (list): List of protein pairs.
    """

    with open('data/scop_pairs.txt', 'a', encoding='utf8') as file:
        for pair in pairs:
            file.write(f'{pair[0]} {pair[1]} {cla} {round(pair[2], 3)} \n')


def get_homs(folds: dict, samp: int):
    """Gets all pairs of proteins within each fold and writes each pair to file along with
    their fold weight, 1 / (number of proteins in fold).

    Args:
        folds (dict): Dictionary of folds and their proteins.
        samp (int): Number of pairs to sample.
    """

    # Get homologous pairs
    pairs = []
    for _, pids in folds.items():
        weight = 1 / len(pids)  # Weight for each pair given fold size
        for i in range(len(pids)):  #pylint: disable=C0200
            for j in range(i + 1, len(pids)):
                pairs.append((pids[i], pids[j], weight))

    # Get random sample and write to file
    pairs = sample(pairs, samp)
    write_pairs('hom', pairs)


def get_nonhoms(folds: dict, samp: int):
    """Gets all pairs of proteins between each fold and writes each pair to file along
    with their fold weight, 1 / (number of proteins in fold).

    Args:
        folds (dict): Dictionary of folds and their proteins.
        samp (int): Number of pairs to sample.
    """

    # Get non-homologous pairs
    pairs = []
    for i, pids1 in enumerate(folds.values()):
        for j, pids2 in enumerate(folds.values()):
            if i >= j:  # Only need to compare each pair once
                continue
            weight = 1 / (len(pids2))  # Weight for each pair given "query" fold size
            for pid1 in pids1:
                for pid2 in pids2:
                    pairs.append((pid1, pid2, weight))

    # Get random sample and write to file
    pairs = sample(pairs, samp)
    write_pairs('nonhom', pairs)


def main():
    """Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int, default=250, help='Number of pairs to sample')
    args = parser.parse_args()

    # Get folds and all pairs within and in between
    folds = get_folds('data/scop_seqs.fa')
    get_homs(folds, args.s)
    get_nonhoms(folds, args.s)


if __name__ == '__main__':
    main()
