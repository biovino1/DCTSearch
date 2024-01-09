"""Creates a file of homologous/non-homologous protein pairs given their classifications.

__author__ = "Ben Iovino"
__date__ = "12/19/23"
"""

import argparse
from random import sample
import regex as re
from Bio import SeqIO


def get_classes(filename: str) -> dict:
    """Returns a dictionary of classifications and their proteins.

    Args:
        filename (str): Name of file to parse.

    Returns:
        dict: dictionary where key is classifications and value is a list of proteins.
    """

    classes = {}
    for record in SeqIO.parse(filename, 'fasta'):
        cla = re.findall(r'\t(\d*)', record.description)[0]
        classes[cla] = classes.get(cla, []) + [record.id]

    return classes


def get_homs(classes: dict, samp: int) -> list:
    """Returns a list of pairs of proteins within each class along with their homology status and 
    class weight, 1 / (number of proteins in class).

    Args:
        classes (dict): Dictionary of classifications and their proteins.
        samp (int): Number of pairs to sample.

    Returns:
        list: List of tuples where each tuple is a pair of proteins, their homology status, and 
            class weight.
    """

    # Get homologous pairs
    pairs = []
    for _, pids in classes.items():
        weight = 1 / len(pids)  # Weight for each pair given class size
        for i in range(len(pids)):  #pylint: disable=C0200
            for j in range(i + 1, len(pids)):
                pairs.append((pids[i], pids[j], 1, weight))
    pairs = sample(pairs, samp)  # Random sample pairs

    return pairs


def get_nonhoms(classes: dict, samp: int) -> list:
    """Returns a list of pairs of proteins within each class along with their homology status and
    class weight, 1 / (number of proteins in class).

    Args:
        classes (dict): Dictionary of classifications and their proteins.
        samp (int): Number of pairs to sample.

    Returns:
        list: List of tuples where each tuple is a pair of proteins, their homology status, and
            class weight.
    """

    # Get non-homologous pairs
    pairs = []
    for i, pids1 in enumerate(classes.values()):
        for j, pids2 in enumerate(classes.values()):
            if i >= j:  # Only need to compare each pair once
                continue
            weight = 1 / (len(pids2))  # Weight for each pair given "query" class size
            for pid1 in pids1:
                for pid2 in pids2:
                    pairs.append((pid1, pid2, 0, weight))
    pairs = sample(pairs, samp)  # Random sample pairs

    return pairs


def write_pairs(pairs: list, filename: str):
    """Writes protein pairs to file.

    Args:
        pairs (list): List of protein pairs.
        filename (str): Name of file to write pairs to.
    """

    with open(filename, 'w', encoding='utf8') as pfile:
        for pair in pairs:
            pfile.write(f'{pair[0]} {pair[1]} {pair[2]} {round(pair[3], 3)} \n')


def main():
    """Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='data/cath_seqs.fa', help='Fasta file')
    parser.add_argument('-s', type=int, default=250, help='Number of pairs to sample')
    args = parser.parse_args()

    # Get classifications and all pairs within and in between
    classes = get_classes(args.f)
    hom_pairs = get_homs(classes, args.s)
    nonhom_pairs = get_nonhoms(classes, args.s)

    # Get pairs file name and write pairs to file
    pairs_file = re.search(r'\/([a-zA-Z]*)', args.f).group(1)
    pairs_file = f'data/{pairs_file}_pairs.txt'
    pairs = hom_pairs + nonhom_pairs
    write_pairs(pairs, pairs_file)


if __name__ == '__main__':
    main()
