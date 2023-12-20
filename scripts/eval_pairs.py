"""Evaluates all protein pairs in a file.

__author__ = "Ben Iovino"
__date__ = "12/19/23"
"""

import logging
import os
import pickle
from datetime import datetime
from scipy.spatial.distance import cityblock

log_filename = 'data/logs/eval_pairs.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def get_pairs(filename: str) -> list:
    """Returns a list of protein pairs.

    Args:
        filename (str): Name of file to parse.
    
    Returns:
        list: List of protein pairs, along with their classification and weight.
    """

    with open(filename, 'r', encoding='utf8') as file:
        pairs = []
        for line in file:
            line = line.split()
            pairs.append((line[0], line[1], line[2], float(line[3])))

    return pairs


def dct_search(pairs: list):
    """Finds distance between each pair of proteins using the manhattan distance between
    their DCT fingerprints.

    Args:
        pairs (list): List of protein pairs.
    """

    # Load DCT fingerprints
    with open('data/scop_quants.pkl', 'rb') as qfile:
        quants = pickle.load(qfile)

    # Find distance and log
    for pair in pairs:
        dist = cityblock(quants[pair[0]], quants[pair[1]])
        logging.info('%s: %s %s %s %s %s',
                      datetime.now(), pair[0], pair[1], pair[2], pair[3], dist)


def main():
    """Main function
    """

    # Get all pairs
    pairs = get_pairs('data/scop_pairs.txt')

    # Evaluate each pair with given method
    dct_search(pairs)


if __name__ == '__main__':
    main()
