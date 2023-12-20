"""Evaluates all protein pairs in a file.

__author__ = "Ben Iovino"
__date__ = "12/19/23"
"""

import logging
import os
import pickle
import subprocess
from datetime import datetime
from Bio import SeqIO
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


def get_seqs(filename: str) -> dict:
    """Returns a dictionary of protein sequences.

    Args:
        filename (str): Name of file to parse.
    
    Returns:
        dict: Dictionary of protein sequences, where key is protein id and value is sequence.
    """

    seqs = {}
    for record in SeqIO.parse(filename, 'fasta'):
        seqs[record.id] = str(record.seq)

    return seqs


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


def blast_search(pairs: list, seqs: dict):
    """Finds E-value between each pair of proteins using BLAST.

    Args:
        pairs (list): List of protein pairs.
        seqs (dict): Dictionary of protein sequences.
    """

    direc = 'data/blast'
    os.makedirs(direc, exist_ok=True)
    db_seq = ''
    for pair in pairs:

        # Make BLAST database if new sequence
        if db_seq != seqs[pair[0]]:
            db_seq = seqs[pair[0]]
            os.system(f'rm -rf {direc}')
            os.makedirs(direc, exist_ok=True)
            with open(f'{direc}/db_seq.fa', 'w', encoding='utf8') as file:
                file.write(f'>{pair[0]}\n{db_seq}')
            os.system(f'makeblastdb -in {direc}/db_seq.fa '
                      f'-dbtype prot -parse_seqids -out {direc}/blastdb/db_seq')

        # Query is always new, write to file
        query_seq = seqs[pair[1]]
        with open(f'{direc}/query_seq.fa', 'w', encoding='utf8') as file:
            file.write(f'>{pair[1]}\n{query_seq}')

        # Get E-value and log
        result = subprocess.getoutput(f'blastp -query {direc}/query_seq.fa '
                                     f'-db {direc}/blastdb/db_seq -evalue 1e6 -outfmt "6 bitscore"')
        try:
            result = result.split()[0]  # Top E-value
        except IndexError:  # No hits detected
            result = 0
        logging.info('%s: %s %s %s %s %s',
                      datetime.now(), pair[0], pair[1], pair[2], pair[3], result)


def main():
    """Main function
    """

    # Get all pairs
    pairs = get_pairs('data/scop_pairs.txt')
    seqs = get_seqs('data/scop_seqs.fa')

    # Evaluate each pair with given method
    #dct_search(pairs)
    blast_search(pairs, seqs)


if __name__ == '__main__':
    main()
