"""Evaluates all protein pairs in a file.

__author__ = "Ben Iovino"
__date__ = "12/19/23"
"""

import logging
import os
import pickle
import subprocess as sp
from datetime import datetime
from Bio import SeqIO

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
        dist = abs(quants[pair[0]] - quants[pair[1]]).sum()
        logging.info('%s: %s %s %s %s %s',
                      datetime.now(), pair[0], pair[1], pair[2], pair[3], dist)


def write_seq(filename: str, pid: str, seq: str):
    """Writes a protein sequence to file.

    Args:
        filename (str): Name of file to write to.
        pid (str): Protein id.
        seq (str): Protein sequence.
    """

    with open(f'{filename}', 'w', encoding='utf8') as file:
        file.write(f'>{pid}\n{seq}')


def blast_search(pairs: list, seqs: dict):
    """Finds E-value between each pair of proteins using BLAST.

    Args:
        pairs (list): List of protein pairs.
        seqs (dict): Dictionary of protein sequences.
    """

    direc, db_seq = 'data/blast', ''
    os.makedirs(direc, exist_ok=True)
    for pair in pairs:

        # Make BLAST database if new sequence
        if db_seq != seqs[pair[0]]:
            db_seq = seqs[pair[0]]
            write_seq(f'{direc}/db.fa', pair[0], db_seq)
            os.system(f'makeblastdb -in {direc}/db.fa -dbtype prot -out {direc}/bldb/db')

        # Query is always new, write to file
        query_seq = seqs[pair[1]]
        write_seq(f'{direc}/q.fa', pair[1], query_seq)

        # Get bitscore and log
        blastp = f'blastp -query {direc}/q.fa -db {direc}/bldb/db -evalue 1e9 -outfmt "6 bitscore"'
        result = sp.getoutput(blastp)
        try:
            result = result.split()[0]  # Top E-value
        except IndexError:  # No hits detected
            result = 0
        logging.info('%s: %s %s %s %s %s',
                      datetime.now(), pair[0], pair[1], pair[2], pair[3], result)

    os.system(f'rm -rf {direc}')


def make_hh_db(direc: str, pid: str, seq: str):
    """Makes a database for hhsearch.

    Args:
        direc (str): Directory to make database in.
        pid (str): Protein id.
        seq (str): Protein sequence.
    """

    # Remove old database and write seq for new one
    os.system(f'rm -rf {direc}')
    os.system(f'mkdir -p {direc}')
    write_seq(f'{direc}/db.fas', pid, seq)

    # Make sure you have database in data directory, use scop40 for this project
    os.system(f'ffindex_from_fasta -s {direc}/db_fas.ffdata '
                f'{direc}/db_fas.ffindex {direc}/db.fas')
    os.system(f'hhblits_omp -i {direc}/db_fas -d data/scop40_01Mar17/scop40 '
                f'-oa3m {direc}/db_a3m -n 2 -cpu 1 -v 0')
    os.system(f'ffindex_apply {direc}/db_a3m.ffdata {direc}/db_a3m.ffindex '
                f'-i {direc}/db_hmm.ffindex -d {direc}/db_hmm.ffdata '
                '-- hhmake -i stdin -o stdout -v 0')
    os.system('cstranslate -f -x 0.3 -c 4 -I a3m '
                f'-i {direc}/db_a3m -o {direc}/db_cs219')


def get_hh_score(result: str) -> float:
    """Returns the bitscore from a hhsearch result.

    Args:
        result (str): Stdout from hhsearch.

    Returns:
        float: Bitscore of result.
    """

    result_line = result.split('\n')
    score_line = [s.find('No Hit') for s in result_line]
    for j, score in enumerate(score_line):
        if score != -1:
            result_line = result_line[j+1].split()
            if result_line != []:
                result = float(result_line[5])
            else:
                result = 0

    return result


def hhsearch_search(pairs: list, seqs: dict):
    """This function takes a list of protein pairs, each pair being used to get their respective
    sequences from a dictionary of seqs. The first sequence is used as the 'database' and the second
    sequence is used as the query sequence in a hhsearch search.

    Args:
        pairs (list): List of protein pairs.
        seqs (dict): Dictionary of protein sequences.
    """

    direc, db_seq = 'data/hhsearch', ''
    os.makedirs(direc, exist_ok=True)
    for pair in pairs:

        # Make HHsearch DB if new sequence
        if db_seq != seqs[pair[0]]:
            db_seq = seqs[pair[0]]
            make_hh_db(direc, pair[0], db_seq)

        # Query sequence is always different so write to file
        query_seq = seqs[pair[1]]
        write_seq(f'{direc}/query_seq.fa', pair[1], query_seq)

        # Get bitscore and log
        result = sp.getoutput(f'hhsearch -i {direc}/query_seq.fa '
                                       f'-d {direc}/db -E 1000000000')
        result = get_hh_score(result)
        logging.info('%s: %s %s %s %s %s',
                      datetime.now(), pair[0], pair[1], pair[2], pair[3], result)

    os.system(f'rm -rf {direc}')


def main():
    """Main function
    """

    # Get all pairs
    pairs = get_pairs('data/scop_pairs.txt')
    seqs = get_seqs('data/scop_seqs.fa')

    # Evaluate each pair with given method
    #dct_search(pairs)
    #blast_search(pairs, seqs)
    hhsearch_search(pairs, seqs)


if __name__ == '__main__':
    main()
