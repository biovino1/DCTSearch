"""Evaluates all protein pairs in a file.

__author__ = "Ben Iovino"
__date__ = "12/19/23"
"""

import argparse
import logging
import os
import subprocess as sp
from datetime import datetime
import numpy as np
import regex as re
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
        list: List of protein pairs.
    """

    with open(filename, 'r', encoding='utf8') as file:
        pairs = []
        for line in file:
            line = line.split()
            pairs.append((line[0], line[1], line[2]))

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


def load_fdb(filename: str) -> dict:
    """Returns a dictionary of DCT fingerprints where each key is a protein id and each value
    is a list of fingerprints for each domain.

    Args:
        filename (str): Name of file to parse.

    Returns:
        dict: Dictionary of DCT fingerprints.
    """

    fdb = np.load(filename)
    fprints = {}

    # Get all fingerprints associated with each pid
    for i, pid in enumerate(fdb['pids']):
        try:
            idx = (fdb['idx'][i], fdb['idx'][i+1])
        except IndexError:  # Last protein in file
            idx = (fdb['idx'][i], len(fdb['doms']))
        quants = fdb['quants'][idx[0]:idx[1]]
        fprints[pid] = quants

    return fprints


def dct_search(filename: str, pairs: list):
    """Finds distance between each pair of proteins using the manhattan distance between
    their DCT fingerprints.

    Args:
        filename (str): Name of file containing DCT fingerprints.
        pairs (list): List of protein pairs.
    """

    fprints = load_fdb(filename)

    # Find most similar fingerprint for each pair of proteins
    for pair in pairs:
        quants1 = fprints[pair[0]]
        quants2 = fprints[pair[1]]

        # Find similarity between each pair of fingerprints
        max_sim = 0
        for q1 in quants1:
            for q2 in quants2:
                sim = 1-abs(q1-q2).sum()/17000
                if sim > max_sim:
                    max_sim = sim
        logging.info('%s: %s %s %s %s',
                      datetime.now(), pair[0], pair[1], pair[2], max_sim)


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
    """Finds bitscore between each pair of proteins using BLAST.

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


def csblast_search(pairs: list, seqs: dict):
    """Returns the bitscore between each pair of proteins using CS-BLAST.
    
    Args:
        pairs (list): List of protein pairs.
        seqs (dict): Dictionary of protein sequences.
    """

    direc, db_seq = 'data/csblast', ''
    os.makedirs(direc, exist_ok=True)
    for pair in pairs:

        # Make BLAST database if new sequence
        if db_seq != seqs[pair[0]]:
            db_seq = seqs[pair[0]]
            write_seq(f'{direc}/db.fa', pair[0], db_seq)
            os.system(f'formatdb -t {direc}/db -i {direc}/db.fa -p T -l {direc}/formatdb.log')

        # Query is always new, write to file
        query_seq = seqs[pair[1]]
        write_seq(f'{direc}/q.fa', pair[1], query_seq)

        # Get bitscore and log
        csblast = f'csblast -i {direc}/q.fa -d {direc}/db.fa ' \
              '-D /home/ben/anaconda3/data/K4000.lib --blast-path $CONDA_PREFIX/bin -e 1e9'
        result = sp.getoutput(csblast)

        # Search for bitscore with regex 'Score = x.x bits'
        reg = re.compile(r'Score = (\d+.\d+) bits')
        result = reg.findall(result)
        try:
            result = result[0]
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

    # Make sure you have database in data directory, use scop70 for this project
    os.system(f'ffindex_from_fasta -s {direc}/db_fas.ffdata '
                f'{direc}/db_fas.ffindex {direc}/db.fas')
    os.system(f'hhblits_omp -i {direc}/db_fas -d data/scop70_01Mar17/scop70 '
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


def hhblits_search(pairs: list, seqs: dict):
    """Finds bitscore between each pair of proteins using HHsearch.

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
        hhsearch = f'hhblits -i {direc}/query_seq.fa -d {direc}/db -E 1e9'
        result = sp.getoutput(hhsearch)
        result = get_hh_score(result)
        logging.info('%s: %s %s %s %s %s',
                      datetime.now(), pair[0], pair[1], pair[2], pair[3], result)

    os.system(f'rm -rf {direc}')


def search(method: str, db: str, pairs: list, seqs: dict):
    """Calls the appropriate search function.

    Args:
        method (str): Method to use for search.
        db (str): Database to use for DCT search.
        pairs (list): List of protein pairs.
        seqs (dict): Dictionary of protein sequences.
    """

    if method == 'dct':
        dct_search(f'data/{db}_quants.npz', pairs)
    elif method == 'blast':
        blast_search(pairs, seqs)
    elif method == 'csblast':
        csblast_search(pairs, seqs)
    elif method == 'hhsearch':
        hhblits_search(pairs, seqs)
    else:
        raise ValueError(f'Invalid method: {method}')


def main():
    """Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='scop', help='Database to use')
    parser.add_argument('-m', type=str, default='dct', help='Method to use')
    args = parser.parse_args()

    # Get all pairs and evaluate
    pairs = get_pairs(f'data/{args.d}_pairs.txt')
    seqs = get_seqs(f'data/{args.d}_seqs.fa')
    search(args.m, args.d, pairs, seqs)


if __name__ == '__main__':
    main()
