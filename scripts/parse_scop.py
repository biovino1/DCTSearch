"""This script downloads a clustered version of the SCOP database and writes sequences
to another fasta file with their classifications on the ID line.

__author__ = "Ben Iovino"
__date__ = "12/17/23"
"""

import os
import regex as re
import requests
from Bio import SeqIO


def download_file(url: str, filename: str):
    """Downloads url to data directory.

    Args:
        url (str): URL to download.
        filename (str): Name of file to save to.
    """

    if not os.path.exists('data'):
        os.mkdir('data')
    req = requests.get(url, timeout=10)
    with open(f'data/{filename}', 'w', encoding='utf8') as file:
        file.write(req.text)


def parse_class(filename: str) -> dict:
    """Returns a dictionary of classifications for each protein.

    Args:
        filename (str): Name of SCOP classification file to parse.

    Returns:
        dict: dictionary where key is protein ID and value is a list of the fold,
            superfamily, and family classifications
    """

    classes = {}
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            if line[0] == '#':  # Header lines
                continue

            # Get protein ID and classifications
            line = line.split()
            pid = line[0]
            reg = r'cf=[0-9]*'  # Fold ID
            cla = re.search(reg, line[5]).group()[3:]
            classes[pid] = cla

    return classes


def parse_db(filename: str) -> dict:
    """Returns a dictionary of sequences.

    Args:
        filename (str): Name of file to parse.

    Returns:
        dict: dictionary where key is protein ID and value is the fasta sequence
    """

    seqs = {}
    for record in SeqIO.parse(filename, 'fasta'):
        pid = record.id
        if pid.startswith('e'):  # PID starts with 'e' if it is discontinuous
            reg = r'[a-zA-Z0-9]*\.[0-9]'
            pid = 'd' + re.search(reg, pid[1:]).group()
        seqs[pid] = seqs.get(pid, '') + str(record.seq.upper())

    return seqs


def write_seqs(classes: dict, seqs: dict):
    """Writes sequences in seqs to fasta file with classifications on ID line.

    Args:
        classes (dict): dictionary where key is protein ID and value is a list of the
            classifications
        seqs (dict): dictionary where key is protein ID and value is the fasta sequence
    """

    with open('data/scop_seqs.fa', 'w', encoding='utf8') as file:
        for pid, seq in seqs.items():
            cla = classes[pid]
            file.write(f'>{pid}\t{cla}\n{seq}\n')


def main():
    """Main function
    """

    # Download database and classification file
    #url1 = 'https://scop.berkeley.edu/astral/subsets/?ver=2.08&get=bib&seqOption=1&item=seqs&cut=20'
    #url2 = 'https://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.08-stable.txt'
    #download_file(url1, 'scop20.fa')
    #download_file(url2, 'scop_class.txt')

    # Parse files and write fasta file with descriptions on ID line
    classes = parse_class('data/scop_class.txt')
    seqs = parse_db('data/scop20.fa')
    write_seqs(classes, seqs)


if __name__ == '__main__':
    main()
