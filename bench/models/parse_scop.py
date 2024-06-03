"""This script downloads a clustered version of the SCOP database and writes sequences
to another fasta file with their classifications on the ID line.

__author__ = "Ben Iovino"
__date__ = "12/17/23"
"""

import os
import regex as re
from utils.utils import read_fasta, download_file


def parse_class(filename: str) -> dict[str, list[str]]:
    """Returns a dictionary of classifications for each protein.

    Args:
        filename (str): Name of SCOP classification file to parse.

    Returns:
        dict: dictionary where key is protein ID and value is a list of the fold,
            superfamily, and family classifications
    """

    classes: dict[str, list[str]] = {}
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            if line[0] == '#':  # Header lines
                continue

            # Get protein ID and classifications
            line = line.split()
            pid, sccs = line[0], line[3]
            classes[pid] = sccs

    return classes


def parse_db(filename: str) -> dict[str, str]:
    """Returns a dictionary of sequences.

    Args:
        filename (str): Name of file to parse.

    Returns:
        dict: dictionary where key is protein ID and value is the fasta sequence
    """

    seqs = read_fasta(filename)
    nseqs: dict[str, str] = {}  # Updated dict with discontinuous sequences
    for pid, seq in seqs.items():
        pid = pid.split()[0]
        if pid.startswith('e'):  # PID starts with 'e' if it is discontinuous
            reg = r'[a-zA-Z0-9]*\.[0-9]'
            pid = 'd' + re.search(reg, pid[1:]).group()
        nseqs[pid] = seq.upper()

    return nseqs


def write_seqs(path: str, classes: dict, seqs: dict):
    """Writes sequences in seqs to fasta file with classifications on ID line.

    Args:
        path (str): Path to write fasta file to.
        classes (dict): dictionary where key is protein ID and value is a list of the
            classifications
        seqs (dict): dictionary where key is protein ID and value is the fasta sequence
    """

    with open(f'{path}/scop_seqs.fa', 'w', encoding='utf8') as file:
        for pid, seq in seqs.items():
            cla = classes[pid]
            file.write(f'>{pid}|{cla}\n{seq}\n')


def main():
    """Main function
    """

    path = 'bench/models/data/scop'
    if not os.path.exists(path):
        url = 'https://scop.berkeley.edu'
        file1 = '?ver=2.08&get=bib&seqOption=1&item=seqs&cut=20'
        file2 = 'dir.cla.scope.2.08-stable.txt'
        download_file(f'{url}/astral/subsets', file1, path)
        os.rename(f'{path}/{file1}', f'{path}/scop20.fa')
        download_file(f'{url}/downloads/parse', file2, path)
        os.rename(f'{path}/{file2}', f'{path}/scop_class.txt')

    # Parse files and write fasta file with descriptions on ID line
    classes = parse_class(f'{path}/scop_class.txt')
    seqs = parse_db(f'{path}/scop20.fa')
    write_seqs(path, classes, seqs)


if __name__ == '__main__':
    main()
