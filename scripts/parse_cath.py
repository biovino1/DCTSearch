"""This script downloads a clustered version of the CATH database and writes sequences
to another fasta file with their classifications on the ID line.

__author__ = "Ben Iovino"
__date__ = "1/08/24"
"""

import os
import urllib.request
import regex as re
from Bio import SeqIO


def download_file(url: str, filename: str):
    """Downloads url to data directory using FTP.

    Args:
        url (str): URL of file to download.
        filename (str): Name of file to save to.
    """

    if not os.path.exists('data'):
        os.mkdir('data')
    urllib.request.urlretrieve(url, f'data/{filename}')


def parse_class(filename: str) -> dict:
    """Returns a dictionary of classifications for each protein.

    Args:
        filename (str): Name of CATH classification file to parse.

    Returns:
        dict: dictionary where key is protein ID and value is homologous family ID
    """

    classes = {}
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            line = line.split()
            classes[line[0]] = line[4]

    return classes


def write_classes(filename: str, classes: dict):
    """Writes classifications for each protein to fasta file.

    Args:
        filename (str): Name of file to add classifications to.
        classes (dict): Dictionary of classifications for each protein.
    """

    # Get all seqs from fasta file
    seqs = {}
    for seq in SeqIO.parse(filename, 'fasta'):
        pid = re.findall(r'\|([a-zA-Z0-9]*)\/', seq.id)[0]
        pid += f"\t{classes[pid]}"
        seqs[pid] = str(seq.seq)

    # Write to new fasta file
    with open('data/cath_seqs.fa', 'w', encoding='utf8') as file:
        for pid, seq in seqs.items():
            file.write(f'>{pid}\n{seq}\n')


def main():
    """Main function
    """

    # Download database and classification file
    url1 = 'ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/' \
        'non-redundant-data-sets/cath-dataset-nonredundant-S20.fa'
    url2 = 'ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/' \
        'cath-classification-data/cath-domain-list-S100.txt'
    download_file(url1, 'cath20.fa')
    download_file(url2, 'cath_class.txt')

    # Parse classification file and add ID's to fasta file
    classes = parse_class('data/cath_class.txt')
    write_classes('data/cath20.fa', classes)


if __name__ == '__main__':
    main()
