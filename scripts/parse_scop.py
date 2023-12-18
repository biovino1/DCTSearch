"""This script downloads a clustered version of the SCOP database and parses the sequences
into groups based on their SCOP classificaiton.

__author__ = "Ben Iovino"
__date__ = "12/17/23"
"""

import os
import requests


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
    """Parses the SCOP classification file into a dictionary.

    Args:
        filename (str): Name of file to parse.

    Returns:
        dict: dictionary where key is protein ID and value is a list of classifications
    """

    classes = {}
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            if line[0] == '#':  # Header lines
                continue

            # Get protein ID and classifications
            line = line.split()
            pid, cla = line[0], line[5]
            fold, superfam, fam = cla.split(',')[1:4]
            classes[pid] = [fold, superfam, fam]

    return classes


def main():
    """Main function
    """

    # Download database and classification file
    url1 = 'https://scop.berkeley.edu/astral/subsets/?ver=2.08&get=bib&seqOption=1&item=seqs&cut=20'
    url2 = 'https://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.08-stable.txt'
    download_file(url1, 'scop20.fa')
    download_file(url2, 'scop_class.txt')

    # Parse files
    classes = parse_class('data/scop_class.txt')



if __name__ == '__main__':
    main()
