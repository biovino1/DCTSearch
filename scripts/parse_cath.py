"""This script downloads a clustered version of the CATH database and writes sequences
to another fasta file with their classifications on the ID line.

__author__ = "Ben Iovino"
__date__ = "1/08/24"
"""

import os
import urllib.request
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


def main():
    """Main function
    """

    # Download database and classification file
    url1 = 'ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/' \
        'non-redundant-data-sets/cath-dataset-nonredundant-S20.fa'
    url2 = 'ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/' \
        'cath-classification-data/cath-domain-list-S35.txt'
    download_file(url1, 'cath20.fa')
    download_file(url2, 'cath_class.txt')


if __name__ == '__main__':
    main()
