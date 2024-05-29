"""Utility functions.

__author__ = "Ben Iovino"
__date__ = "5/29/24"
"""

import os
from urllib.request import urlretrieve
from zipfile import ZipFile



def read_fasta(fafile: str) -> dict[str, str]:
    """Returns a dictionary of sequences from a fasta file.

    Args:
        fafile (str): Path to fasta file.

    Returns:
        dict: Dictionary of sequences.
    """

    seqs = {}
    with open(fafile, 'r', encoding='utf8') as f:
        for line in f:
            if line.startswith('>'):
                pid = line.split()[1]
                seqs[pid] = ''
            else:
                seqs[pid] += line.strip()
        
    return seqs


def download_file(url: str, filename: str, path: str):
    """Downloads filename (any file) from url to path.

    Args:
        url (str): URL to download.
        filename (str): Name of file to save to.
        path (str): Path to save file to.
    """

    if not os.path.exists(path):
        os.mkdir(path)
    print(f'Downloading {url}/{filename} to {path}...')
    urlretrieve(f'{url}/{filename}', f'{path}/{filename}')

    # Unzip if necessary
    if filename.endswith('.gz'):
        with ZipFile(f'{path}/{filename}', 'r') as zip_ref:
            zip_ref.extractall(path)
        os.remove(f'{path}/{filename}')
