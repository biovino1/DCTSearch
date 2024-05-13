"""Downloads Pfam v33.1 and prepares it for benchmarking.

__author__ = "Ben Iovino"
__date__ = "5/08/24"
"""

import argparse
import os
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
import subprocess as sp
from urllib.request import urlretrieve
from zipfile import ZipFile


def get_files(path: str):
    """Downloads files from ftp.ebi.ac.uk.

    Args:
        path (str): Path to download files to.
    """

    if not os.path.exists(path):
        os.mkdir(path)

    # Fasta file
    file = 'Pfam-A.full.gz'
    url = f"http://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam33.1/{file}"
    print(f'Downloading {file}...')
    urlretrieve(url, f'{path}/{file}')
    with ZipFile(f'{path}/{file}', 'r') as zip_ref:
        zip_ref.extractall(f'{path}/')


def main():
    """Downloads Pfam 33.1 and fingerprints sequences.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--maxlen', type=int, default=1000, help='max sequence length to embed')
    parser.add_argument('--cpu', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--gpu', type=int, required=False, help='number of gpus to use')
    args = parser.parse_args()

    path = 'bench/pfam/data'
    if not os.path.exists(path):
        get_files(path)


if __name__ == "__main__":
    main()
