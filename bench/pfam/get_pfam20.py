"""Downloads Pfam v33.1 and prepares it for benchmarking.

__author__ = "Ben Iovino"
__date__ = "5/08/24"
"""

import argparse
import os
from random import sample
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
from urllib.request import urlretrieve
from zipfile import ZipFile


def sample_seqs(path: str, fam: str, pids: list[str], n: int):
    """Writes n number of sequences from dictionary to file.

    Args:
        path (str): Path to write sequences to.
        fam (str): Pfam family.
        pids (list[str]): List of protein ID's
        n (int): Number of sequences to write.
    """

    if len(pids) < n:
       return
    pids = sample(pids, n)
    
    # Write each pid to file for getting sequence later
    with open(f'{path}/pfam20.txt', 'a', encoding='utf-8') as file:
        for pid in pids:
            file.write(f'{pid}\n')


def read_pfam(path: str):
    """Reads Pfam-A.fasta and samples a number of sequences from each family.
    """

    fam, pids = '', []
    with open(f'{path}/Pfam-A.fasta', 'r', encoding='utf-8') as file:
        for line in file:

            # New sequence, sample current group of seqs if new family
            if line.startswith('>'):
                line = line.split(';')
                if line[1] != fam:
                    sample_seqs(path, fam, pids, 20)
                    fam, pids = line[1], []
                pid = line[0].split('/')[0][1:]
                pids.append(pid)


def get_files(path: str):
    """Downloads files from ftp.ebi.ac.uk.

    Args:
        path (str): Path to download files to.
    """

    if not os.path.exists(path):
        os.mkdir(path)

    # Fasta file
    file = 'Pfam-A.fasta.gz'
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
        read_pfam(path)


if __name__ == "__main__":
    main()
