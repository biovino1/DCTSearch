"""Downloads Pfam v33.1 and prepares it for benchmarking.

__author__ = "Ben Iovino"
__date__ = "5/08/24"
"""

import argparse
import os
from random import sample
import subprocess as sp
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
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
    

def get_seqs(path: str):
    """Gets full length sequences from pfamseq and writes them to file.

    Args:
        path (str): Path to read from and write sequences to.
    """

    # Prepare dictionary of sequences
    seqs: dict[str, str] = {}
    with open(f'{path}/pfam20.txt', 'r', encoding='utf-8') as file:
        for line in file:
            seqs[line.strip()] = ''

    # Read full-length sequences from pfamseq
    full_seqs = read_fasta(f'{path}/pfamseq')
    for pid in seqs:
        seqs[pid] = full_seqs[pid.split('|')[0]]

    # Write sequences to file
    with open(f'{path}/pfam20.fa', 'w', encoding='utf-8') as file:
        for pid, seq in seqs.items():
            file.write(f'>{pid}\n{seq}\n')


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
            file.write(f'{pid}|{fam}\n')


def read_pfam(path: str):
    """Reads Pfam-A.fasta and samples a number of sequences from each family.

    Args:
        path (str): Path to read Pfam-A.fasta from.
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
    """Downloads files from ftp.ebi.ac.uk. Both files can take several hours to download.
    Pfam-A.fasta is 3.4G zipped and 7.5G unzipped, pfamseq is 11G zipped and 21G unzipped.

    Args:
        path (str): Path to download files to.
    """

    if not os.path.exists(path):
        os.mkdir(path)

    # Fasta file
    file = 'Pfam-A.fasta.gz'
    url = 'http://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam33.1'
    print(f'Downloading {url}/{file}...')
    urlretrieve(f'{url}/{file}', f'{path}/{file}')
    with ZipFile(f'{path}/{file}', 'r') as zip_ref:
        zip_ref.extractall(f'{path}/')

    # Full sequence file
    file = 'pfamseq.gz'
    print(f'Downloading {url}/{file}...')
    urlretrieve(f'{url}/{file}', f'{path}/{file}')
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
        get_seqs(path)

    # Fingerprint fasta file
    if args.gpu:
        sp.run(['python', 'src/make_db.py', f'--fafile={path}/pfam20.fa', f'--dbfile={path}/pfam20',
                f'--maxlen={args.maxlen}', f'--cpu={args.cpu}', f'--gpu={args.gpu}'])
    else:
        sp.run(['python', 'src/make_db.py', f'--fafile={path}/pfam20.fa', f'--dbfile={path}/pfam20',
            f'--maxlen={args.maxlen}', f'--cpu={args.cpu}'])


if __name__ == "__main__":
    main()
