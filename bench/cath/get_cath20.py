"""Downloads CATH20 dataset and prepares it for benchmarking.

__author__ = "Ben Iovino"
__date__ = "4/13/24"
"""

import argparse
import os
import subprocess as sp
from urllib.request import urlretrieve


def get_files(path: str):
    """Downloads files from orengoftp.biochem.ucl.ac.uk.

    Args:
        path (str): Path to download files to.
    """

    if not os.path.exists(path):
        os.mkdir(path)

    # Fasta file
    prefix = "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/all-releases/v4_2_0/"
    url = prefix + "non-redundant-data-sets/cath-dataset-nonredundant-S20-v4_2_0.fa"
    print('Downloading CATH20 dataset...')
    urlretrieve(url, f'{path}/cath20.fa')

    # Classification file
    url = prefix + "cath-classification-data/cath-domain-list-v4_2_0.txt"
    print('Downloading CATH20 classification...')
    urlretrieve(url, f'{path}/cath20_class.txt')


def read_classes(path: str) -> dict[str, str]:
    """Modifies headers in cath20.fa to include homologous superfamily classification.

    Args:
        path (str): Path that contains cath20.fa and cath20_class.txt.

    Returns:
        dict[str, str]: key: CATH ID, value: classification
    """

    classes: dict[str, str] = {}  # key: CATH ID, value: classification
    with open(f'{path}/cath20_class.txt', 'r', encoding='utf8') as file:
        for line in file:
            if line.startswith('#'):
                continue
            line = line.split()
            classes[line[0]] = '.'.join(line[1:5])

    return classes


def modify_fasta(path: str, classes: dict[str, str]):
    """Modifies headers in cath20.fa to include homologous superfamily classification.

    Args:
        path (str): Path that contains cath20.fa.
        classes (dict[str, str]): key: CATH ID, value: classification
    """

    # Read fasta file for CATH IDs and sequences
    seqs: dict[str, str] = {}  # key: CATH ID, value: sequence
    with open(f'{path}/cath20.fa', 'r', encoding='utf8') as file:
        for line in file:
            if line.startswith('>'):
                cath_id = line.split('|')[2].split('/')[0]
                cath_id = cath_id + '|' + classes[cath_id]
            else:
                seqs[cath_id] = line.strip()

    # Rewrite fasta file with modified headers
    with open(f'{path}/cath20.fa', 'w', encoding='utf8') as file:
        for cath_id, seq in seqs.items():
            file.write(f'>{cath_id}\n{seq}\n')


def main():
    """Downloads CATH20 v4.2.0 and fingerprints sequences.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--maxlen', type=int, default=1000, help='max sequence length to embed')
    parser.add_argument('--cpu', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--gpu', type=int, required=False, help='number of gpus to use')
    args = parser.parse_args()

    path = 'bench/cath/data'
    if not os.path.exists(path):
        get_files(path)
        classes = read_classes(path)
        modify_fasta(path, classes)

    # Fingerprint fasta file
    if args.gpu:
        sp.run(['python', 'src/make_db.py', f'--fafile={path}/cath20.fa', f'--dbfile={path}/cath20',
                f'--maxlen={args.maxlen}', '--index', f'--cpu={args.cpu}', f'--gpu={args.gpu}'])
    else:
        sp.run(['python', 'src/make_db.py', f'--fafile={path}/cath20.fa', f'--dbfile={path}/cath20',
            f'--maxlen={args.maxlen}', '--index', f'--cpu={args.cpu}'])


if __name__ == "__main__":
    main()
