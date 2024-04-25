"""Downloads MMseqs2 benchmark and prepares it for benchmarking.

__author__ = "Ben Iovino"
__date__ = "4/24/24"
"""

import argparse
import os
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
import subprocess as sp
from urllib.request import urlretrieve
from zipfile import ZipFile


def get_files(path: str):
    """Downloads files from MMseqs2 server and SCOPe. MMseqs2 data can take some time to download.

    Args:
        path (str): Path to download files to.
    """

    if not os.path.exists(path):
        os.mkdir(path)

    # MMseqs2 benchmark
    url = 'https://wwwuser.gwdg.de/~compbiol/mmseqs2/mmseqs2-benchmark.tar.gz'
    print('Downloading MMseqs2 dataset...')
    urlretrieve(url, f'{path}/mmseqs2-benchmark.tar.gz')
    with ZipFile(f'{path}/mmseqs2-benchmark.tar.gz', 'r') as zip_ref:
        zip_ref.extractall(f'{path}/')

    # Classification file
    url = 'https://scop.berkeley.edu/downloads/parse/dir.cla.scop.1.75.txt'
    print('Downloading SCOP classification...')
    urlretrieve(url, f'{path}/scop_class.txt')


def get_unshuffled(path: str):
    """Get unshuffled query sequences from the target database.

    Args:
        path (str): Path to query/targetdb
    """

    # Read each sequence in query file
    queries = []
    with open(f'{path}/mmseqs2-benchmark-pub/db/query.fasta', 'r') as file:
        for line in file:
            if line.startswith('>'):
                pid = line.split()[0][1:].split('_')[0]
                queries.append(pid)

    # Get sequence of each query from target database
    seqs = {}
    with open(f'{path}/mmseqs2-benchmark-pub/db/targetannotation.fasta', 'r') as file:
        pid, desc = '', ''
        for line in file:
            if line.startswith('>'):
                line = line.split()
                pid = line[0].split('_')[1]
                desc = ' '.join(line[1:])
                continue
            if pid in queries:
                seqs[pid] = (desc, line.strip())
    
    # Write sequences to file
    with open(f'{path}/mmseqs2-benchmark-pub/db/uquery.fasta', 'w') as file:
        for pid, (desc, seq) in seqs.items():
            file.write(f'>{pid} {desc}\n{seq}\n')


def main():
    """Downloads MMseqs2 benchmark and fingerprints sequences. Fingerprinting this database can
    take several days so it is recommended to run this script on a server with GPU support.

    SCOP(e) concise classification string (sccs)
    -------------------------------
    Character 1: Class
    Character 2: Fold
    Character 3: Superfamily
    Character 4: Family
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--maxlen', type=int, default=1000, help='max sequence length to embed')
    parser.add_argument('--cpu', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--gpu', type=int, required=False, help='number of gpus to use')
    args = parser.parse_args()

    path = 'bench/scop/data'
    if not os.path.exists(path):
        get_files(path)
        get_unshuffled(path)

    # Fingerprint fasta file
    db = 'mmseqs2-benchmark-pub/db/targetannotation.fasta'
    if args.gpu:
        sp.run(['python', 'src/make_db.py', f'--fafile={path}/{db}', f'--dbfile={path}/mmseqs2.db',
                f'--maxlen={args.maxlen}', f'--cpu={args.cpu}', f'--gpu={args.gpu}'])
    else:
        sp.run(['python', 'src/make_db.py', f'--fafile={path}/{db}', f'--dbfile={path}/mmseqs2.db',
            f'--maxlen={args.maxlen}', f'--cpu={args.cpu}'])


if __name__ == "__main__":
    main()
