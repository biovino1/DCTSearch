"""Tests sequence length and RAM usage for fingerprinting sequences.

__author__ = "Ben Iovino"
__date__ = "5/14/24"
"""

import argparse
import os
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
import subprocess as sp
from random import choice


def generate_sequence(path: str, length: int):
    """Writes a protein sequence of the given length to test.fa

    Args:
        path (str): Path to save test sequence.
        length (int): Length of sequence to generate.
    """

    if not os.path.exists(path):
        os.mkdir(path)

    # Generate and write
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    sequence = ''.join(choice(amino_acids) for _ in range(length))
    with open(f'{path}/ram_test.fa', 'w') as file:
        file.write(f'>test\n{sequence}')


def measure_ram(path: str, args: argparse.Namespace, length: int) -> sp.CompletedProcess:
    """Returns output from make_db.py for a given sequence length.

    Args:
        path (str): Path to save test database.
        args (argparse.Namespace): Command line arguments.
        length (int): Maximum length of subsequences to embed.

    Returns:
        sp.CompletedProcess: Output from make_db.py.
    """

    if args.gpu:
        result = sp.run(['python', 'src/make_db.py', f'--fafile={path}/ram_test.fa',
                        f'--dbfile={path}/ram_test', f'--maxlen={length}', f'--cpu={args.cpu}',
                        f'--gpu={args.gpu}', '--index'], capture_output=True, text=True)
    else:
        result = sp.run(['python', 'src/make_db.py', f'--fafile={path}/ram_test.fa',
                        f'--dbfile={path}/ram_test', f'--maxlen={length}', f'--cpu={args.cpu}',
                        '--index'], capture_output=True, text=True)

    return result


def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--textmax', type=int, default=2000, help='max sequence length to embed')
    parser.add_argument('--cpu', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--gpu', type=int, required=False, help='number of gpus to use')
    args = parser.parse_args()

    # Steps of 100, starting at 300 until --testmax (default 2000)
    path = 'utils/data'
    generate_sequence(path, args.textmax)
    for i in range(3, args.textmax//100+1):
        length = i*100
        print(f'Testing subsequence length of {length}...')
        result = measure_ram(path, args, length)

        # Check if database was created
        if result.stdout:
            print('Passed!\n')
        else:  # If not, print max sequence length and exit
            print(f'Failed! Your max sequence length is {length-100}\n')
            os.remove(f'{path}/ram_test.db')
            break
        os.remove(f'{path}/ram_test.db')


if __name__ == "__main__":
    main()
