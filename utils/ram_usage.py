"""Tests sequence length and RAM usage for fingerprinting sequences.

__author__ = "Ben Iovino"
__date__ = "5/14/24"
"""

import argparse
import os
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
import subprocess as sp


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
        result = sp.run(['python', 'src/make_db.py', f'--fafile={path}/test.fa',
                        f'--dbfile={path}/test', f'--maxlen={length}', f'--cpu={args.cpu}',
                        f'--gpu={args.gpu}', '--index'], capture_output=True, text=True)
    else:
        result = sp.run(['python', 'src/make_db.py', f'--fafile={path}/test.fa',
                        f'--dbfile={path}/test', f'--maxlen={length}', f'--cpu={args.cpu}',
                        '--index'], capture_output=True, text=True)

    return result


def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--gpu', type=int, required=False, help='number of gpus to use')
    args = parser.parse_args()

    # Range of max subsequence length is [3, 20]
    path = 'utils/data'
    for i in range(3, 21):
        length = i*100
        print(f'Testing sequence length of {length}...')
        result = measure_ram(path, args, length)
        print(result)
            
        # Check if database was created
        if result.stdout:
            print('Passed!\n')
        else:
            print(f'Failed! Your max sequence length is {length-100}\n')
            os.remove(f'{path}/test.fa')
            break
        os.remove(f'{path}/test.fa')


if __name__ == "__main__":
    main()
