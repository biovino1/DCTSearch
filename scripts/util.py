"""Collection of functions used across multiple scripts.

__author__ = "Ben Iovino"
__date__ = "1/23/23"
"""

import numpy as np


def load_fdb(filename: str) -> dict:
    """Returns a dictionary of DCT fingerprints from a npz file.

    Args:
        filename (str): Name of file to parse.

    Returns:
        dict: Nested dictionary where key is protein ID and value is a dictionary of DCT
        fingerprints for each predicted domain.
    """

    fdb = np.load(filename)
    fprints = {}

    # Get all fingerprints associated with each pid
    for i, pid in enumerate(fdb['pids']):
        try:
            idx = (fdb['idx'][i], fdb['idx'][i+1])
        except IndexError:  # Last protein in file
            idx = (fdb['idx'][i], len(fdb['doms']))
        quants = fdb['quants'][idx[0]:idx[1]]
        doms = fdb['doms'][idx[0]:idx[1]]
        fprints[pid] = dict(zip(doms, quants))
    fdb.close()

    return fprints


def load_seqs(filename: str) -> dict:
    """Returns a dictionary of protein sequences from a fasta file.

    Args:
        filename (str): Name of fasta file to parse.
    
    Returns:
        dict: dictionary where key is protein ID and value is the sequence
    """

    seqs = {}
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            if line.startswith('>'):
                pid = line[1:].strip().split()[0]  #ex. '>16vpA00    110129010'
                seqs[pid] = ''
            else:
                seqs[pid] += line.strip()

    return seqs


def yield_seqs(filename: str, maxlen: int):
    """Yields a dictionary of protein sequences from a fasta file with a total number of amino
    acids less than maxlen.

    Args:
        filename (str): Name of fasta file to parse.
        maxlen (int): Maximum length of total sequence to yield.

    Returns:
        dict: dictionary where key is protein ID and value is the sequence
    """

    seqs, curr_len = {}, 0
    file = open(filename, 'r', encoding='utf8')  # less nested than with statement
    for line in file:
        if line.startswith('>'):

            # If dict is too large, yield all but last and reset
            if len(seqs) > 1 and curr_len > maxlen:
                last_seq = seqs.popitem()
                yield seqs
                curr_len = len(last_seq[1])
                seqs = {last_seq[0]: last_seq[1]}

            # Add new sequence to dict
            pid = line[1:].strip().split()[0]
            seqs[pid] = ''
        else:
            curr_len += len(line.strip())
            seqs[pid] += line.strip()

    # Last batch in file may be too large
    if len(seqs) > 1 and curr_len > maxlen:
        last_seq = seqs.popitem()
        yield seqs
    yield {last_seq[0]: last_seq[1]}
    file.close()
