"""Collection of functions used across multiple scripts.

__author__ = "Ben Iovino"
__date__ = "1/23/23"
"""

import numpy as np


def yield_seqs(filename: str, maxlen: int, dim1: int = 3, dim2: int = 80):
    """Yields a list of protein sequences from a fasta file with a total number of amino acids
    less than maxlen.

    Args:
        filename (str): Name of fasta file to parse.
        maxlen (int): Maximum length of total sequence to yield.
        dim1 (int): First dimension of quantization.
        dim2 (int): Second dimension of quantization.

    Yields:
        list: List of tuples where first element is protein ID and second element is the sequence.
    """

    seqs, curr_len, min_size = [], 0, dim1*dim2
    file = open(filename, 'r', encoding='utf8')  # less nested than with statement
    for line in file:
        if line.startswith('>'):

            # If dict is too large, yield all but last and reset
            if len(seqs) > 1 and curr_len > maxlen:
                last_seq = seqs.pop()
                yield seqs
                curr_len = len(last_seq[1])
                seqs = [(last_seq[0], last_seq[1])]

            # Add new sequence to dict
            pid = line[1:].strip().split()[0]
            seqs.append((pid, ''))
        else:
            curr_len += len(line.strip())
            seqs[-1] = (seqs[-1][0], seqs[-1][1] + line.strip())
            if (len(seqs[-1][1])-2) * dim2 < (min_size):
                seqs.pop() # Ignore VERY short seqs, depends on quant dims

    # Last batch in file may be too large
    if len(seqs) > 1 and curr_len > maxlen:
        last_seq = seqs.pop()
        yield seqs
    yield [(last_seq[0], last_seq[1])]
    file.close()
