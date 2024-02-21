"""Makes a database of DCT fingerprints from a fasta file of protein sequences.

__author__ = "Ben Iovino"
__date__ = "12/18/23"
"""

import argparse
import datetime
import time
import logging
import os
import torch
import torch.multiprocessing as mp
from embedding import Model, Embedding
from fingerprint import Fingerprint
from database import Database
from util import load_seqs

log_filename = 'data/logs/make_db.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def get_fprint(seq: tuple, model: Model, device: str, args: argparse.Namespace):
    """Returns a Fingerprint object for a given sequence.

    Args:
        seqs (tuple): Tuple of protein ID and sequence
        model (Model): Object containing encoder and tokenzier
        device (str): gpu/cpu
        args (argparse.Namespace): Command line arguments

    Returns:
        Fingerprint: Fingerprint object containing DCT fingerprints
    """


    # Embed and fingerprint sequence
    pid, seq = seq[0], seq[1]
    nowt = time.time()
    emb = Embedding(pid=pid, seq=seq)
    emb.embed_seq(model, device, args.layers, args.maxlen)
    fprint = Fingerprint(pid=pid, seq=seq, embed=emb.embed, contacts=emb.contacts)
    fprint.reccut(2.6)
    fprint.quantize(args.quantdims)
    endt = time.time()
    logging.info('%s Sequence: %s, Length: %s, Time: %s',
                  datetime.datetime.now(), pid, len(seq), endt-nowt)

    return fprint


def queue_seq(rank: int, queue: mp.Queue, args: argparse.Namespace):
    """Moves through queue of sequences to fingerprint and add to database.

    :param rank: GPU to load model on
    :param queue: queue of families to embed and transform
    :param args: explained in main()
    """

    # Load tokenizer and encoder
    device = torch.device(f'cuda:{rank}')  #pylint: disable=E1101
    model = Model('esm2', 't30')  # pLM encoder and tokenizer
    model.to_device(device)

    # Fingerprint each sequence and add to database
    db = Database()
    while True:
        seq = queue.get()
        if seq is None:
            break
        fprint = get_fprint(seq, model, device, args)
        db.add_fprint(fprint)

    db.save_db(args.dbfile)


def embed_gpu(seqs: dict, args: argparse.Namespace):
    """Embeds sequences on gpu(s).

    Args:
        seqs (dict): Dictionary of protein sequences
        args (argparse.Namespace): Command line arguments
    """

    mp_queue = mp.Queue()
    processes = []
    for rank in args.gpulist:
        proc = mp.Process(target=queue_seq, args=(args.gpulist[rank], mp_queue, args))
        proc.start()
        processes.append(proc)
    for seq in seqs.items():  # seq is a tuple of protein ID and sequence
        mp_queue.put(seq)
    for _ in range(len(args.gpulist)):  # send None to each process to signal end of queue
        mp_queue.put(None)
    for proc in processes:
        proc.join()


def main():
    """Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--fafile', type=str, required=True, help='fasta file to embed')
    parser.add_argument('--dbfile', type=str, required=True, help='db file to write to')
    parser.add_argument('--maxlen', type=int, default=1000, help='max sequence length to embed')
    parser.add_argument('--gpu', type=bool, default=False, help='gpu (True) or cpu (False)')
    parser.add_argument('--gpulist', type=int, nargs='+', default=[0], help='list of cuda devices')
    parser.add_argument('--layers', type=int, nargs='+', default=[15, 21], help='embedding layers')
    parser.add_argument('--quantdims', type=int, nargs='+', default=[3, 80, 3, 80],
                         help='quantization dimensions, each pair of dimensions quantizes a layer')
    args = parser.parse_args()

    seqs = load_seqs(args.fafile)  # all sequences in memory
    if args.gpu:
        embed_gpu(seqs, args)
    else:
        model = Model('esm2', 't30')
        device = torch.device('cpu')
        model.to_device(device)
        db = Database()
        for seq in seqs.items():
            fprint = get_fprint(seq, model, device, args)
            db.add_fprint(fprint)
        db.save_db(args.dbfile)


if __name__ == '__main__':
    main()
