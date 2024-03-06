"""Makes a database of DCT fingerprints from a fasta file of protein sequences.

__author__ = "Ben Iovino"
__date__ = "12/18/23"
"""

import argparse
import datetime
import logging
import os
import torch
import torch.multiprocessing as mp
from embedding import Model, Batch
from fingerprint import Fingerprint
from database import Database
from util import yield_seqs

log_filename = 'data/logs/make_db.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def queue_cpu(cpu_queue: mp.Queue, args: argparse.Namespace):
    """Moves through queue of Fingerprints and quantizes their embeddings.

    Args:
        queue (mp.Queue): Queue of embeddings to fingerprint
        args (argparse.Namespace): Command line arguments
    """

    while True:
        fp = cpu_queue.get()
        if fp is None:
            break
        fp.reccut(2.6)
        fp.quantize(args.quantdims)
        logging.info(f'{datetime.datetime.now()} Fingerprinted {fp.pid}')


def fprint_cpu(batch: list, args: argparse.Namespace):
    """Puts batches of Fingerprints in queue to be quantized on cpu(s).

    Args:
        batch (list): List of fingerprints to quantize
        args (argparse.Namespace): Command line arguments
    """

    cpu_queue = mp.Queue()
    processes = []
    for _ in batch:
        proc = mp.Process(target=queue_cpu, args=(cpu_queue, args))
        proc.start()
        processes.append(proc)
    for fp in batch:
        cpu_queue.put(fp)
    for _ in range(len(batch)):  # send None to each process to signal end of queue
        cpu_queue.put(None)
    for proc in processes:
        proc.join()


def queue_gpu(rank: int, queue: mp.Queue, args: argparse.Namespace):
    """Moves through queue of sequences to fingerprint and add to database.

    :param rank: GPU to load model on
    :param queue: queue of families to embed and transform
    :param args: explained in main()
    """

    # Load tokenizer and encoder
    device = torch.device(f'cuda:{rank}')
    model = Model('esm2', 't30')  # pLM encoder and tokenizer
    model.to_device(device)

    # Embed batches of sequences
    cpu_queue = []
    while True:
        seqs = queue.get()
        if seqs is None:
            break
        batch = Batch(seqs, model, device)
        batch.embed_batch(args.layers, args.maxlen)

        # Create Fingerprint object and add to queue
        for emb in batch.embeds:
            fp = Fingerprint(pid=emb.pid, seq=emb.seq, embed=emb.embed, contacts=emb.contacts)
            cpu_queue.append(fp)
        if len(cpu_queue) >= args.cpu:
            fprint_cpu(cpu_queue, args)
            cpu_queue = []


def embed_gpu(args: argparse.Namespace):
    """Puts batches of sequences in queue to be embedded on gpu(s).

    Args:
        args (argparse.Namespace): Command line arguments
    """

    mp_queue = mp.Queue()
    processes = []
    for rank in range(args.gpu):
        proc = mp.Process(target=queue_gpu, args=(rank, mp_queue, args))
        proc.start()
        processes.append(proc)
    for seqs in yield_seqs(args.fafile, args.maxlen):
        mp_queue.put(seqs)
    for _ in range(args.gpu):  # send None to each process to signal end of queue
        mp_queue.put(None)
    for proc in processes:
        proc.join()


def embed_cpu(args: argparse.Namespace):
    """Embeds sequences on cpu.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """

    model = Model('esm2', 't30')
    device = torch.device('cpu')
    model.to_device(device)
    
    # Embed one sequence at a time (batching on cpu is very slow)
    cpu_queue = []
    for seqs in yield_seqs(args.fafile, 1):
        batch = Batch(seqs, model, device)
        batch.embed_batch(args.layers, args.maxlen)

        # Create Fingerprint object and add to queue
        for emb in batch.embeds:
            fp = Fingerprint(pid=emb.pid, seq=emb.seq, embed=emb.embed, contacts=emb.contacts)
            cpu_queue.append(fp)
        if len(cpu_queue) >= args.cpu:
            fprint_cpu(cpu_queue, args)
            cpu_queue = []


def main():
    """Sequences from a fasta file of protein sequences go through two processes:

    1. Embedding: sequences are embedded using ESM2, recommended to be done on GPU.
    2. Fingerprinting: domains are cut and quantized, performed on CPU.

    The resulting fingerprints are then written to a database file. Increasing the number of
    both available GPUs and CPUs will speed up the process. Make sure --maxlen is set to a
    value that is appropriate for the available memory.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--fafile', type=str, required=True, help='fasta file to embed')
    parser.add_argument('--dbfile', type=str, required=True, help='db file to write to')
    parser.add_argument('--maxlen', type=int, default=1000, help='max sequence length to embed')
    parser.add_argument('--cpu', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--gpu', type=int, required=False, help='number of gpus to use')
    parser.add_argument('--layers', type=int, nargs='+', default=[15, 21], help='embedding layers')
    parser.add_argument('--quantdims', type=int, nargs='+', default=[3, 80, 3, 80],
                         help='quantization dimensions, each pair of dimensions quantizes a layer')
    args = parser.parse_args()

    if args.gpu:
        embed_gpu(args)
    else:
        embed_cpu(args)


if __name__ == '__main__':
    main()
