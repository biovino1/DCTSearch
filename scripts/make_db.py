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
from multiprocessing import Pool
from embedding import Model, Batch
from fingerprint import Fingerprint
from database import Database
from util import yield_seqs

log_filename = 'data/logs/make_db.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def queue_cpu(fp: Fingerprint, args: argparse.Namespace) -> Fingerprint:
    """Predicts domains and quantizes embeddings on cpu. Returns Fingerprint.

    Args:
        queue (mp.Queue): Queue of embeddings to fingerprint
        args (argparse.Namespace): Command line arguments

    Returns:
        Fingerprint: Fingerprint object with quantized domains.
    """

    fp.reccut(2.6)
    fp.quantize(args.quantdims)
    logging.info(f'{datetime.datetime.now()} Fingerprinted {fp.pid}')

    return fp


def fprint_cpu(batch: list, args: argparse.Namespace) -> list:
    """Puts batches of Fingerprints in queue to be quantized on cpu(s). Returns a list of
    Fingerprint objects with quantized domains.

    Args:
        batch (list): List of fingerprints to quantize
        args (argparse.Namespace): Command line arguments
    """

    with Pool(processes=args.cpu) as pool:
        results = pool.starmap(queue_cpu, [(fp, args) for fp in batch])
    
    return results


def queue_gpu(rank: int, queue: mp.Queue, args: argparse.Namespace, db: list):
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

        # If queue is full, start multiprocess fingerprinting
        if len(cpu_queue) >= args.cpu/args.gpu:
            fps = fprint_cpu(cpu_queue, args)
            for fp in fps:
                db.append(fp)
            cpu_queue = []


def embed_gpu(args: argparse.Namespace):
    """Puts batches of sequences in queue to be embedded on gpu(s).

    Args:
        args (argparse.Namespace): Command line arguments
    """

    mp_queue = mp.Queue()
    shared_db = mp.Manager().list()  # shared list for fingerprint objects
    processes = []
    for rank in range(args.gpu):
        proc = mp.Process(target=queue_gpu, args=(rank, mp_queue, args, shared_db))
        proc.start()
        processes.append(proc)
    for seqs in yield_seqs(args.fafile, args.maxlen):
        mp_queue.put(seqs)
    for _ in range(args.gpu):  # send None to each process to signal end of queue
        mp_queue.put(None)
    for proc in processes:
        proc.join()

    # Add fingerprints to database and save
    database = Database()
    for fp in shared_db:
        database.add_fprint(fp)
    database.save_db(args.dbfile)


def embed_cpu(args: argparse.Namespace):
    """Embeds sequences on cpu.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """

    model = Model('esm2', 't30')
    device = torch.device('cpu')
    model.to_device(device)
    
    # Embed one sequence at a time (batching on cpu is very slow)
    cpu_queue, db = [], Database()
    for seqs in yield_seqs(args.fafile, 1):
        batch = Batch(seqs, model, device)
        batch.embed_batch(args.layers, args.maxlen)

        # Create Fingerprint object and add to queue
        for emb in batch.embeds:
            fp = Fingerprint(pid=emb.pid, seq=emb.seq, embed=emb.embed, contacts=emb.contacts)
            cpu_queue.append(fp)

        # If queue is full, start multiprocess fingerprinting
        if len(cpu_queue) >= args.cpu/args.gpu:
            fps = fprint_cpu(cpu_queue, args)
            for fp in fps:  # add each fp to db
                db.add_fprint(fp)
            cpu_queue = []

    db.save_db(args.dbfile)


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
