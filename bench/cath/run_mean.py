"""Queries CATH20 database with sequences from cath20_queries.txt using the mean embedding method.
Has to first embed sequences and then create an index to search, like in run_dct.py, except there
is no fingerprinting.

__author__ = "Ben Iovino"
__date__ = "5/10/24"
"""

import argparse
import os
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
import numpy as np
from io import BytesIO
import torch
from src.embedding import Model, Embedding
from src.database import Database


def embed_seqs(args: argparse.Namespace, db: Database, vid: int):
    """Embeds sequences with final layer of model and averages embeddings. In terms of the
    database, this mean embedding is the only fingerprint stored.

    Args:
        args (argparse.Namespace): Command line arguments
        db (Database): Database object connected to SQLite database
        vid (int): Fingerprint count tracker
    """

    model = Model('esm2', 't30')
    if args.gpu:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    model.to_device(device)

    # Multiprocessing was not built for embedding only
    for i, seqs in enumerate(db.yield_seqs(1, 1)):
        emb = Embedding(pid=seqs[0][0], seq=seqs[0][1])
        emb.embed_seq(model, device, [30], args.maxlen)
        mean = np.mean(emb.embed[30], axis=0)
        
        # Add to database
        emb_bytes = BytesIO()
        np.save(emb_bytes, mean)
        update = """ UPDATE sequences SET fpcount = ? WHERE pid = ? """
        db.cur.execute(update, (1, emb.pid))
        insert = """ INSERT INTO fingerprints(vid, domain, fingerprint, pid)
            VALUES(?, ?, ?, ?) """
        db.cur.execute(insert, (i+vid, f'1-{len(emb.seq)}', emb_bytes.getvalue(), emb.pid))
        db.conn.commit()
    
    db.close()


def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1, help='Number of cpu cores to use for knn')
    parser.add_argument('--gpu', type=int, default=False, help='GPU to load model on')
    parser.add_argument('--maxlen', type=int, default=1000, help='Max sequence length to embed')
    args = parser.parse_args()

    # Embed sequences
    path = 'bench/cath/data'
    db = Database(f'{path}/mean.db', f'{path}/cath20.fa')
    vid = db.get_last_vid()
    embed_seqs(args, db, vid)


if __name__ == "__main__":
    main()
