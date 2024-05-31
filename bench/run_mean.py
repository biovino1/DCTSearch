"""Queries CATH20 database with sequences from cath20_queries.txt using the mean embedding method.
Has to first embed sequences and then create an index to search, like in run_dct.py, except there
is no fingerprinting.

__author__ = "Ben Iovino"
__date__ = "5/10/24"
"""

import argparse
import faiss
import logging
import os
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
import numpy as np
from io import BytesIO
import torch
from src.embedding import Model, Embedding
from src.database import Database
from bench.run_dct import get_queries, search_db


def embed_seqs(args: argparse.Namespace, db: Database, vid: int):
    """Embeds sequences with final layer of model and averages embeddings. In terms of the
    database, this mean embedding is the only fingerprint stored.

    Args:
        args (argparse.Namespace): Command line arguments
        db (Database): Database object connected to SQLite database
        vid (int): Fingerprint count tracker
    """

    model = Model()
    device = torch.device(f'cuda:{args.gpu}' if args.gpu else 'cpu')
    model.to_device(device)

    # Embed and calculate mean embedding of last hidden layer (24)
    for i, seqs in enumerate(db.yield_seqs(1, 1)):
        pid, seq = seqs[0][0], seqs[0][1]
        emb = Embedding(pid, seq)
        emb.embed_seq(model, device, [24], args.maxlen)
        mean = np.mean(emb.embed[24], axis=0)
        
        # Add 'fingerprint' (mean embedding) to database
        emb_bytes = BytesIO()
        np.save(emb_bytes, mean)
        update = """ UPDATE sequences SET fpcount = ? WHERE pid = ? """
        db.cur.execute(update, (1, pid))
        insert = """ INSERT INTO fingerprints(vid, domain, fingerprint, pid)
            VALUES(?, ?, ?, ?) """
        db.cur.execute(insert, (i+vid, f'1-{len(seq)}', emb_bytes.getvalue(), pid))
        db.conn.commit()


def load_embs(db: Database) -> list[np.ndarray]:
    """Returns a list of embeddings from the database.
    """

    embs = []
    select = """ SELECT fingerprint FROM fingerprints """
    for row in db.cur.execute(select):
        emb_bytes = BytesIO(row[0])
        embs.append(np.load(emb_bytes))

    return embs


def create_index(path: str, db: Database):
    """Creates index of fingerprints for fast querying with FAISS.

    Args:
        path (str): Path to save index
        db (Database): Database object connected to SQLite database
    """

    # Load fingerprints as a flat numpy array
    embs = load_embs(db)
    embs = np.array(embs, dtype=np.float32)

    # Normalize for cosine similarity (as done in knnProtT5 paper)
    for i in range(embs.shape[0]):
        fp = np.expand_dims(embs[i], axis=0)
        faiss.normalize_L2(fp)
        embs[i] = fp
    
    # Create index
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, f'{path}/mean.index')


def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=str, required=True, help='benchmark to test')
    parser.add_argument('--cpu', type=int, default=1, help='Number of cpu cores to use for knn')
    parser.add_argument('--gpu', type=int, default=False, help='GPU to load model on')
    parser.add_argument('--khits', type=int, default=14433, help='Number of nearest neighbors to find')
    parser.add_argument('--maxlen', type=int, default=1000, help='Max sequence length to embed')
    args = parser.parse_args()

    # Determine query and db files
    if args.bench == 'cath':
        path = 'bench/cath/data'
        query = 'cath20_queries.fa'
        db = 'cath20.fa'
    elif args.bench == 'pfam':
        path = 'bench/pfam/data'
        query = 'pfam20.fa'
        db = 'pfam20.fa'
    elif args.bench == 'scop':
        path = 'bench/scop/data'
        query = 'query.fa'
        db = 'target.fa'

    # Embed sequences
    db = Database(f'{path}/mean.db', f'{path}/{db}')
    vid = db.get_last_vid()
    embed_seqs(args, db, vid)
    create_index(path, db)

    # Get queries and search against database
    queries = get_queries(f'{path}/{query}')
    logging.basicConfig(level=logging.INFO, filename=f'{path}/results_mean.txt',
                         filemode='w', format='%(message)s')
    search_db(f'{path}/mean.db', queries, args.khits, 'sim')


if __name__ == "__main__":
    main()
