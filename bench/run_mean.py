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
import re
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
import numpy as np
from io import BytesIO
import torch
from transformers import T5EncoderModel, T5Tokenizer
from src.database import Database
from bench.run_dct import get_queries, search_db


def prot_t5_embed(model: T5EncoderModel, tokenizer: T5Tokenizer, seq: str, device: str) -> np.ndarray:
    """Returns the mean embedding of a protein sequence using ProtT5_XL model.

    Args:
        model (T5EncoderModel): ProtT5_XL model
        tokenizer (T5Tokenizer): ProtT5_XL tokenizer
        seq (str): Protein sequence
        device (str): Device to load model on

    Returns:
        np.ndarray: Mean embedding of protein sequence (1024, 1)
    """

    seq = re.sub(r"[UZOB]", "X", seq)
    seq = [' '.join([*seq])]

    # Tokenize, encode, and load sequence
    ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)  # pylint: disable=E1101
    attention_mask = torch.tensor(ids['attention_mask']).to(device)  # pylint: disable=E1101

    # Extract final layer of model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    emb = outputs.last_hidden_state.cpu().numpy()

    # Remove padding and special tokens
    features = []
    for seq_num in range(len(emb)):  # pylint: disable=C0200
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = emb[seq_num][:seq_len-1]
        features.append(seq_emd)
    emb = np.mean(features[0], axis=0)

    return emb


def embed_seqs(args: argparse.Namespace, db: Database, vid: int):
    """Embeds sequences with final layer of model and averages embeddings. In terms of the
    database, this mean embedding is the only fingerprint stored.

    Args:
        args (argparse.Namespace): Command line arguments
        db (Database): Database object connected to SQLite database
        vid (int): Fingerprint count tracker
    """

    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')
    model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')
    if args.gpu:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    model.to(device)

    # Embed with prot_t5_xl model
    for i, seqs in enumerate(db.yield_seqs(1, 1)):
        pid, seq = seqs[0][0], seqs[0][1]
        emb = prot_t5_embed(model, tokenizer, seq, device)
        
        # Add to database
        emb_bytes = BytesIO()
        np.save(emb_bytes, emb)
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
    embs = np.array(embs)
    dim = embs.shape[1]
    
    # Create index
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    faiss.write_index(index, f'{path}/mean.index')


def main():
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=str, help='benchmark to test')
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
    search_db(f'{path}/mean.db', queries, args.khits)


if __name__ == "__main__":
    main()
