"""Queries a database of DCT fingerprints for most similar protein to each query sequence.

__author__ = "Ben Iovino"
__date__ = "3/19/23"
"""

import argparse
import os
from database import Database
from make_db import embed_cpu, embed_gpu


def compare_fprints(qfp: list, dfp: list) -> tuple:
    """Compares two sets of fingerprints and returns the highest similarity score between
    all pairs.

    Args:
        qfp (list): List of query fingerprints (vid, fp)
        dfp (list): List of database fingerprints (vid, fp)

    Returns:
        tuple (float, tuple): max sim, (query dom index, database dom index)
    """

    max_sim, vids = 0, ()
    for i, q in qfp:  # i is query vid, q is query fingerprint
        for j, d in dfp:  # j is database vid, d is database fingerprint
            sim = 1-abs(q-d).sum()/17000
            if sim > max_sim:
                max_sim = sim
                vids = (i, j)
    return max_sim, vids


def search_db(query_db: str, fp_db: str):
    """Searches a database of DCT fingerprints for the most similar protein to each query sequence.

    Args:
        query_db (str): Name of query database
        fp_db (str): Name of fingerprint database
    """

    # Connect to databases
    query_db = Database(query_db)
    query_db.db_info()
    fp_db = Database(fp_db)
    fp_db.db_info()
    db_fps = fp_db.load_fprints()

    # Get each sequence from query db
    select = """ SELECT pid FROM sequences """
    query_fps = query_db.cur.execute(select).fetchall()
    for query in query_fps:
        qfps = query_db.load_fprints(query[0])
        sim, vids = compare_fprints(qfps, db_fps)
    
        # Print matches
        select = 'SELECT pid, domain FROM fingerprints WHERE vid = ? '
        db_match = fp_db.cur.execute(select, (vids[1],)).fetchone()
        query_match = query_db.cur.execute(select, (vids[0],)).fetchone()
        print(f'Query: {query_match[0]}, {query_match[1]}\n'
              f'Match: {db_match[0]}, {db_match[1]}\n'
              f'Similarity: {sim:.2f}\n')

    query_db.close()
    fp_db.close()


def main():
    """Processes sequences same as make_db.py and queries --dbfile for most similar sequence for
    each sequence in the query database.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True, help='can be .fa or .db file')
    parser.add_argument('--dbfile', type=str, required=True, help='fingerprint database (.db)')
    parser.add_argument('--maxlen', type=int, default=1000, help='max sequence length to embed')
    parser.add_argument('--cpu', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--gpu', type=int, default=False, help='number of gpus to use')
    args = parser.parse_args()

    # Embed query sequences
    query_db = os.path.splitext(args.query)[0]
    db = Database(query_db, args.query)
    if args.gpu:
        embed_gpu(args, db)
    else:
        embed_cpu(args, db)

    # Query database for most similar sequence
    search_db(query_db, args.dbfile)
   

if __name__ == '__main__':
    main()
