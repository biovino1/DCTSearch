"""Queries a database of DCT fingerprints for most similar protein to each query sequence.

__author__ = "Ben Iovino"
__date__ = "3/19/23"
"""

import argparse
from datetime import datetime
import logging
import os
from database import Database
from make_db import embed_cpu, embed_gpu

log_filename = 'data/logs/query_db.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def compare_fprints(qfp: list, dfp: list) -> list:
    """Compares two sets of fingerprints and returns dictionary of highest similarity scores.

    Args:
        qfp (list): List of query fingerprints (vid, fp)
        dfp (list): List of database fingerprints (vid, fp)

    Returns:
        list: List of tuples ((query vid, database vid) similarity) sorted by similarity
    """

    sims = {}
    for i, q in qfp:  # i is query vid, q is query fingerprint
        for j, d in dfp:  # j is database vid, d is database fingerprint
            sim = 1-abs(q-d).sum()/17000

            # If similarity is higher than top 100, add to dictionary
            if len(sims) < 100:
                sims[(i, j)] = sim
            else:
                if sim > min(sims.values()):
                    del sims[min(sims, key=sims.get)]
                    sims[(i, j)] = round(sim, 3)

    sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    return sims


def write_sims(sims: list, query_db: Database, fp_db: Database):
    """Prints the most similar protein to a query sequence.

    Args:
        sims (list): List of tuples ((query vid, database vid) similarity) sorted by similarity
        query_db (Database): Database object connected to query database
        fp_db (Database): Database object connected to SQLite database
    """

    for i, j in sims:
        select = 'SELECT pid, domain FROM fingerprints WHERE vid = ? '
        db_match = fp_db.cur.execute(select, (i[1],)).fetchone()
        query_match = query_db.cur.execute(select, (i[0],)).fetchone()
        logging.info('%s, %s %s, %s %s, %.2f',
                     datetime.now(), query_match[0], query_match[1], db_match[0], db_match[1], j)


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

    # Get each sequence from query db and compare to db
    select = """ SELECT pid FROM sequences """
    query_fps = query_db.cur.execute(select).fetchall()
    for query in query_fps:
        qfps = query_db.load_fprints(query[0])
        sims = compare_fprints(qfps, db_fps)
        write_sims(sims, query_db, fp_db)

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
