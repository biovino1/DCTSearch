"""Defines the Database class, which is used to manage the SQLite database.

__author__ = "Ben Iovino"
__date__ = "3/18/24"
"""

import sqlite3
import os


class Database:
    """Initializes .db file and manages database operations.

    Attributes:
        path (str): Path to database file.
        conn (sqlite3.Connection): Connection to database.
        cur (sqlite3.Cursor): Cursor for database.
    """

    def __init__(self, path: str):
        """Initializes database.
        
        Args:
            path (str): Path to database file.
        """

        # Check if fasta
        self.path = path
        if '.fa' in self.path:
            seqs = self.read_fasta()
            self.init_db(seqs)


    def read_fasta(self) -> dict:
        """Returns a dictionary of sequences from a fasta file.

        Returns:
            dict: Dictionary of sequences.
        """

        seqs = {}
        with open(self.path, 'r', encoding='utf8') as f:
            for line in f:
                if line.startswith('>'):
                    pid = line.strip()
                    seqs[pid] = ''
                else:
                    seqs[pid] += line.strip()
        
        return seqs
    

    def init_db(self, seqs: dict):
        """Creates a new database file and fills table with sequences.

        Args:
            seqs (dict): Dictionary of sequences.
        """

        filename = os.path.splitext(self.path)[0]
        self.conn = sqlite3.connect(f'{filename}.db')
        self.cur = self.conn.cursor()

        # Create table
        table = """CREATE TABLE sequences (
                id text PRIMARY KEY,
                sequence text NOT NULL,
                length integer NOT NULL,
                domains text,
                fingerprint blob
                ); """
        self.cur.execute(table)

        # Insert sequences
        insert = """ INSERT INTO sequences(id, sequence, length, domains, fingerprint)
            VALUES(?, ?, ?, ?, ?) """
        for pid, seq in seqs.items():
            self.cur.execute(insert, (pid, seq, len(seq), '', None))
        self.conn.commit()
