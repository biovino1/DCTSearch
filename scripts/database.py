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
        """Initializes database file and cursor. Will create new database if fasta file is given.
        
        Args:
            path (str): Path to database file.
        """

        self.path = path
        if '.fa' in self.path:
            seqs = self.read_fasta()
            self.init_db(seqs)
        else:
            self.conn = sqlite3.connect(self.path)
            self.cur = self.conn.cursor()

    
    def close(self):
        """Closes the database connection.
        """

        self.conn.close()


    def read_fasta(self) -> dict:
        """Returns a dictionary of sequences from a fasta file.

        Returns:
            dict: Dictionary of sequences.
        """

        seqs = {}
        with open(self.path, 'r', encoding='utf8') as f:
            for line in f:
                if line.startswith('>'):
                    pid = line.strip().split()[0]
                    seqs[pid] = ''
                else:
                    seqs[pid] += line.strip()
        seqs = {k: v for k, v in sorted(seqs.items(), key=lambda item: len(item[1]))}
        
        return seqs
    

    def init_db(self, seqs: dict):
        """Creates a new database file and fills table with sequences.

        Args:
            seqs (dict): Dictionary of sequences.
        """

        self.path = os.path.splitext(self.path)[0]
        self.conn = sqlite3.connect(f'{self.path}.db')
        self.cur = self.conn.cursor()

        # Create table
        table = """CREATE TABLE sequences (
                pid text PRIMARY KEY,
                sequence text NOT NULL,
                length integer NOT NULL,
                domains text,
                fingerprint blob
                ); """
        self.cur.execute(table)

        # Insert sequences
        insert = """ INSERT INTO sequences(pid, sequence, length, domains, fingerprint)
            VALUES(?, ?, ?, ?, ?) """
        for pid, seq in seqs.items():
            self.cur.execute(insert, (pid, seq, len(seq), '', None))
        self.conn.commit()


    def yield_seqs(self, maxlen: int, dim1: int = 3, dim2: int = 80):
        """Yields sequences from the database.

        Args:
            maxlen (int): Maximum length of total sequence to yield
            dim1 (int): First dimension of quantization.
            dim2 (int): Second dimension of quantization.

        Yields:
            list: List of tuples where elements are protein ID and sequence.
        """

        seqs, curr_len, min_size = [], 0, dim1*dim2
        select = """ SELECT pid, sequence, length FROM sequences WHERE domains == ''"""
        for row in self.cur.execute(select):
            pid, seq, length = row
            if (length-2) * dim2 < min_size:  # Short sequences can't be quantized
                continue
            curr_len += length

            # If list is too large, yield all but last and reset
            if len(seqs) > 1 and curr_len > maxlen:
                last_seq = seqs.pop()
                yield seqs
                curr_len = len(last_seq[1])
                seqs = [(last_seq[0], last_seq[1])]
            seqs.append((pid, seq))  # New sequence

        # Last batch in file may be too large
        if len(seqs) > 1 and curr_len > maxlen:
            last_seq = seqs.pop()
            yield seqs
        yield [(last_seq[0], last_seq[1])]
