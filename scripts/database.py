"""Defines the Database class, which is used to manage the SQLite database.

__author__ = "Ben Iovino"
__date__ = "3/18/24"
"""

import sqlite3
import os
import numpy as np
from io import BytesIO
from fingerprint import Fingerprint


class Database:
    """Initializes .db file and manages database operations.

    Attributes:
        path (str): Path to database file.
        conn (sqlite3.Connection): Connection to database.
        cur (sqlite3.Cursor): Cursor for database.
    """


    def __init__(self, dbfile: str, fafile: str = None):
        """Initializes database file and cursor. Will create new database if fasta file is given.
        
        Args:
            dbfile (str): Path to database file.
            fafile (str): Path to fasta file.
        """

        # Check for .db file first
        if os.path.exists(f'{os.path.splitext(dbfile)[0]}.db'):
            print(f'Opening existing database: {os.path.splitext(dbfile)[0]}.db')
            self.path = f'{os.path.splitext(dbfile)[0]}.db'
            self.conn = sqlite3.connect(self.path)
            self.cur = self.conn.cursor()
        else:
            if fafile:
                print(f'Creating new database: {os.path.splitext(dbfile)[0]}.db')
                self.path = dbfile
                seqs = self.read_fasta(fafile)
                self.init_db(seqs)
            else:
                print('No database file found. Provide a fasta file to create a new database.')

    
    def close(self):
        """Closes the database connection.
        """

        print(f'Closing database: {self.path}\n')
        self.conn.close()


    def read_fasta(self, fafile: str) -> dict:
        """Returns a dictionary of sequences from a fasta file. They are sorted by lenth to
        optimize fingerprinting.

        Args:
            fafile (str): Path to fasta file.

        Returns:
            dict: Dictionary of sequences.
        """

        seqs = {}
        with open(fafile, 'r', encoding='utf8') as f:
            for line in f:
                if line.startswith('>'):
                    pid = line.strip().split()[0][1:]
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
                fpcount integer,
                fingerprints blob
                ); """
        self.cur.execute(table)

        # Insert sequences
        insert = """ INSERT INTO sequences(pid, sequence, length, domains, fpcount, fingerprints)
            VALUES(?, ?, ?, ?, ?, ?) """
        for pid, seq in seqs.items():
            self.cur.execute(insert, (pid, seq, len(seq), '', 0, None))
        self.conn.commit()


    def yield_seqs(self, maxlen: int, cpu: int, dim1: int = 3, dim2: int = 80):
        """Yields sequences from the database.

        Args:
            maxlen (int): Maximum length of total sequence to yield
            cpu (int): Number of cpu cores to use (consequently, hard maximum of seqs to yield)
            dim1 (int): First dimension of quantization.
            dim2 (int): Second dimension of quantization.

        Yields:
            list: List of tuples where elements are protein ID and sequence.
        """

        seqs, curr_len, min_size = [], 0, dim1*dim2
        select = """ SELECT pid, sequence, length FROM sequences WHERE domains = '' """
        rows = self.cur.execute(select).fetchall()
        for row in rows:
            pid, seq, length = row
            if (length-2) * dim2 < min_size:  # Short sequences can't be quantized
                continue
            curr_len += length

            # If list is too large (length/number of seqs), yield and reset
            if (len(seqs) > 1 and curr_len > maxlen) or len(seqs) > cpu:
                last_seq = seqs.pop()
                yield seqs
                curr_len = len(last_seq[1])
                seqs = [(last_seq[0], last_seq[1])]
            seqs.append((pid, seq))  # New sequence

        # Last batch in file may be too large
        if len(seqs) > 1 and curr_len > maxlen:
            last_seq = seqs.pop()
            yield seqs

        # Yield unless there are no sequences selected from database
        try:
            yield [(last_seq[0], last_seq[1])]
        except UnboundLocalError:
            print('No sequences to fingerprint!')


    def add_fprint(self, fp: Fingerprint):
        """Adds domains and fingerprints to database.

        Args:
            fp (Fingerprint): Fingerprint object to add to database.
        """

        doms = ', '.join([str(x) for x in fp.domains])
        quants = np.array([fp.quants[dom] for dom in fp.domains])
        quants_bytes = BytesIO()
        np.save(quants_bytes, quants, allow_pickle=True)

        # Update database with domains as a string and fingerprints as a blob
        update = """ UPDATE sequences SET domains = ?, \
                        fpcount = ?, fingerprints = ? WHERE pid = ? """
        self.cur.execute(update, (doms, len(fp.domains), quants_bytes.getvalue(), fp.pid))
        self.conn.commit()


    def load_fprints(self) -> dict:
        """Returns a dictionary of fingerprints from the database.

        Returns:
            dict: Dictionary of fingerprints where key is protein ID and value is np.array of
                  fingerprints.
        """

        select = """ SELECT pid, fingerprints FROM sequences WHERE fpcount > 0 """
        self.cur.execute(select)
        pids, fingerprints = zip(*self.cur.fetchall())
        fingerprints = [np.load(BytesIO(fp), allow_pickle=True) for fp in fingerprints]

        return dict(zip(pids, fingerprints))


    def db_info(self):
        """Prints information about the database.
        """

        # Number of sequences
        select = """ SELECT COUNT(*) FROM sequences """
        num_seqs = self.cur.execute(select).fetchone()[0]
        print(f'Number of sequences: {num_seqs}')

        # Average sequence length
        select = """ SELECT AVG(length) FROM sequences """
        avg_len = self.cur.execute(select).fetchone()[0]
        print(f'Average sequence length: {avg_len:.2f}')

        # Number of domains in the database
        select = """ SELECT SUM(fpcount), COUNT(*) FROM sequences WHERE fpcount > 0 """
        nom_dom, dom_seqs = self.cur.execute(select).fetchone()
        print(f'Number of fingerprints: {nom_dom} ({dom_seqs}/{num_seqs} sequences)\n')


    def seq_info(self, seq: str):
        """Prints information about a specific sequence.

        Args:
            seq (str): Protein ID of sequence in database.
        """

        # Get sequence length and domains
        print(f'Protein ID: {seq}')
        select = """ SELECT sequence, domains FROM sequences WHERE pid = ? """
        try:
            sequence, domains = self.cur.execute(select, (seq,)).fetchone()
        except TypeError:
            print('Sequence not found in database\n')
            return
        
        # Print information
        print(f'Sequence: {sequence}')
        if domains:
            print(f'Domains: {domains}\n')
        else:
            print('No domains in database\n')
