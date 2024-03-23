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

        Sequences table is primarily used for taking protein sequences to create the fingerprints
        table. Fingerprints are stored in a separate table to allow ease of access when searching
        using FAISS which returns the indices of the fingerprints in a flat numpy array.

        Args:
            seqs (dict): Dictionary of sequences.
        """

        self.path = os.path.splitext(self.path)[0]
        self.conn = sqlite3.connect(f'{self.path}.db')
        self.cur = self.conn.cursor()

        # Create table for protein sequences
        table = """CREATE TABLE sequences (
                pid text PRIMARY KEY,
                sequence text NOT NULL,
                length integer NOT NULL,
                fpcount integer NOT NULL
                ); """
        self.cur.execute(table)

        # Create table for domain fingerprints
        table = """CREATE TABLE fingerprints (
                vid integer PRIMARY KEY,
                domain text NOT NULL,
                fingerprint blob NOT NULL,
                pid text NOT NULL,
                FOREIGN KEY(pid) REFERENCES sequences(pid)
                ); """
        self.cur.execute(table)

        # Insert sequences
        insert = """ INSERT INTO sequences(pid, sequence, length, fpcount)
            VALUES(?, ?, ?, ?) """
        for pid, seq in seqs.items():
            self.cur.execute(insert, (pid, seq, len(seq), 0))
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
        select = """ SELECT pid, sequence, length FROM sequences WHERE fpcount = 0 """
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
            if seqs:  # only one sequence in fasta/database
                yield seqs
            else:
                print('No sequences to fingerprint!')


    def add_fprint(self, fp: Fingerprint):
        """Adds domains and fingerprints to database.

        Args:
            fp (Fingerprint): Fingerprint object to add to database.
        """

        # Convert quantizations to bytes for db storage
        quants = np.array([fp.quants[dom] for dom in fp.domains])
        quants_bytes = BytesIO()
        np.save(quants_bytes, quants, allow_pickle=True)

        # Update sequences table with number of fingerprints to keep track of progress
        update = """ UPDATE sequences SET fpcount = ? WHERE pid = ? """
        self.cur.execute(update, (len(fp.domains), fp.pid))

        # Get id of last fingerprint in database
        select = "SELECT vid FROM fingerprints ORDER BY vid DESC LIMIT 1"
        try:
            vid = self.cur.execute(select).fetchone()[0] + 1
        except TypeError:
            vid = 1

        # Add each domain and it's fingerprint to the fingerprints table
        insert = """ INSERT INTO fingerprints(vid, domain, fingerprint, pid)
            VALUES(?, ?, ?, ?) """
        for i, (dom, quant) in enumerate(zip(fp.domains, quants)):
            quants_bytes = BytesIO()
            np.save(quants_bytes, quant, allow_pickle=True)
            self.cur.execute(insert, (vid+i, dom, quants_bytes.getvalue(), fp.pid))
        self.conn.commit()  


    def load_fprints(self, pid: str = '') -> list:
        """Loads fingerprints from database.

        Args:
            pid (str): Protein ID of sequence in database (optional)

        Returns:
            list: List of numpy arrays
        """

        if pid:  # specific sequence
            select = """ SELECT vid, fingerprint FROM fingerprints WHERE pid = ? """
            fps = self.cur.execute(select, (pid,)).fetchall()
        else:  # all sequences
            select = """ SELECT vid, fingerprint FROM fingerprints """
            fps = self.cur.execute(select).fetchall()

        # Load fingerprints
        fprints = []
        for fp in fps:
            fprint = np.load(BytesIO(fp[1]), allow_pickle=True)
            fprints.append((fp[0], fprint))
        
        return fprints


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

        # Get sequence from sequences table
        print(f'Protein ID: {seq}')
        select = """ SELECT sequence FROM sequences WHERE pid = ? """
        try:
            sequence = self.cur.execute(select, (seq,)).fetchone()[0]
        except TypeError:
            print('Sequence not found in database\n')
            return
        
        # Get domains from fingerprints table
        select = """ SELECT domain FROM fingerprints WHERE pid = ? """
        domains = self.cur.execute(select, (seq,)).fetchall()
        
        # Print information
        print(f'Sequence: {sequence}')
        if domains:
            print(f'Domains: {", ".join([dom[0] for dom in domains])}\n')
        else:
            print('No domains in database\n')
