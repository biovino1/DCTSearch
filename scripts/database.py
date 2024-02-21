"""Defines the Database class, which is used to store more than one Fingerprint object.

__author__ = "Ben Iovino"
__date__ = "2/19/24"
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Database:
    """This class stores multiple Fingerprint objects.

    Attributes:
        pids (list): Protein IDs.
        idx (list): List of indices for domain/fingerprint list positions.
        doms (list): Domain boundaries.
        quants (list): Quantizations.
    """
    file: str = field(default_factory=None)
    pids: list = field(default_factory=list)
    idx: list = field(default_factory=list)
    doms: list = field(default_factory=list)
    quants: list = field(default_factory=list)


    def __post_init__(self):
        """If file is given, load database from file.
        """

        if self.file:
            self.load_db(self.file)


    def load_db(self, dbfile: str):
        """Loads a database from a npz file as four arrays: protein ids, domain/fingerprint
        indices, domain boundaries, and quantizations.

        Args:
            dbfile (str): Path to input file.
        """

        db = np.load(dbfile)
        self.pids = db['pids']
        self.idx = db['idx']
        self.doms = db['doms']
        self.quants = db['quants']
        db.close()


    def add_fprint(self, fprint):
        """Adds a Fingerprint object to the database.

        Args:
            fprint (Fingerprint): Fingerprint object.
        """

        # Get next starting index for domain boundaries
        if self.idx:
            prev_doms = len(self.doms[self.idx[-1]:])
            self.idx.append(self.idx[-1] + prev_doms)
        else:
            self.idx.append(0)

        # Add protein ID, domain boundaries, and quantizations
        self.pids.append(fprint.pid)
        self.doms.extend(fprint.domains)
        self.quants.extend(fprint.quants.values())


    def save_db(self, dbfile: str):
        """Saves the database to a npz file as four arrays: protein ids, domain/fingerprint
        indices, domain boundaries, and quantizations.

        Args:
            dbfile (str): Path to output file.
        """

        np.savez_compressed(dbfile, pids=self.pids, idx=self.idx,
                             doms=self.doms, quants=self.quants)
