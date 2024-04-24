"""Queries CATH20 database with sequences from query.fasta using DCTSearch.

__author__ = "Ben Iovino"
__date__ = "4/24/24"
"""

import argparse
import faiss
import logging
import os
import sys
sys.path.append(os.getcwd()+'/src')  # Add src to path
import numpy as np
from src.database import Database
from src.query_db import get_top_hits