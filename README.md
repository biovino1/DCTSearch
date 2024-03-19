**************************************************************************************************************
# Protein Similarity Searching Using DCT Representations
**************************************************************************************************************

This project uses ESM-2 to generate embeddings and contact maps for protein sequences and then performs iDCT quantization on their RecCut predicted domains to generate DCT representations, or fingerprints, for quick and accurate similarity searches.

## Installation and Dependencies
To install this respository:

```
git clone https://github.com/biovino1/DCTSearch
```

With python<=3.9 installed, you can install all required packages with pip:

```
pip install -r req.txt
```

## Creating a DCT Fingerprint Database
To create a database of DCT fingerprints for which you can query against, run the following command:

```
python scripts/make_db.py --fafile <.fa file> --dbfile <output>
```

This will generate a SQLite database file with one table containing a protein id, sequence, list of domains, and DCT fingerprints for each sequence in the fasta file. If the process is interrupted for any reason, you can restart it and the script will skip any sequences that have already had fingerprints generated.

You can interact with the database like any other SQLite database, but database.py provides a simple interface to interact with the database. For example, the following command will print basic information about the database, such the number of sequences, average length of sequences, and the number of fingerprints in the database:

```
from database import Database
db = Database('<.db file>')
db.db_info()
db.close()
```

You can also search the database for a specific sequence with the following command:

```
db.seq_info('<fasta id>')
```

### Embedding and Fingerprinting Protein Sequences
Embedding protein sequences can be a memory-intensive process. To allow for the embedding of very large sequences, we split any sequence larger than the '--maxlen' parameter into overlapping windows and embed each window separately. The final embedding is a a concatenation of each window's embedding and the overlapping segments are averaged together. The default value for '--maxlen' is 1000, but you can change this parameter to a larger or smaller value depending on your system's memory constraints. If you would like to use a GPU to embed your protein sequences, you can specify the number of GPUs with the '--gpu' parameter, otherwise it will default to using the CPU to embed sequences.

Generating fingerprints from these embeddings is much less memory intensive, but is performed only on the CPU. You can specify the number of CPU cores to use with the '--cpu' parameter to fingerprint multiple sequences at once. For example, the command below will embed sequences on one GPU and fingerprint them on 12 CPU cores:

```
python scripts/make_db.py --fafile <.fa file> --dbfile <output> --gpu 1 --cpu 12
```

## Querying a DCT Fingerprint Database
You can query a DCT fingerprint database with a fasta file containing one or more protein sequences with the following command:

```
python scripts/query_db.py --fafile <.fa file> --dbfile <.db file>
```

Again, you can change the '--maxlen' and '--gpu' parameters as described above.

As of right now, the query script will return the top hit from the database for each query sequence, along with the regions of the query and database sequence that were most similar and their similarity score. More functionality will be added in the future.