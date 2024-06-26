**************************************************************************************************************
# Protein Similarity Searching Using DCT Representations
**************************************************************************************************************

This project uses ProtT5-XL-U50 to generate embeddings and ESM-2 to predict contact maps for protein sequences and then performs iDCT quantization on their RecCut predicted domains to generate DCT representations, or fingerprints, for quick and accurate similarity searches.

## Installation and Dependencies
With git and conda installed, you can clone the repository and install the required dependencies with the following commands:

```
git clone https://github.com/biovino1/DCTSearch
cd DCTSearch
conda env create -f env.yml
g++ -o src/RecCut src/RecCut.cpp
```

## Creating a DCT Fingerprint Database
To create a database of DCT fingerprints for which you can query against, run the following command:

```
python src/make_db.py --fafile <.fa file> --dbfile <output>
```

This will generate a SQLite database file with two tables containing, one for protein sequences and one for DCT fingerprints. If the process is interrupted for any reason, you can restart it and every sequence that has already had fingerprints generated will be skipped.

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

## Embedding and Fingerprinting Protein Sequences
Embedding protein sequences can be a memory-intensive process. To allow for the embedding of very large sequences, we split any sequence larger than the `--maxlen` parameter into overlapping windows (of length 200) and embed each window separately. The final embedding is a a concatenation of each window's embedding and the overlapping segments are averaged together. The default value for `--maxlen` is 500, but you can change this parameter to a larger or smaller value depending on your system's memory constraints. If you would like to use a GPU to embed your protein sequences, you can specify the number of GPUs with the `--gpu` parameter, otherwise it will default to using the CPU to embed sequences.

Generating fingerprints from these embeddings is much less memory intensive, but is performed only on the CPU. You can specify the number of CPU cores to use with the `--cpu` parameter to fingerprint multiple sequences at once (default is 1). For example, the command below will embed sequences on one GPU and fingerprint them on 12 CPU cores:

```
python src/make_db.py --fafile <.fa file> --dbfile <output> --gpu 1 --cpu 12
```

If you are having memory issues, you can determine your system's `--maxlen` parameter by running the following script:

```
python -m utils.ram_usage
```

This script will report to you how long of a sequence you can embed at once given your system's memory constraints. See the utils/README.md for more information.

## Querying a DCT Fingerprint Database
You can query a DCT fingerprint database with a fasta file containing one or more protein sequences with the following command:

```
python src/query_db.py --query <.fa file> --db <.db file>
```

Again, you can change the `--maxlen`, `--cpu`, and `--gpu` parameters to suit your system's hardware.

There is also a `--khits` parameter that will return the top k hits from the database for each query sequence, or as many hits as are available if there are fewer than k hits. Along with the hit, with the regions of the query and database sequence that were most similar and their similarity score will also be shown. More functionality will be added in the future.

## Saving Fingerprints

If you would like to use the fingerprints for other purposes, it is faster to save them as a numpy array (.npz) and work with them directly. You can save the fingerprints with the following command:

```
db = Database('<.db file>')
db.save_fprints('<output>')
db.close()
```

This array has four data types: pid, idx, dom, and fp. The pid is the protein id, idx is the starting index of the protein's fingerprints, dom is predicted domain region, and fp is the fingerprint itself.
