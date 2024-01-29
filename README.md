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

### Embedding Protein Sequences
If you would like to use a GPU to embed your protein sequences, you can add '--gpu True' to the above command, otherwise it will default to using the CPU.

## Querying a DCT Fingerprint Database
You can query a DCT fingerprint database with a protein sequence in fasta format with the following command:

```
python scripts/query_db.py --fafile <.fa file> --dbfile <.db file>
```

Again, if you would like to use a GPU to embed your query sequence, you can add '--gpu True' to the above command.

As of right now, the query script will return the top hit from the database along with the similarity score. More functionality will be added in the future.