**************************************************************************************************************
# Benchmarking - CATH20
**************************************************************************************************************

This directory tests DCTSearch using CATH20 v4.2.0 in the same way as knnProtT5 [1]. This clustered version of CATH contains 14,433 domain sequences in 5,125 families. All domains were added to the target database. 10,874 domains from 1,566 of the families with more than one domain were used as queries. Results are considered true positives if they belong to the same homologous superfamily as the query, and false positives if they belong to different superfamilies. We calculate both the top-1 hit rates and AUC1 scores for each query (ignoring self hits).

## Running the benchmark
You can prepare the cath20 dataset by running the follow command from the root directory:

```
python -m bench.cath.get_cath20
```

This command will download the dataset and fingerprint each sequence, as well as prepare the query sequences.

The benchmark can then be run with the following command:

```
python -m bench.cath.run_cath20
```

This will search the queries against the database and calculate/graph the AUC1 score for each query.

## References

[1] Schütze, Konstantin, et al. "Nearest neighbor search on embeddings rapidly identifies distant protein relations." Frontiers in Bioinformatics 2 (2022): 1033775.