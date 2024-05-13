**************************************************************************************************************
# Benchmarking - PFAM20
**************************************************************************************************************

This directory tests DCTSearch using Pfam v33.1 in the same way as knnProtT5 [1]. From the full Pfam database, we only consider families that have 20 or more domains in them. From these, we take 20 sequences (at random) and retrieve their full length sequences. This provides 313,518 proteins with which we perform an all-vs-all search where we expect 20 correct hits for each query protein. We calculate both the top-1 hit rates and AUC1 scores for each query (ignoring self hits).

## Running the benchmark
You can prepare the pfam20 dataset by running the follow command from the root directory:

```
python -m bench.cath.get_pfam20
```

## References

[1] Schütze, Konstantin, et al. "Nearest neighbor search on embeddings rapidly identifies distant protein relations." Frontiers in Bioinformatics 2 (2022): 1033775.