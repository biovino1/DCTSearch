**************************************************************************************************************
# Benchmarking
**************************************************************************************************************

This directory contains the scripts to run several different benchmarks testing DCTSearch, MMseqs2 [1], and knnProtT5's [2] method (mean embeddings). Each benchmark has a README file with more information on the benchmark and how to run it.

To install the required dependences to run the benchmarks, run the following commands:

```
conda env create -f bench/env.yml
conda activate DCTSearch_bench
```

## References

[1] Steinegger, Martin, and Johannes Söding. "MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets." Nature biotechnology 35.11 (2017): 1026-1028.
[2] Schütze, Konstantin, et al. "Nearest neighbor search on embeddings rapidly identifies distant protein relations." Frontiers in Bioinformatics 2 (2022): 1033775.