**************************************************************************************************************
# Benchmarking - SCOP
**************************************************************************************************************

This directory tests DCTSearch using the MMseqs2 [1] benchmark. This benchmark contains full length sequences with annotated SCOP (1.75) domains, 6370 as queries about 3.4 million as targets. In their original benchmark, they shuffle each query sequence outside of it's SCOP domain, however this greatly decreases DCTSearch sensitivity so we use the original query sequences. They also added another 27 million reverse sequences to the database to add noise but we do not include these sequences in our benchmark due to size and time constraints. Results are considered true positives if they belong to the same family as the query, and false positives if they are outside of the same fold. We calculate the AUC1 scores for each query (ignoring self hits).

## Running the benchmark
You can prepare the MMseqs2 benchmark by running the follow command from the root directory:

```
python -m bench.scop.get_bench
```

This command will download the dataset and fingerprint each sequence, as well as prepare the query sequences.

The benchmark can then be run and evaluated using the following commands:

```
python -m bench.run_dct --bench scop
python -m bench.run_mean --bench scop
python -m bench.run_mmseqs --bench scop
python -m bench.plot_results --bench scop
```


## References

[1] Steinegger, Martin, and Johannes Söding. "MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets." Nature biotechnology 35.11 (2017): 1026-1028.