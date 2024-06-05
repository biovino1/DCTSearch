**************************************************************************************************************
# Benchmarking - Models
**************************************************************************************************************

This directory tests different protein language models for the usefulness of their embeddings in similarity search. We use a benchmark similar to HHblits [1] where we take a clustered version of SCOP v2.08 in which sequences are clustered to 20% sequence identity, leaving 7978 sequences in total. We then generate fingerprints from each layer of the model and search them against each other to determine which layer is best for similarity search. 

## References

[1] Remmert, Michael, et al. "HHblits: lightning-fast iterative protein sequence searching by HMM-HMM alignment." Nature methods 9.2 (2012): 173-175.
