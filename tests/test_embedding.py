"""Testing Embedding class in embedding.py

__author__ = "Ben Iovino"
__date__ = "4/13/24"
"""

from src.embedding import Model, Embedding


def read_file(file: str) -> dict[str, str]:
    """Returns dictionary of fasta sequences.

    Args:
        file (str): Path to fasta file.

    Returns:
        dict[str, str]: Dictionary where key is protein ID, value is sequence.
    """

    seqs = {}
    with open(file, 'r') as fafile:
        for line in fafile:
            if line.startswith('>'):
                pid = line.strip()[1:]
                seqs[pid] = ''
            else:
                seqs[pid] += line.strip()

    return seqs


def test_model():
    """Test Model class.
    """

    model = Model('esm2', 't30')
    assert model.encoder is not None
    assert model.alphabet is not None
    assert model.tokenizer is not None


def test_embedding():
    """Read fasta file and test Embedding class.
    """
    
    seqs = read_file('tests/data/test.fa')
    model = Model('esm2', 't30')

    for pid, seq in seqs.items():
        embedding = Embedding(pid, seq)

        # Test split_seq
        subseqs = embedding.split_seq(500, 100)
        assert len(subseqs) == len(seq) // 400 + 1

        # Test embedding and contact map shapes
        embedding.embed_seq(model, 'cpu', [15, 21], 500)
        for embed in embedding.embed.values():
            assert embed.shape[0] == len(seq)
        assert embedding.contacts.shape[0] == len(seq)
        assert embedding.contacts.shape[1] == len(seq)


def main():

    test_model()
    test_embedding()
    print('All tests passed!')


if __name__ == '__main__':
    main()
        