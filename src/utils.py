
def build_kmers(sequence, ksize):
    """
    Building kmer from the sequence

    Args:
        sequence (str): input sequence
        ksize (int): k-mer

    Returns:
        list: list of kmer
    """
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)

    return kmers
