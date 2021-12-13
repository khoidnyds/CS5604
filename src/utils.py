
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



def read_fasta_to_dict(fasta_file):
    gene_dict = {}
    with open(fasta_file, "r") as f:
        all_fastas = f.read().split("\n\n")
        for fasta in all_fastas:
            lines = fasta.split("\n")
            gene_dict[lines[0]] = ''.join(lines[1:])
    return gene_dict

def build_kmer_token_list(fasta_file, ksize):
    kmer_list = []
    gene_dict = read_fasta_to_dict(fasta_file)
    for label in gene_dict.keys():
        seq_kmers = build_kmers(gene_dict[label], ksize)
        kmer_list.extend(seq_kmers)
    kmer_list = list(set(kmer_list))
    
    return kmer_list

def pretty_time_delta(seconds):
    sign_string = '-' if seconds < 0 else ''
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%s%dd%dh%dm%ds' % (sign_string, days, hours, minutes, seconds)
    elif hours > 0:
        return '%s%dh%dm%ds' % (sign_string, hours, minutes, seconds)
    elif minutes > 0:
        return '%s%dm%ds' % (sign_string, minutes, seconds)
    else:
        return '%s%ds' % (sign_string, seconds)

