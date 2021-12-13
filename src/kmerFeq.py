import subprocess
from pathlib import Path
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm


class KmerFreq():
    def __init__(self, query, mer, elements, out="results/kmerFreq") -> None:
        self.query = query
        self.mer = mer
        self.elements = elements
        self.out = Path(out)
        Path.mkdir(self.out, parents=True, exist_ok=True)

    def get_kmer_count_presentation(self):
        kmer_freq = None
        data = Fasta(self.query)

        temp_path = str(self.out.joinpath("temp.fna"))
        temp_dump_jf_path = str(self.out.joinpath("dump_temp.jf"))
        temp_dump_fna_path = str(self.out.joinpath("dump_temp.fna"))
        for genome in tqdm(data, total=len(data.keys())):
            with open(temp_path, 'w') as f:
                f.write('>' + genome.long_name + "\n")
                f.write(str(genome))
            subprocess.run(
                f"jellyfish count -m {self.mer} -s {self.elements} -t 10 {temp_path} -o {temp_dump_jf_path}", shell=True)
            subprocess.run(
                f"jellyfish dump {temp_dump_jf_path} -c > {temp_dump_fna_path}", shell=True)
            kmer_count = pd.read_csv(temp_dump_fna_path, sep=" ",
                                     names=['Kmer', genome.name], index_col='Kmer')
            if kmer_freq is None:
                kmer_freq = kmer_count
            else:
                kmer_freq = kmer_freq.join(kmer_count, how='outer')
        kmer_freq = kmer_freq.fillna(0)
        kmer_freq = (kmer_freq - kmer_freq.mean())/kmer_freq.std()
        kmer_freq.to_csv(str(self.out.joinpath("kmer_freq.csv")))


a = KmerFreq("raw_data/full_viral_complete_genome.fna", 5, "1000")
a.get_kmer_count_presentation()
