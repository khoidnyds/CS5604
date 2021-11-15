import subprocess
from pathlib import Path
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm


class Kmer():
    def __init__(self, query, mer, elements, out="temp") -> None:
        self.query = query
        self.mer = mer
        self.elements = elements
        out = Path(out)
        Path.mkdir(out, parents=True, exist_ok=True)
        self.out = str(out.joinpath("kmer_count.jf"))

    def get_kmer_count_presentation(self):
        df = None
        data = Fasta(self.query)
        for genome in tqdm(data, total=len(data.keys())):
            with open('temp/temp.fna', 'w') as f:
                f.write('>' + genome.long_name + "\n")
                f.write(str(genome))
            subprocess.run(
                f"jellyfish count -m {self.mer} -s {self.elements} -t 10 temp/temp.fna -o temp/dump_temp.jf", shell=True)
            subprocess.run(
                f"jellyfish dump temp/dump_temp.jf -c > temp/dump_temp.fna", shell=True)
            a = pd.read_csv("temp/dump_temp.fna", sep=" ",
                            names=['Kmer', genome.name], index_col='Kmer')
            if df is None:
                df = a
            else:
                df = df.join(a, how='outer')
            df.to_csv("temp/kmer_count.csv")


a = Kmer("raw_data/full_viral_complete_genome.fna", 5, "1000")
a.get_kmer_count_presentation()
