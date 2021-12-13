from pyfaidx import Fasta
import pandas as pd
import subprocess
from pathlib import Path

for i in range(1, 5):
    subprocess.run(
        ["wget", f"https://ftp.ncbi.nlm.nih.gov/refseq/release/viral/viral.{i}.1.genomic.fna.gz"])
    subprocess.run(['gunzip', '-d', f'viral.{i}.1.genomic.fna.gz'])

subprocess.run(["cat *.fna > full_viral.fna"], shell=True)
path = Path("raw_data")
Path.mkdir(path, parents=True, exist_ok=True)
subprocess.run(['mv', 'full_viral.fna', str(path)])
subprocess.run(["rm *.fna"], shell=True)

accession = []
description = []
category = []
data = Fasta("raw_data/full_viral.fna")
for v in data:
    l = v.long_name.split(" ")
    a = l[0]
    c = " ".join(l[-2:])
    d = " ".join(l[1:-2])[:-1]
    accession.append(a)
    description.append(d)
    category.append(c)
metadata = pd.DataFrame(
    {"Accession": accession, "Description": description, "Category": category})
metadata.to_csv("raw_data/annotation.csv", index=False)


metadata = pd.read_csv("raw_data/annotation.csv")
complete_genomes = metadata[metadata['Category']
                            == 'complete genome']['Accession'].values
with open(path.joinpath('full_viral_complete_genome.fna'), 'w') as f:
    for g in complete_genomes:
        seqFile = data[g]
        seqLength = len(str(seqFile))
        if seqLength > 6000 and seqLength < 7000:
            f.write('>' + seqFile.long_name + "\n")
            f.write(str(seqFile) + "\n")
