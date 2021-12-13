#!/bin/sh
#SBATCH --time=12:00:00
#SBATCH --partition=largemem_q
#SBATCH -n 32
#SBATCH --mem=600G
#SBATCH --account=genomicdatacompress

python src/main.py -i data/sample --test