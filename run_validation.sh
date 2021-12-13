#!/bin/sh

eval "$(conda shell.bash hook)"
conda activate CS5604-proj
sbatch run_validation_job.sh