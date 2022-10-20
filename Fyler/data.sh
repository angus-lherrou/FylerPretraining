#!/bin/bash
#SBATCH --partition=chip-compute
#SBATCH --account=chip
#SBATCH --job-name=fyler-data
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=angus.dawson@childrens.harvard.edu
#SBATCH --output=log/%x_%j.txt
#SBATCH --error=log/err/%x_%j.err
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=10-0

conda init bash
source ~/.bashrc
conda activate pretraining

python -u experiments.py data cfgs
