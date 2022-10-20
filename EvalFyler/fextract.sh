#!/bin/bash
#SBATCH --partition=chip-gpu
#SBATCH --account=chip
#SBATCH --job-name=fyler-fextract
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=angus.dawson@childrens.harvard.edu
#SBATCH --output=log/%x_%j.txt
#SBATCH --error=log/err/%x_%j.err
#SBATCH --gres=gpu:Titan_RTX:1
#SBATCH --ntasks=2
#SBATCH --mem=16G
#SBATCH --time=10-0

conda init bash
source ~/.bashrc
conda activate pretraining

results_dir="fextract_results/$1/search3"
mkdir -p "$results_dir"
ls -1 "$results_dir" | sed s/_train\.txt//g > /tmp/fyler-fextract-exclude.txt
CUDA_LAUNCH_BLOCKING=1 python -u fyler_fextract.py batch -o "$results_dir" -g 1 --exclude /tmp/fyler-fextract-exclude.txt cfgs_search/$1 ../Fyler/models
