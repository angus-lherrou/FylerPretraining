#!/bin/bash
#SBATCH --partition=chip-gpu
#SBATCH --account=chip
#SBATCH --job-name=dima-model
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=angus.dawson@childrens.harvard.edu
#SBATCH --output=log/%x_%j.txt
#SBATCH --error=log/err/%x_%j.err
#SBATCH --gres=gpu:Titan_RTX:1
#SBATCH --ntasks=2
#SBATCH --mem=32G
#SBATCH --time=10-0

conda init bash
source ~/.bashrc
conda activate pretraining

CUDA_LAUNCH_BLOCKING=1 python -u fyler_fextract.py fextract dima.cfg "$1" --model codes
