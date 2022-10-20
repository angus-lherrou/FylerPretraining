#!/bin/bash
#SBATCH --partition=chip-gpu
#SBATCH --account=chip
#SBATCH --job-name=fyler-train
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
ls -1 train_results | sed s/_train\.txt//g > /tmp/fyler-train-exclude.txt
python -u fyler_bow.py batch -o train_results -g 1 cfgs _docs models --exclude /tmp/fyler-train-exclude.txt
