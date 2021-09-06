#!/bin/bash
#SBATCH --output=sbatch_log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --mail-type=ALL

eval "$(/itet-stor/segerm/net_scratch/conda/bin/conda shell.bash hook)"
conda activate pytcu10

python -u display_patch_drop.py "$@"
