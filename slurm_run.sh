#!/bin/bash
#SBATCH --output=sbatch_log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --mail-type=ALL
##SBATCH --array=0-11

eval "$(/itet-stor/segerm/net_scratch/conda/bin/conda shell.bash hook)"
conda activate pytcu10

DATA_PATH="/scratch_net/biwidl215/segerm/ImageNetVal2012/"  # seems to run faster from here??
#DATA_PATH="/srv/beegfs-benderdata/scratch/density_estimation/data/segerm/ImageNetVal2012"

#python -u -m pip install -r requirements.txt

python -u main.py
