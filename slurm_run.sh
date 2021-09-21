#!/bin/bash
#SBATCH --output=sbatch_log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --array=1-12

eval "$(/itet-stor/segerm/net_scratch/conda/bin/conda shell.bash hook)"
conda activate pytcu10

DATA_PATH="/scratch_net/biwidl215/segerm/ImageNetVal2012_split"  # seems to run faster from here??
#DATA_PATH="/srv/beegfs-benderdata/scratch/density_estimation/data/segerm/ImageNetVal2012_split"

#python -u -m pip install -r requirements.txt

#python -u main.py --model_name dino_small --patch_size 8

python -u mask_predictor.py --is-sbatch --imgnet-val-dir $DATA_PATH  --batch-size 64 --epochs 20 --ratio-weight 2.0 \
  --pruning-locs $SLURM_ARRAY_TASK_ID --keep-ratios 0.4

#python -u ddp_hello_world.py "$@"
