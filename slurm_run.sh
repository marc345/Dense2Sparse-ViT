#!/bin/bash
#SBATCH --output=sbatch_log/%j.out
#SBATCH --gres=gpu:1
<<<<<<< HEAD
#SBATCH --mem=30G
=======
#SBATCH --mem=50G
>>>>>>> optimized_attention_map
#SBATCH --mail-type=ALL
#SBATCH --array=1-11

eval "$(/itet-stor/segerm/net_scratch/conda/bin/conda shell.bash hook)"
conda activate pytcu10

DATA_PATH="/scratch_net/biwidl215/segerm/ImageNetVal2012"  # seems to run faster from here??
#DATA_PATH="/srv/beegfs-benderdata/scratch/density_estimation/data/segerm/ImageNetVal2012_split"

#python -u -m pip install -r requirements.txt

#python -u main.py --model_name dino_small --patch_size 8

python -u mask_predictor.py --is-sbatch --imgnet-val-dir $DATA_PATH  --batch-size 64 --epochs 20 --ratio-weight 2.0 \
  --pruning-locs $SLURM_ARRAY_TASK_ID --keep-ratios 0.4
#$SLURM_ARRAY_TASK_ID
#python -u ddp_hello_world.py "$@"
