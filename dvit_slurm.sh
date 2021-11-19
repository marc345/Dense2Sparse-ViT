#!/bin/bash
##SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --output=sbatch_log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
##SBATCH --array=5
#SBATCH --mail-user=$USER@student.ethz.ch
#SBATCH --constraint='titan_xp|geforce_gtx_titan_x'
##SBATCH --array=3,5,7


while getopts ":n:d:" opt; do
  case $opt in
    n) export WANDB_NAME="$OPTARG"
    ;;
    d) export WANDB_NOTES="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

printf "Argument WANDB_NAME is %s\n" "$WANDB_NAME"
printf "Argument WANDB_NOTES is %s\n" "$WANDB_NOTES"

export WANDB_API_KEY=452b4cee4464de7d444c267f72d9b961904cb42a
export WANDB_USERNAME=marc345
export WAND_USER_EMAIL=m.seger345@gmail.com

eval "$(/itet-stor/segerm/net_scratch/conda/bin/conda shell.bash hook)"
conda activate pytcu10

DATA_PATH="/scratch_net/biwidl215/segerm/ImageNetVal2012"  # seems to run faster from here??
#DATA_PATH="/srv/beegfs-benderdata/scratch/density_estimation/data/segerm/ImageNetVal2012_split"

#python -u -m pip install -r requirements.txt

#0.001\

python -u train_dynamic_vit.py --is-sbatch --imgnet-val-dir $DATA_PATH --batch-size 64 --epochs 30 \
  --pruning-locs 2 --keep-ratios 0.7  --warmup-steps 5 --lr 0.0000625\
  --freeze-backbone --wandb

# --predictor-bn --use-mse-loss
# --use-ratio-loss --use-token-dist-loss
# --use-kl-div-loss --predictor-vit\
# --topk-selection --attn-selection --initial-sigma 5e-2
#--predictor-vit --use-kl-div-loss --softmax-temp 5e-2
# --topk-selection --attn-selection --initial-sigma 0.0005 --mean-heads
#--early-exit \
#--topk-selection --attn-selection --initial-sigma 0.05
# --use-ratio-loss
# --use-token-dist-loss
# $SLURM_ARRAY_TASK_ID

