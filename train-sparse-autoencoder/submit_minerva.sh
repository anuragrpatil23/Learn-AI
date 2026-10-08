#!/bin/bash
# Submit one training run to Minerva's GPU queue.
# Usage: ./submit_minerva.sh <name> <features> <lam> <rows> [hours] [queue] [token files]
# Example: ./submit_minerva.sh f8192_lam0.2 8192 0.2 1e8 4
# Submitting again with the same name carries the run on from its last checkpoint.
set -e
NAME=$1; FEATURES=$2; LAM=$3; ROWS=$4; HOURS=${5:-4}; QUEUE=${6:-gpu}
ROOT=/sc/arion/work/patila06/Learn-AI/train-sparse-autoencoder
TOKENS=${7:-$ROOT/data/tokens_*.npy}
mkdir -p $ROOT/runs/$NAME
bsub <<JOB
#BSUB -J sae_$NAME
#BSUB -P acc_rg_HPIMS
#BSUB -q $QUEUE
#BSUB -n 4
#BSUB -R "rusage[mem=12000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -R "select[a10080g||h10080g||h100nvl]"
# 80 GB cards only: the lam 0.05 run ran out of memory on a 40 GB card when the row buffer refilled
#BSUB -W $HOURS:00
#BSUB -o $ROOT/runs/$NAME/job.out
#BSUB -e $ROOT/runs/$NAME/job.err
module load python/3.10.17
source /sc/arion/work/patila06/VirtualEnvs/sae_venv_31017/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less memory lost to fragments
cd $ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv
python train.py --tokens '$TOKENS' --features $FEATURES --lam $LAM --rows $ROWS --out runs/$NAME
JOB
