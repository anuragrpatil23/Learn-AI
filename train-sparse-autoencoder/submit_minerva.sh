#!/bin/bash
# Submit one training run to Minerva's GPU queue.
# Usage: ./submit_minerva.sh <name> <features> <lam> <rows> [hours]
# Example: ./submit_minerva.sh f8192_lam0.2 8192 0.2 1e8 4
set -e
NAME=$1; FEATURES=$2; LAM=$3; ROWS=$4; HOURS=${5:-4}
ROOT=/sc/arion/work/patila06/Learn-AI/train-sparse-autoencoder
mkdir -p $ROOT/runs/$NAME
bsub <<JOB
#BSUB -J sae_$NAME
#BSUB -P acc_rg_HPIMS
#BSUB -q gpu
#BSUB -n 4
#BSUB -R "rusage[mem=12000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -W $HOURS:00
#BSUB -o $ROOT/runs/$NAME/job.out
#BSUB -e $ROOT/runs/$NAME/job.err
module load python/3.10.17
source /sc/arion/work/patila06/VirtualEnvs/sae_venv_31017/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv
python train.py --tokens '$ROOT/data/tokens_*.npy' --features $FEATURES --lam $LAM --rows $ROWS --out runs/$NAME
JOB
