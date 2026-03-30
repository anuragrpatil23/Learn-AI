#BSUB -J anurag_torch_run_large          # Job name
#BSUB -n 2                        # Number of cores
#BSUB -P acc_rg_HPIMS             # Project name
#BSUB -q gpu                      # Queue name
#BSUB -R "rusage[mem=5000]"       # Memory requirement
#BSUB -R "h10080g"                # GPU resource requirement
#BSUB -gpu "num=2"                # Number of GPUs
#BSUB -o anurag_torch_run_large.out     # Standard output file
#BSUB -e anurag_torch_run_large.err     # Standard error file
#BSUB -W 90:00                           # Time limit (12 hours)
#BSUB -R "span[hosts=1]"                 # Everything on same host

# Load the necessary modules
module load python/3.10.4

# Set the PYTHONPATH environment variable
export PYTHONPATH="/sc/arion/work/patila06/VirtualEnvs/nanogpt_venv_3104/lib/python3.10/site-packages:$PYTHONPATH"

# Activate your Python virtual environment
source /sc/arion/work/patila06/VirtualEnvs/nanogpt_venv_3104/bin/activate

# Change directory to build-nanogpt
cd /sc/arion/work/patila06/Learn-AI/build-nanogpt/

# Start GPU monitoring in the background
watch -n 1 nvidia-smi > gpu_usage.log &

export CUDA_LAUNCH_BLOCKING=1

# Run your Python script
# python your_script.py
# or, for PyTorch with distributed training
torchrun --nproc-per-node=2 build_nanogpt/train_gpt2.py
