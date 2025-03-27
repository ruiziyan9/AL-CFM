#!/bin/bash
#SBATCH --job-name=condgplvm_run
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Load any necessary modules (adjust if you're using a module system)
# module load python/3.9 cuda/11.7  # example
source /vol/cuda/11.8.0/setup.sh

# Activate your virtual environment if needed
conda activate meta_causal_inf

# Run your script
python3 experiments/file_train.py \
    --run_name=condgplvm_higherklinit0p001_10enc_0dec_plottest \
    --decoder_depth=0 \
    --num_layers_encoder=10 \
    --num_workers=10 \
    --batch_size=16

