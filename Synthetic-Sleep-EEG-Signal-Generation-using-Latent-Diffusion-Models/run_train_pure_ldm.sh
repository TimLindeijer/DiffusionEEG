#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=dm_spec_min_2
#SBATCH --output=outputs/dm_caueeg2_spec_min_2_%j.out
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

# Signal handling for automatic requeue
_requeue_() {
    echo "Caught SIGUSR1 signal, requeueing job"
    scontrol requeue $SLURM_JOB_ID
    exit 0
}

# Trap the USR1 signal
trap _requeue_ SIGUSR1

# Report which node we're running on
echo "Running on $(hostname)"
echo "Job started at $(date)"

# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda activate ldm-eeg

cd Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models

# Add run identifier to track job in logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Set descriptive names for conditions
if [ $# -eq 0 ]; then
    echo "Error: No condition specified. Usage: sbatch $0 <condition>"
    echo "Available conditions: hc, mci, dementia"
    exit 1
fi

CONDITION=$1

# Validate condition
if [[ ! "$CONDITION" =~ ^(hc|mci|dementia)$ ]]; then
    echo "Error: Invalid condition '$CONDITION'. Must be one of: hc, mci, dementia"
    exit 1
fi

echo "Run ID: ${TIMESTAMP}, training on ${CONDITION} only"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Run the training script
python src/train_pure_ldm.py \
    --dataset caueeg2 \
    --path_pre_processed /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
    --label_filter $CONDITION \
    --config_file /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_dm.yaml \
    --spe spectral

# Record completion
echo "Job completed/interrupted at $(date)"