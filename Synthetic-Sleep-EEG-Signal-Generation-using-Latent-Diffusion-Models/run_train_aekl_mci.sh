#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=aekl_mci
#SBATCH --output=outputs/aekl_caueeg_mci_4channels.out
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
echo "Run ID: ${TIMESTAMP}, training on MCI only"

# Run the training script with MCI filter
python -u src/train_autoencoderkl.py \
    --dataset caueeg2 \
    --path_pre_processed /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
    --label_filter mci \
    --spe spectral \
    --config_file /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_encoder_eeg_new.yaml


# Record completion
echo "Job completed/interrupted at $(date)"