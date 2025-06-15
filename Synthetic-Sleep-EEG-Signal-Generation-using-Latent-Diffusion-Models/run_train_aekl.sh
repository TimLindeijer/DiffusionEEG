#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --job-name=aekl
#SBATCH --output=outputs/aekl_caueeg_%j.out

# Check if condition argument is provided
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
conda activate ldm-env

cd Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models

# Add run identifier to track job in logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Run ID: ${TIMESTAMP}, training on ${CONDITION} only"

# Run the training script with specified condition filter
python -u src/train_autoencoderkl.py \
    --dataset caueeg2 \
    --path_pre_processed /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
    --config_file /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_encoder_eeg.yaml \
    --label_filter $CONDITION \
    --clip_grad 0.5 \
    # --spe spectral \
    # --spectral_cap 10
# Uncomment the above line to enable spectral training

# Record completion
echo "Job completed/interrupted at $(date)"