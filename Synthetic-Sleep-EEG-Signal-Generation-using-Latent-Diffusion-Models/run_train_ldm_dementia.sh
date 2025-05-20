#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=ldm_dementia
#SBATCH --output=outputs/ldm_caueeg2_dementia.out
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

# echo "Updating PyTorch for A100 compatibility..."
# pip uninstall -y torch torchvision torchaudio
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

cd Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models

# Add run identifier to track job in logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Run ID: ${TIMESTAMP}, training LDM on Healthy Controls only"

# Run the training script with dementia filter
python src/train_ldm.py \
    --dataset caueeg2 \
    --path_pre_processed /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
    --label_filter dementia \
    --num_channels "[16, 32, 64, 128]" \
    --latent_channels 8 \
    --best_model_path "/home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/aekl_eeg_caueeg2_label_dementia" \
    --autoencoderkl_config_file_path "/home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_encoder_eeg_new.yaml" \
    --config_file "/home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_ldm_new.yaml"

# Record completion
echo "Job completed/interrupted at $(date)"