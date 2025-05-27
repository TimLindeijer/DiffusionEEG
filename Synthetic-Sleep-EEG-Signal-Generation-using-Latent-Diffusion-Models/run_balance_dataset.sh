#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=balance_dataset
#SBATCH --output=outputs/balance_dataset_caueeg2.out

# Report which node we're running on
echo "Running on $(hostname)"
echo "Job started at $(date)"

# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda activate ldm-eeg

cd Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models

# Can balance to max, mean or to a specific number
python src/dataset_creation/balance_dataset.py \
  --genuine_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
  --synthetic_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/LDM_PSD_Normalized \
  --output_dir /home/stud/timlin/bhome/DiffusionEEG/dataset/ldm_norm_balanced_datasets \
  --test_size 0.2 \
  --stratify \
  --balance_to max

echo "Job finished at $(date)"