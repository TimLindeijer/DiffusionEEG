#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=48:00:00
#SBATCH --job-name=balance_dataset
#SBATCH --output=outputs/balance_dataset_dm.out

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
  --synthetic_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2_FTSurrogate \
  --output_dir /home/stud/timlin/bhome/DiffusionEEG/dataset/ftsurrogate_balanced_datasets \
  --test_size 0.2 \
  --stratify \
  --balance_to max \
  --dataset_type augmented  # Add this line

echo "Job finished at $(date)"