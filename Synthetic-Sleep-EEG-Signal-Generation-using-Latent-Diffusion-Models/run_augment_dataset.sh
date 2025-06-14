#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=48:00:00
#SBATCH --job-name=augment_dataset
#SBATCH --output=outputs/augment_dataset_caueeg2.out

# Report which node we're running on
echo "Running on $(hostname)"
echo "Job started at $(date)"

# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda activate ldm-eeg

cd Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models

python src/dataset_creation/augment_dataset.py \
  --genuine_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
  --synthetic_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/DM_SPEC_MINUS_2 \
  --output_dir /home/stud/timlin/bhome/DiffusionEEG/dataset/dm_norm_fix_spec_mins_2_ready_datasets \
  --test_size 0.2 \
  --stratify \
  --percentages 20,40,60,80,100