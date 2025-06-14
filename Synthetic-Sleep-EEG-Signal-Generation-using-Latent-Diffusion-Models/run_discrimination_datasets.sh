#!/bin/bash
#SBATCH --partition=cpu36
#SBATCH --time=48:00:00
#SBATCH --job-name=discriminate_dataset
#SBATCH --output=outputs/discriminate_dataset_caueeg2.out

# Report which node we're running on
echo "Running on $(hostname)"
echo "Job started at $(date)"

# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda activate ldm-eeg

cd Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models

  
python src/dataset_creation/discrimination_datasets.py \
  --genuine_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
  --comparison_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/DM_NO_SPEC \
  --comparison_type synthetic \
  --output_dir /home/stud/timlin/bhome/DiffusionEEG/dataset/dm_no_spec_discrimination_synthetic \
  --val_size 0.2