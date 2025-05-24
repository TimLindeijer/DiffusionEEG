#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=discriminate_augmented
#SBATCH --output=outputs/discriminate_ftsurrogate_dataset.out

# Report which node we're running on
echo "Running on $(hostname)"
echo "Job started at $(date)"

# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda activate ldm-eeg

cd Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models

# Run discrimination test between genuine and FTSurrogate augmented data
python src/dataset_creation/discrimination_datasets.py \
  --genuine_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
  --comparison_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2_FTSurrogate \
  --comparison_type augmented \
  --output_dir /home/stud/timlin/bhome/DiffusionEEG/dataset/discrimination_ftsurrogate \
  --val_size 0.2

# Record completion
echo "Job completed at $(date)"