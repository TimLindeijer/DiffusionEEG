#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=ftsurrogate_aug
#SBATCH --output=outputs/ftsurrogate_augmentation.out

# Report which node we're running on
echo "Running on $(hostname)"
echo "Job started at $(date)"

# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda activate ldm-eeg

# Run the augmentation script
python augmentation/augment_dataset.py \
  --input_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2 \
  --output_dataset /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2_FTSurrogate \
  --delta_phi_max 2.827433388 \
  --independent_channels \
  --seed 42

# Record completion
echo "Job completed at $(date)"