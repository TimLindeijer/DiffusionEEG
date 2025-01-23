#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --job-name=categorize_participants
#SBATCH --output=outputs/categorize_participants

uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py311
conda activate TorchDiffEEG

#pip list
python -u categorize_participants/categorize_participants.py