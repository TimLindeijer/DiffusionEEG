#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=03:15:00
#SBATCH --job-name=plot
#SBATCH --output=outputs/plot_eeg.out
 
# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py311
conda activate TorchDiffEEG
python -u EEGFormer/plot_eeg.py