#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=03:15:00
#SBATCH --job-name=models_test
#SBATCH --output=outputs/models_test.out
 
# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py311
conda activate TorchDiffEEG
python -u models.py