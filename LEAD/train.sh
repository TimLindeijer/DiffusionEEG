#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=DDPM_LEAD
#SBATCH --output=outputs/DDPM_LEAD.out
 
# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda activate LEAD39
# pip install comet_ml
python -u LEAD/train_2.py