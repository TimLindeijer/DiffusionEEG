#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=03:15:00
#SBATCH --job-name=preproc2
#SBATCH --output=outputs/preproc2.out
 
# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py38
conda activate LEADEEG
# pip install mne
python -u LEAD/caueeg2.py