#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=24:00:00
#SBATCH --job-name=norm_lead
#SBATCH --output=outputs/norm_lead.out
 
# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py38
conda activate LEADEEG
# pip install mne
python -u LEAD/normalize_lead.py