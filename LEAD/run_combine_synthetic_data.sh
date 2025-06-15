#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=24:00:00
#SBATCH --job-name=combine_lead
#SBATCH --output=outputs/combine_lead.out
 
# Activate environment
uenv miniconda3-py38
conda activate test-lead
# pip install mne
python -u LEAD/combine_synthetic_data.py