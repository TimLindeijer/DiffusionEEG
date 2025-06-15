#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=24:00:00
#SBATCH --job-name=preprocess_caueeg2
#SBATCH --output=outputs/preprocess_caueeg2.out

# Create dataset directory if it doesn't exist
mkdir -p dataset

uenv miniconda3-py311
conda activate preprocess-eeg
#pip list
python -u preprocess/caueeg2.py
