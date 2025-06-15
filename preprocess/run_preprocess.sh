#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=24:00:00
#SBATCH --job-name=pre_process_EEG
#SBATCH --output=outputs/pre_process_EEG.out

uenv miniconda3-py311
conda activate test_env
#pip list
python -u preprocess/preprocess.py
