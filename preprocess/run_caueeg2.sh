#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=24:00:00
#SBATCH --job-name=preprocess_caueeg2
#SBATCH --output=outputs/preprocess_caueeg2.out

uenv miniconda3-py311
conda activate test_env
#pip list
python -u preprocess/caueeg2.py
