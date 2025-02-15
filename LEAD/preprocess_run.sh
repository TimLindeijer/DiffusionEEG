#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=03:15:00
#SBATCH --job-name=preproc
#SBATCH --output=outputs/preproc.out
 
# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py311
conda activate /home/stud/timlin/.conda/envs/DiffusionEEG
echo "Activated environment path: $CONDA_PREFIX"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
python -u LEAD/LEAD_preprocessing.py