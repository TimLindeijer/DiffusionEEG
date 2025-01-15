#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --job-name=pre_process_EEG
#SBATCH --output=outputs/pre_process_EEG

#uenv verbose cuda-11.4 cudnn-11.4-8.2.4
#uenv miniconda3-py311
conda activate /home/stud/tordmy/.conda/envs/DiffusionEEG
echo "Activated environment path: $CONDA_PREFIX"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

#pip list
python -u preprocess.py