#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=03:15:00
#SBATCH --job-name=env_setup_compile
#SBATCH --output=outputs/env_setup.out
 
# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py38
conda create --name LEADEEG python=3.8
conda activate LEADEEG
pip install -r LEAD/LEAD/requirements.txt
pip install mne
pip install h5py
# pip uninstall torch torchvision torchaudio
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
