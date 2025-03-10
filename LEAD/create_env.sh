#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=03:15:00
#SBATCH --job-name=env_setup_compile
#SBATCH --output=outputs/env_setup.out
 
# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda create --name LEAD39 python=3.9
conda activate LEAD39
pip install -r LEAD/src/requirements.txt
pip install mne
pip install h5py
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
