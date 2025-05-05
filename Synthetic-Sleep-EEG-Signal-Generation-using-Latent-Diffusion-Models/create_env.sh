#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=03:15:00
#SBATCH --job-name=env_setup_compile
#SBATCH --output=outputs/env_setup.out
 
# Activate environment
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py39
conda deactivate
conda remove -n ldm-eeg --all -y
conda env create -f Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/environment.yml
# pip install monai-generative
# pip install lpips
# pip install omegaconf
# pip install tensorboard
# pip install mlflow
# pip install mne
# pip install tensorboardX

