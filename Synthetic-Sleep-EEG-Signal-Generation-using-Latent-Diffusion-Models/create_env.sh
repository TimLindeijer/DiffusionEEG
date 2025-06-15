#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=24:00:00
#SBATCH --job-name=env_setup_compile
#SBATCH --output=outputs/env_setup.out
 
# Activate environment
uenv miniconda3-py39
conda env create -f Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/environment.yml
# pip install monai-generative
# pip install lpips
# pip install omegaconf
# pip install tensorboard
# pip install mlflow
# pip install mne
# pip install tensorboardX
# pip install wandb

