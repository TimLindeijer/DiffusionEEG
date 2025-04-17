#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=03:15:00
#SBATCH --job-name=env_setup_compile
#SBATCH --output=outputs/env_setup.out
 
# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda env create -f neuro-green/environment.yml

