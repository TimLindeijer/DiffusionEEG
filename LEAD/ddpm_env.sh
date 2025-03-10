#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu
#SBATCH --time=03:15:00
#SBATCH --job-name=env_setup_compile
#SBATCH --output=outputs/env_setup.out
 
# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda activate LEAD39
pip install -r LEAD/DS-DDPM/requirements.txt
pip install tensorboardx
pip install labml
pip install scikit-learn
pip install torch torchvision
pip install labml-helpers

