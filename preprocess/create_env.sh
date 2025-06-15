#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=24:00:00
#SBATCH --job-name=env_setup_compile
#SBATCH --output=outputs/env_setup.out

# Create outputs directory if it doesn't exist
mkdir -p outputs
 
# Activate environment
uenv miniconda3-py311
conda env create -f preprocess/environment.yml
