#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=48:00:00
#SBATCH --job-name=create_integrated_PSD_stat_analysis
#SBATCH --output=outputs/create_integrated_PSD_stat_analysis_%j.out

# Report which node we're running on
echo "Running on $(hostname)"
echo "Job started at $(date)"

# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda activate ldm-eeg

python plots/integrated_PSD_stat_analysis_group.py