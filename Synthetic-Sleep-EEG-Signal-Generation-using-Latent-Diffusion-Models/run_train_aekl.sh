#!/bin/bash
# Script to submit all SLURM jobs

# Make output directory if it doesn't exist
mkdir -p outputs

# Submit all jobs
echo "Submitting Healthy Controls job..."
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_aekl_hc.sh

echo "Submitting MCI job..."
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_aekl_mci.sh

echo "Submitting Dementia job..."
sbatch Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/run_train_aekl_dementia.sh

echo "All jobs submitted. Check status with 'squeue -u $USER'"