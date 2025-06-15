#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --job-name=generate_eeg
#SBATCH --output=outputs/ldm_caueeg_generate_%j.out

# Signal handling for automatic requeue
_requeue_() {
    echo "Caught SIGUSR1 signal, requeueing job"
    scontrol requeue $SLURM_JOB_ID
    exit 0
}

# Trap the USR1 signal
trap _requeue_ SIGUSR1

# Report which node we're running on
echo "Running on $(hostname)"
echo "Job started at $(date)"

# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda activate ldm-env

cd Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models

# Add run identifier to track job in logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Set descriptive names for conditions
if [ $# -eq 0 ]; then
    echo "Error: No condition specified. Usage: sbatch $0 <condition>"
    echo "Available conditions: hc, mci, dementia"
    exit 1
fi

CONDITION=$1

# Validate condition
if [[ ! "$CONDITION" =~ ^(hc|mci|dementia)$ ]]; then
    echo "Error: Invalid condition '$CONDITION'. Must be one of: hc, mci, dementia"
    exit 1
fi

echo "Run ID: ${TIMESTAMP}, generating on ${CONDITION} only"

# Run the generation script with PSD normalization
python src/generate_eeg.py \
  --category $CONDITION \
  --hc_model_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/ldm_caueeg2_4ch_116size_label_hc/final_model.pth \
  --mci_model_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/ldm_caueeg2_4ch_116size_label_mci/final_model.pth \
  --dementia_model_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/ldm_caueeg2_4ch_116size_label_dementia/final_model.pth \
  --hc_autoencoder_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/aekl_eeg_4channels_label_hc/best_model.pth \
  --mci_autoencoder_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/aekl_eeg_4channels_label_mci/best_model.pth \
  --dementia_autoencoder_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/aekl_eeg_4channels_label_dementia/best_model.pth \
  --autoencoder_config /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_encoder_eeg.yaml \
  --diffusion_config /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_ldm.yaml \
  --original_label_path /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2/Label/label.npy \
  --original_data_path /home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2/Feature \
  --output_dir /home/stud/timlin/bhome/DiffusionEEG/dataset/LDM_PSD_Normalized_FIX \
  --num_timepoints 1000 \
  --diffusion_steps 1000 \
  --batch_epochs 64 \
  --normalize_psd \
  --per_channel \
  --plot_psd \
  --reference_sample_count 10 \
  --sampling_rate 200

# Record completion
echo "Job completed at $(date)"