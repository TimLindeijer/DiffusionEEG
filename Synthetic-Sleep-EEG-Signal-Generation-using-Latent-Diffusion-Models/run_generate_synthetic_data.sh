#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --job-name=gen_eeg
#SBATCH --output=outputs/gen_synthetic_eeg_mean.out

# Report which node we're running on
echo "Running on $(hostname)"
echo "Job started at $(date)"

# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda activate ldm-eeg

cd Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models

BASE_DIR="/home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models"


# Define paths to model weights and configs with absolute paths
HC_MODEL="${BASE_DIR}/project/outputs/ldm_caueeg2_label_hc/final_model.pth"
MCI_MODEL="${BASE_DIR}/project/outputs/ldm_caueeg2_label_mci/final_model.pth"
DEMENTIA_MODEL="${BASE_DIR}/project/outputs/ldm_caueeg2_label_dementia/final_model.pth"

HC_AUTOENCODER="${BASE_DIR}/project/outputs/aekl_eeg_stable_label_hc/best_model.pth"
MCI_AUTOENCODER="${BASE_DIR}/project/outputs/aekl_eeg_stable_label_mci/best_model.pth"
DEMENTIA_AUTOENCODER="${BASE_DIR}/project/outputs/aekl_eeg_stable_label_dementia/best_model.pth"

AUTOENCODER_CONFIG="${BASE_DIR}/project/config/config_encoder_eeg_old.yaml"
DIFFUSION_CONFIG="${BASE_DIR}/project/config/config_ldm.yaml"

# Define original data directory and output directory with absolute paths
ORIGINAL_DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2"
OUTPUT_DIR="${BASE_DIR}/project/outputs/synthetic_caueeg2_mean_length"

# Path to original label file
REFERENCE_LABEL="${ORIGINAL_DATA_DIR}/Label/label.npy"

# Path to original label file
REFERENCE_LABEL="${ORIGINAL_DATA_DIR}/Label/label.npy"

# Number of samples to generate for each category
NUM_HC=1  # Increase as desired
NUM_MCI=1  # Increase as desired
NUM_DEMENTIA=1  # Increase as desired

# Run the generation script
python src/generate_synthetic_data.py \
  --hc_model_path $HC_MODEL \
  --mci_model_path $MCI_MODEL \
  --dementia_model_path $DEMENTIA_MODEL \
  --hc_autoencoder_path $HC_AUTOENCODER \
  --mci_autoencoder_path $MCI_AUTOENCODER \
  --dementia_autoencoder_path $DEMENTIA_AUTOENCODER \
  --autoencoder_config $AUTOENCODER_CONFIG \
  --diffusion_config $DIFFUSION_CONFIG \
  --original_data_dir $ORIGINAL_DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --reference_label_file $REFERENCE_LABEL \
  --num_hc $NUM_HC \
  --num_mci $NUM_MCI \
  --num_dementia $NUM_DEMENTIA \
  --batch_size 32 \
  --timepoints 464 \
  --fixed_length 71 \
  --seed 42 \
  --use_gpu
# Match actual trained dimensions (928)
# 71 is mean amount of genuine epochs