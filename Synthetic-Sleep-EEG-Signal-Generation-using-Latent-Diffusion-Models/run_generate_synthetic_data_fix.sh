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


# Run the generation script
python src/generate_fixed_eeg.py \
  --model_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/ldm_caueeg2_4ch_116size_label_hc/final_model.pth \
  --autoencoder_path /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/outputs/aekl_eeg_4channels_label_hc/best_model.pth \
  --autoencoder_config /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_encoder_eeg.yaml \
  --diffusion_config /home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_ldm.yaml \
  --num_samples 50 \
  --label 0 \
  --batch_size 16 \
  --output_dir synthetic_hc_data
