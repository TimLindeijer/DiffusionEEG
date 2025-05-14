#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=gen_hc_norm
#SBATCH --output=outputs/gen_synthetic_eeg_psd_normalized_caueeg2_hc.out

# Report which node we're running on
echo "Running on $(hostname)"
echo "Job started at $(date)"

# Activate environment
uenv verbose cuda-11.8.0 cudnn-11.x-8.7.0
uenv miniconda3-py39
conda activate ldm-eeg

cd Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models

# Run the generation script with PSD normalization
python src/generate_eeg.py \
  --category hc \
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
  --output_dir /home/stud/timlin/bhome/DiffusionEEG/dataset/LDM_PSD_Normalized \
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