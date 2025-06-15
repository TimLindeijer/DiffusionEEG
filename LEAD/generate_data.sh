#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=gen_lead_data
#SBATCH --output=outputs/gen_lead_data_%j.out
 
# Activate environment
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate lead-env

python /mnt/beegfs/home/timlin/DiffusionEEG/LEAD/src/synthetic_data_generator.py \
  --checkpoints_path "checkpoints/LEAD/diffusion/LEAD/Simplified-Diffusion-LR-0001-SS-1000-DS2-dementia/nh8_el12_dm128_df256_seed41/checkpoint.pth" \
  --samples_per_subject 1000 \
  --num_subjects 219 \
  --seq_len 128 \
  --enc_in 19 \
  --dec_in 19 \
  --c_out 19 \
  --d_model 128 \
  --n_heads 8 \
  --e_layers 12 \
  --d_layers 1 \
  --d_ff 256 \
  --patch_len_list "4" \
  --up_dim_list "76" \
  --dropout 0.1 \
  --task_name "diffusion" \
  --model "LEAD" \
  --activation "gelu" \
  --embed "timeF" \
  --num_class 3 \
  --output_dir "./dataset/SYNTH-CAUEEG2-dementia" \
  --file_format "numpy" \
  --subject_conditional \