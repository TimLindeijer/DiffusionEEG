#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=024:00:00
#SBATCH --job-name=LEAD_diff
#SBATCH --output=outputs/p_LEAD_diff_50e_pre.out
 
# Activate environment
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate LEADEEG

# Use just one GPU for simplicity
export CUDA_VISIBLE_DEVICES=0

# Diffusion model training with simplified approach
python -u LEAD/LEAD/run.py --method LEAD --task_name diffusion --is_training 1 \
--root_path ./dataset/ --model_id D-1-CAUEEG-Diff-Simple --model LEAD --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 \
--seq_len 128 --enc_in 19 \
--des 'Exp' --itr 1 --learning_rate 0.0001 --train_epochs 50 --patience 15 \
--n_steps 1000 --sample_steps 50 --time_diff_constraint --init_diffusion

# Uncomment to run diffusion model sampling/inference
# python -u LEAD/LEAD/run.py --method LEAD --task_name diffusion --is_training 0 \
# --root_path ./dataset/ --model_id D-1-CAUEEG-Diff-Simple --model LEAD --data MultiDatasets \
# --testing_datasets CAUEEG \
# --e_layers 12 --batch_size 16 --n_heads 8 --d_model 128 --d_ff 256 \
# --seq_len 128 --enc_in 19 \
# --des 'Exp' --n_steps 1000 --sample_steps 50 --init_diffusion \
# --use_multi_gpu false