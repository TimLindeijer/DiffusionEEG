#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=168:00:00
#SBATCH --job-name=pretrain
#SBATCH --output=outputs/pretrain.out
 
# Activate environment
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate LEADEEG

# Use just one GPU for simplicity
export CUDA_VISIBLE_DEVICES=0

# Finetune only on public datasets

python -u LEAD/LEAD/run.py --method LEAD --task_name pretrain_lead --is_training 1 --root_path ./dataset/ --model_id P-11-Base --model LEAD --data MultiDatasets \
--pretraining_datasets ADSZ,APAVA-19,ADFSU,AD-Auditory,REEG-PD-19,PEARL-Neuro-19,Depression-19,REEG-SRM-19,REEG-BACA-19 \
--training_datasets ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19 \
--testing_datasets ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0002 --train_epochs 50