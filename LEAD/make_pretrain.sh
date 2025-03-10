#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=168:00:00
#SBATCH --job-name=BioNaturalEEG-Diffusion-PRETRAIN
#SBATCH --output=outputs/BioNaturalEEG_Diffusion_PRETRAIN.out
 
# Activate environment
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate LEADEEG

# Use just one GPU for simplicity
export CUDA_VISIBLE_DEVICES=0

# Set health filter - choose one of: "all", "hc", "mci", "dementia"
# "all" = use all data, "hc" = healthy controls only, "mci" = MCI patients only, "dementia" = dementia patients only
HEALTH_FILTER="all"

# Set model ID based on health status
MODEL_ID="BioNaturalEEG-Diffusion-ArcMargin-LR-0001"
if [ "$HEALTH_FILTER" != "all" ]; then
    MODEL_ID="${MODEL_ID}-${HEALTH_FILTER}"
fi

# Diffusion model training with subject conditioning, ArcMargin, enhanced naturalness
python -u LEAD/src/run.py --method LEAD --task_name diffusion --is_training 1 \
--root_path ./dataset/ --model_id $MODEL_ID --model LEAD --data MultiDatasets \
--training_datasets ADFTD,BrainLat-19 \
--testing_datasets ADFTD,BrainLat-19 \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 \
--seq_len 128 --enc_in 19 \
--des 'Exp' --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 20 \
--n_steps 1000 --sample_steps 200  --time_diff_constraint --init_diffusion \
--subject_conditional \
--arc_margin_s 30.0 --arc_margin_m 0.5 \
--noise_content_kl_co 1.0 --arc_subject_co 0.1 --orgth_co 2.0 \
--num_samples 546

# Uncomment to run diffusion model sampling/inference with enhanced naturalness
# python -u LEAD/src/run.py --method LEAD --task_name diffusion --is_training 0 \
# --root_path ./dataset/ --model_id $MODEL_ID --model LEAD --data MultiDatasets \
# --testing_datasets CAUEEG \
# --e_layers 12 --batch_size 16 --n_heads 8 --d_model 128 --d_ff 256 \
# --seq_len 128 --enc_in 19 \
# --des 'Exp' --n_steps 1000 --sample_steps 200 \
# --generate_samples --save_samples --samples_path ./generated_samples_${HEALTH_FILTER}/ \
# --subject_conditional --samples_per_batch 16 --num_samples 546 \
# --health_filter $HEALTH_FILTER --label_mapping "0,1,2" \
# --time_diff_constraint --init_diffusion