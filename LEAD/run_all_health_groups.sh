#!/bin/bash
# Script to run diffusion model for all health groups

# Activate environment
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate LEADEEG

# Use just one GPU for simplicity
export CUDA_VISIBLE_DEVICES=0

# Base command for training
BASE_TRAIN_CMD="python -u LEAD/LEAD/run.py --method LEAD --task_name diffusion --is_training 1 \
--root_path ./dataset/ --model LEAD --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--e_layers 12 --batch_size 32 --n_heads 8 --d_model 128 --d_ff 256 \
--seq_len 128 --enc_in 19 \
--des 'Exp' --itr 1 --learning_rate 0.0001 --train_epochs 50 --patience 15 \
--n_steps 1000 --sample_steps 50 --time_diff_constraint --init_diffusion \
--subject_conditional --num_subjects 15 \
--arc_margin_s 30.0 --arc_margin_m 0.5 \
--noise_content_kl_co 1.0 --arc_subject_co 0.1 --orgth_co 2.0 \
--label_mapping \"0,1,2\""

# Base command for sampling
BASE_SAMPLE_CMD="python -u LEAD/LEAD/run.py --method LEAD --task_name diffusion --is_training 0 \
--root_path ./dataset/ --model LEAD --data MultiDatasets \
--testing_datasets CAUEEG \
--e_layers 12 --batch_size 16 --n_heads 8 --d_model 128 --d_ff 256 \
--seq_len 128 --enc_in 19 \
--des 'Exp' --n_steps 1000 --sample_steps 50 --init_diffusion \
--generate_samples --save_samples \
--subject_conditional --samples_per_batch 16 \
--use_multi_gpu false \
--label_mapping \"0,1,2\""

# Health groups to process
HEALTH_GROUPS=("hc" "mci" "dementia")
GROUP_NAMES=("HealthyControls" "MCI" "Dementia")

# Process each health group
for i in "${!HEALTH_GROUPS[@]}"; do
    HEALTH_FILTER="${HEALTH_GROUPS[$i]}"
    GROUP_NAME="${GROUP_NAMES[$i]}"
    MODEL_ID="D-2-CAUEEG-Diff-${GROUP_NAME}"
    
    echo "========================================="
    echo "Processing health group: ${GROUP_NAME}"
    echo "========================================="
    
    # Train model for this health group
    TRAIN_CMD="${BASE_TRAIN_CMD} --model_id ${MODEL_ID} --health_filter ${HEALTH_FILTER}"
    echo "Running training command:"
    echo ${TRAIN_CMD}
    eval ${TRAIN_CMD}
    
    # Generate samples for this health group
    SAMPLE_CMD="${BASE_SAMPLE_CMD} --model_id ${MODEL_ID} --health_filter ${HEALTH_FILTER} --samples_path ./generated_samples_${HEALTH_FILTER}/"
    echo "Running sampling command:"
    echo ${SAMPLE_CMD}
    eval ${SAMPLE_CMD}
    
    echo "Completed processing for health group: ${GROUP_NAME}"
    echo ""
done

echo "All health groups processed successfully!"