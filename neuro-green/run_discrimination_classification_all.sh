#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=discrim
#SBATCH --output=outputs/discrimination_%j.out

# Activate environment (adjust based on your system)
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate green-env

# Set paths - change these to your specific discrimination dataset paths
DISCRIMINATION_TYPE="all"  # Options: hc, mci, dementia, all

# Build paths based on discrimination type
DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/dataset/discrimination_datasets/${DISCRIMINATION_TYPE}_train"
TEST_DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/dataset/discrimination_datasets/${DISCRIMINATION_TYPE}_val"
OUTPUT_DIR="results/discrimination_${DISCRIMINATION_TYPE}"
RUN_NAME="Discrimination_${DISCRIMINATION_TYPE}_$(date +%Y%m%d_%H%M%S)"

# W&B Authentication - using API key from file
export WANDB_API_KEY=$(cat ~/.wandb_key)

# Print information about the run
echo "Starting GREEN training for genuine vs synthetic discrimination task: ${DISCRIMINATION_TYPE}"
echo "Run name: $RUN_NAME"
echo "Train data directory: $DATA_DIR"
echo "Test data directory: $TEST_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "W&B enabled: Yes"

# Run the training script
python neuro-green/train_green_model.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --test_data_dir $TEST_DATA_DIR \
    --use_separate_test \
    --batch_size 16 \
    --learning_rate 0.0003 \
    --weight_decay 1e-5 \
    --max_epochs 300 \
    --n_freqs 8 \
    --kernel_width_s 0.5 \
    --hidden_dim 64 32 \
    --dropout 0.5 \
    --num_workers 4 \
    --detailed_evaluation \
    --sfreq 200 \
    --seed 42 \
    --use_wandb \
    --wandb_project "green-discrimination" \
    --wandb_name "$RUN_NAME" \
    --wandb_tags "discrimination" "${DISCRIMINATION_TYPE}"

# Save information about the completed job
echo "Job completed at $(date)"
echo "Results saved to $OUTPUT_DIR"

# Compress results for easy downloading
tar -czvf ${OUTPUT_DIR}_results.tar.gz $OUTPUT_DIR