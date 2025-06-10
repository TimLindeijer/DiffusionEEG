#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=100pct_augmented_classification
#SBATCH --output=outputs/100pct_augmented_classification_%j.out

# =====================================
# CONFIGURATION - CHANGE THESE VALUES
# =====================================
# Set the combination you want to run (True/False)
SHUFFLE=true
SHUFFLE_FIRST_EPOCH=false
RANDOMIZE_EPOCHS=false

# =====================================
# AUTOMATIC CONFIGURATION
# =====================================
# Convert boolean values to T/F for naming
SHUFFLE_STR=$([ "$SHUFFLE" = true ] && echo "T" || echo "F")
SHUFFLE_FIRST_STR=$([ "$SHUFFLE_FIRST_EPOCH" = true ] && echo "T" || echo "F")
RANDOMIZE_STR=$([ "$RANDOMIZE_EPOCHS" = true ] && echo "T" || echo "F")

# Create combination string (e.g., "TFT")
COMBINATION="${SHUFFLE_STR}${SHUFFLE_FIRST_STR}${RANDOMIZE_STR}"

# Update job name to include combination

# Set paths
DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/dataset/lead_norm_ready_datasets/train_augmented_100pct"
TEST_DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/dataset/lead_norm_ready_datasets/test_genuine"

# Create output directory name with combination
OUTPUT_DIR="results/lead_norm_ready_datasets${COMBINATION}"
mkdir -p $OUTPUT_DIR

# Create run name with combination and timestamp
RUN_NAME="lead_norm_ready_datasets${COMBINATION}_$(date +%Y%m%d_%H%M%S)"

# Activate environment (adjust based on your system)
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate green-env

# W&B Authentication - using API key from file
export WANDB_API_KEY=$(cat ~/.wandb_key)

# Print information about the run
echo "========================================"
echo "Starting GREEN training"
echo "========================================"
echo "Combination: $COMBINATION"
echo "  shuffle: $SHUFFLE"
echo "  shuffle_first_epoch: $SHUFFLE_FIRST_EPOCH"
echo "  randomize_epochs: $RANDOMIZE_EPOCHS"
echo "Run name: $RUN_NAME"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "W&B enabled: Yes"
echo "========================================"

# Build the command with conditional flags
CMD_ARGS="--data_dir $DATA_DIR \
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
    --wandb_project green-diff-shuffle2 \
    --wandb_name $RUN_NAME \
    --wandb_tags caueeg2 genuine shuffle_${SHUFFLE_STR} first_epoch_${SHUFFLE_FIRST_STR} randomize_${RANDOMIZE_STR}"

# Add shuffle flags conditionally
if [ "$SHUFFLE" = true ]; then
    CMD_ARGS="$CMD_ARGS --shuffle"
fi

if [ "$SHUFFLE_FIRST_EPOCH" = true ]; then
    CMD_ARGS="$CMD_ARGS --shuffle_first_epoch"
fi

if [ "$RANDOMIZE_EPOCHS" = true ]; then
    CMD_ARGS="$CMD_ARGS --randomize_epochs"
fi

# Run the training script
python neuro-green/train_green_model.py $CMD_ARGS

# Save information about the completed job
echo "========================================"
echo "Job completed at $(date)"
echo "Combination: $COMBINATION"
echo "Results saved to $OUTPUT_DIR"
echo "========================================"

# Compress results for easy downloading
tar -czvf ${OUTPUT_DIR}_results.tar.gz $OUTPUT_DIR