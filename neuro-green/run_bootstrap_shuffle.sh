#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=bootstrap_mlr-gen_shuffle
#SBATCH --output=outputs/bootstrap_mlr_gen_shuffle_%j.out

# =====================================
# CONFIGURATION - CHANGE THESE VALUES
# =====================================
# Set the combination you want to run (True/False)
SHUFFLE=true
SHUFFLE_FIRST_EPOCH=false
RANDOMIZE_EPOCHS=false

# Bootstrap configuration
N_BOOTSTRAP=10                    # Number of bootstrap samples
BOOTSTRAP_SAMPLE_RATIO=1.0       # Ratio of original training size (1.0 = same size with replacement)

# =====================================
# AUTOMATIC CONFIGURATION
# =====================================
# Convert boolean values to T/F for naming
SHUFFLE_STR=$([ "$SHUFFLE" = true ] && echo "T" || echo "F")
SHUFFLE_FIRST_STR=$([ "$SHUFFLE_FIRST_EPOCH" = true ] && echo "T" || echo "F")
RANDOMIZE_STR=$([ "$RANDOMIZE_EPOCHS" = true ] && echo "T" || echo "F")

# Create combination string (e.g., "TFT")
COMBINATION="${SHUFFLE_STR}${SHUFFLE_FIRST_STR}${RANDOMIZE_STR}"

# Set paths
DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/dataset/ldm_norm_fix_ready_datasets/train_genuine"
TEST_DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/dataset/ldm_norm_fix_ready_datasets/test_genuine"

# Create output directory name with combination and bootstrap info
OUTPUT_DIR="results/bootstrap_mlr_gen_shuffle_${COMBINATION}_${N_BOOTSTRAP}runs"
mkdir -p $OUTPUT_DIR

# Create run name with combination, bootstrap info, and timestamp
RUN_NAME="BOOTSTRAP_MLR_GENUINE_${COMBINATION}_${N_BOOTSTRAP}runs_$(date +%Y%m%d_%H%M%S)"

# Activate environment (adjust based on your system)
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate green-env

# W&B Authentication - using API key from file
export WANDB_API_KEY=$(cat ~/.wandb_key)

# Print information about the run
echo "========================================"
echo "Starting GREEN Bootstrap Training"
echo "========================================"
echo "Shuffle combination: $COMBINATION"
echo "  shuffle: $SHUFFLE"
echo "  shuffle_first_epoch: $SHUFFLE_FIRST_EPOCH"
echo "  randomize_epochs: $RANDOMIZE_EPOCHS"
echo "Bootstrap Configuration:"
echo "  n_bootstrap: $N_BOOTSTRAP"
echo "  sample_ratio: $BOOTSTRAP_SAMPLE_RATIO"
echo "Run details:"
echo "  Run name: $RUN_NAME"
echo "  Data directory: $DATA_DIR"
echo "  Test directory: $TEST_DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  W&B enabled: Yes"
echo "  Estimated time: $(echo "$N_BOOTSTRAP * 30" | bc) minutes (approx)"
echo "========================================"

# Build the command with conditional flags
CMD_ARGS="--data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --test_data_dir $TEST_DATA_DIR \
    --use_separate_test \
    --n_bootstrap $N_BOOTSTRAP \
    --bootstrap_sample_ratio $BOOTSTRAP_SAMPLE_RATIO \
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
    --wandb_project green-diff-shuffle-bootstrap \
    --wandb_name $RUN_NAME \
    --wandb_tags caueeg2 genuine bootstrap shuffle_${SHUFFLE_STR} first_epoch_${SHUFFLE_FIRST_STR} randomize_${RANDOMIZE_STR} ${N_BOOTSTRAP}runs"

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

# Print the command for debugging
echo "Running command:"
echo "python neuro-green/train_neuro_bootstrap.py $CMD_ARGS"
echo "========================================"

# Run the bootstrap training script
python neuro-green/train_neuro_bootstrap.py $CMD_ARGS

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Bootstrap training completed successfully at $(date)"
    echo "Shuffle combination: $COMBINATION"
    echo "Bootstrap runs: $N_BOOTSTRAP"
    echo "Results saved to $OUTPUT_DIR"
    echo "========================================"
    
    # Print summary of results if available
    if [ -f "$OUTPUT_DIR/bootstrap_statistics.csv" ]; then
        echo "Bootstrap Statistics Summary:"
        echo "========================================"
        head -n 10 "$OUTPUT_DIR/bootstrap_statistics.csv"
        echo "========================================"
    fi
    
    # Compress results for easy downloading
    echo "Compressing results..."
    tar -czvf ${OUTPUT_DIR}_results.tar.gz $OUTPUT_DIR
    echo "Results compressed to ${OUTPUT_DIR}_results.tar.gz"
    
else
    echo "========================================"
    echo "Bootstrap training failed at $(date)"
    echo "Check the log file for details"
    echo "========================================"
    exit 1
fi

# =====================================
# CONFIGURATION TESTING HELPER
# =====================================
# To test all shuffle combinations with bootstrap, create separate scripts or modify the top:
#
# Test all 8 combinations:
# COMBINATION 1: FFF (no shuffling)
# COMBINATION 2: TFF (shuffle only)  
# COMBINATION 3: FTF (shuffle_first_epoch only)
# COMBINATION 4: FFT (randomize_epochs only)
# COMBINATION 5: TTF (shuffle + shuffle_first_epoch)
# COMBINATION 6: TFT (shuffle + randomize_epochs) ← YOUR CURRENT CONFIG
# COMBINATION 7: FTT (shuffle_first_epoch + randomize_epochs)
# COMBINATION 8: TTT (all three enabled)
#
# To run multiple combinations:
# for combo in "FFF" "TFF" "FTF" "FFT" "TTF" "TFT" "FTT" "TTT"; do
#   # Set SHUFFLE, SHUFFLE_FIRST_EPOCH, RANDOMIZE_EPOCHS based on combo
#   # Submit job
# done
# =====================================