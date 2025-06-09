#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=bootstrap_augmented_classification
#SBATCH --output=outputs/bootstrap_augmented_classification_%j.out

# =====================================
# USAGE EXAMPLES
# =====================================
# Run with different percentages:
#   sbatch neuro-green/run_neuro_bootstrap.sh 20    → uses train_augmented_20pct,  output: bootstrap_augmented_20pct_TFF_20runs
#   sbatch neuro-green/run_neuro_bootstrap.sh 40    → uses train_augmented_40pct,  output: bootstrap_augmented_40pct_TFF_20runs  
#   sbatch neuro-green/run_neuro_bootstrap.sh 60    → uses train_augmented_60pct,  output: bootstrap_augmented_60pct_TFF_20runs
#   sbatch neuro-green/run_neuro_bootstrap.sh 80    → uses train_augmented_80pct,  output: bootstrap_augmented_80pct_TFF_20runs
#   sbatch neuro-green/run_neuro_bootstrap.sh 100   → uses train_augmented_100pct, output: bootstrap_augmented_100pct_TFF_20runs
#   sbatch neuro-green/run_neuro_bootstrap.sh       → uses train_augmented_100pct (default)
#
# To run all percentages:
#   for pct in 20 40 60 80 100; do sbatch neuro-green/run_neuro_bootstrap.sh $pct; done

# =====================================
# COMMAND LINE ARGUMENT PARSING
# =====================================
# Get percentage from command line argument
PERCENTAGE=${1:-100}  # Default to 100% if no argument provided

# Validate percentage
if [[ ! "$PERCENTAGE" =~ ^[0-9]+$ ]] || [ "$PERCENTAGE" -lt 1 ] || [ "$PERCENTAGE" -gt 100 ]; then
    echo "Error: Invalid percentage '$PERCENTAGE'. Must be a number between 1 and 100."
    echo "Usage: sbatch $0 [PERCENTAGE]"
    echo "Examples: sbatch $0 20, sbatch $0 40, sbatch $0 60, sbatch $0 80, sbatch $0 100"
    exit 1
fi

# =====================================
# CONFIGURATION - CHANGE BASE PATHS HERE
# =====================================
# Base data paths - the percentage will be automatically inserted
BASE_DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/dataset/pure_ldm_norm_fix_spec_ready_datasets"

# Construct training data directory with percentage
DATA_DIR="$BASE_DATA_DIR/train_augmented_${PERCENTAGE}pct"
TEST_DATA_DIR="$BASE_DATA_DIR/test_genuine"

# Debug: Print what we're constructing
echo "DEBUG: Constructing paths with PERCENTAGE=$PERCENTAGE"
echo "  BASE_DATA_DIR: $BASE_DATA_DIR"
echo "  Constructed DATA_DIR: $DATA_DIR"
echo "  TEST_DATA_DIR: $TEST_DATA_DIR"

# Verify that the data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Training data directory does not exist: $DATA_DIR"
    echo "Available directories in $BASE_DATA_DIR:"
    ls -la "$BASE_DATA_DIR" | grep "train_augmented" || echo "No train_augmented directories found"
    exit 1
fi

if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "Error: Test data directory does not exist: $TEST_DATA_DIR"
    exit 1
fi

echo "SUCCESS: Both directories exist!"
echo "  Training: $DATA_DIR"
echo "  Testing: $TEST_DATA_DIR"

# Set the combination you want to run (True/False)
SHUFFLE=true
SHUFFLE_FIRST_EPOCH=false
RANDOMIZE_EPOCHS=false

# Bootstrap configuration
N_BOOTSTRAP=20                    # Number of bootstrap samples
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

# =====================================
# AUTOMATIC NAMING FROM DATA DIRECTORY
# =====================================
# Extract dataset name from DATA_DIR path for automatic naming
# Examples:
#   /path/to/train_augmented_100pct → train_augmented_100pct
#   /path/to/dataset_name/train_data → dataset_name_train_data
#   /path/to/my_experiment → my_experiment
TRAIN_DIR_NAME=$(basename "$DATA_DIR")                    # e.g., "train_augmented_100pct"
PARENT_DIR_NAME=$(basename "$(dirname "$DATA_DIR")")      # e.g., "pure_ldm_norm_fix_spec_ready_datasets"

# Create a clean dataset identifier
if [[ "$TRAIN_DIR_NAME" == *"train"* ]]; then
    # If it contains "train", use both parent and train dir names
    DATASET_NAME="${PARENT_DIR_NAME}_${TRAIN_DIR_NAME}"
else
    # Otherwise, use train dir name and check parent
    if [[ "$PARENT_DIR_NAME" != "dataset" && "$PARENT_DIR_NAME" != "data" ]]; then
        DATASET_NAME="${PARENT_DIR_NAME}_${TRAIN_DIR_NAME}"
    else
        DATASET_NAME="$TRAIN_DIR_NAME"
    fi
fi

# Clean up the dataset name (remove common suffixes/prefixes, limit length)
DATASET_NAME=$(echo "$DATASET_NAME" | sed 's/_datasets//g' | sed 's/ready_//g')

# Create output directory name with auto-generated dataset name
# Format: results/bootstrap_{DATASET_NAME}_{COMBINATION}_{N_BOOTSTRAP}runs
OUTPUT_DIR="results/bootstrap_${DATASET_NAME}_${COMBINATION}_${N_BOOTSTRAP}runs"
mkdir -p $OUTPUT_DIR

# Create run name with auto-generated dataset name
# Format: bootstrap_{DATASET_NAME}_{COMBINATION}_{N_BOOTSTRAP}runs_{TIMESTAMP}
RUN_NAME="bootstrap_${DATASET_NAME}_${COMBINATION}_${N_BOOTSTRAP}runs_$(date +%Y%m%d_%H%M%S)"

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
echo "Dataset: ${PERCENTAGE}% augmented data"
echo "  Dataset name: $DATASET_NAME"
echo "  Training dir: $DATA_DIR"
echo "  Test dir: $TEST_DATA_DIR"
echo "Shuffle combination: $COMBINATION"
echo "  shuffle: $SHUFFLE"
echo "  shuffle_first_epoch: $SHUFFLE_FIRST_EPOCH"
echo "  randomize_epochs: $RANDOMIZE_EPOCHS"
echo "Bootstrap Configuration:"
echo "  n_bootstrap: $N_BOOTSTRAP"
echo "  sample_ratio: $BOOTSTRAP_SAMPLE_RATIO"
echo "Run details:"
echo "  Run name: $RUN_NAME"
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
    --wandb_tags caueeg2 genuine bootstrap ${PERCENTAGE}pct shuffle_${SHUFFLE_STR} first_epoch_${SHUFFLE_FIRST_STR} randomize_${RANDOMIZE_STR} ${N_BOOTSTRAP}runs"

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
    echo "Dataset: ${PERCENTAGE}% augmented data"
    echo "  Dataset name: $DATASET_NAME"
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
    echo "Dataset: ${PERCENTAGE}% augmented data"
    echo "Check the log file for details"
    echo "========================================"
    exit 1
fi

# =====================================
# HELPER: To run all percentages at once, use:
# for pct in 20 40 60 80 100; do sbatch neuro-green/run_neuro_bootstrap.sh $pct; done
# =====================================