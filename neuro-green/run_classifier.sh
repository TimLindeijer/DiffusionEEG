#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=green_caueeg2
#SBATCH --output=outputs/GREEN_EEG_Classification_CAUEEG2_300_epochs_%j.out

# Create output directories
mkdir -p outputs
mkdir -p results/caueeg2_classification_sova

# Activate environment (adjust based on your system)
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate green-env

# Set paths - using absolute paths from your config
DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/data/caueeg_bids"
OUTPUT_DIR="results/caueeg2_classification_sova"
DERIVATIVES_SUBDIR="derivatives/sovaharmony"  # Relative to DATA_DIR
RUN_NAME="CAUEEG2_300_EPOCHS_$(date +%Y%m%d_%H%M%S)"

# W&B Authentication - using API key
export WANDB_API_KEY=$(cat ~/.wandb_key)

# Print information about the run
echo "Starting GREEN training on CAUEEG2 dataset"
echo "Run name: $RUN_NAME"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "W&B enabled: Yes"
echo ""
echo "Expected data structure:"
echo "  Participants file: $DATA_DIR/participants.tsv"
echo "  Derivatives: $DATA_DIR/$DERIVATIVES_SUBDIR/"
echo "  Example EEG file: $DATA_DIR/$DERIVATIVES_SUBDIR/sub-00001/eeg/sub-00001_task-eyesClosed_desc-reject[]_eeg.fif"
echo ""

# Quick check if paths exist
if [ -f "$DATA_DIR/participants.tsv" ]; then
    echo "✓ Found participants.tsv"
else
    echo "✗ ERROR: participants.tsv not found at $DATA_DIR/participants.tsv"
fi

if [ -d "$DATA_DIR/$DERIVATIVES_SUBDIR" ]; then
    echo "✓ Found derivatives directory"
    echo "  Number of subject folders: $(ls -d $DATA_DIR/$DERIVATIVES_SUBDIR/sub-* 2>/dev/null | wc -l)"
else
    echo "✗ ERROR: derivatives directory not found at $DATA_DIR/$DERIVATIVES_SUBDIR"
fi

echo ""

echo -e "\n\n"
echo "=========================================="
echo "Starting training..."
echo "=========================================="

# Run the training script with parameters from config
python neuro-green/caueeg_green_classifier.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --derivatives_subdir "$DERIVATIVES_SUBDIR" \
    --n_freqs 10 \
    --kernel_width_s 0.5 \
    --hidden_dim 32 16 \
    --dropout 0.5 \
    --bi_out 10 \
    --logref "logeuclid" \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --max_epochs 300 \
    --num_workers 4 \
    --n_epochs_per_subject 20 \
    --n_splits 5 \
    --seed 42 \
    --detailed_evaluation \
    --use_wandb \
    --wandb_project "green-caueeg" \
    --wandb_name "$RUN_NAME" \
    --wandb_tags "caueeg2" "300epochs" "sovaharmony"

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Training failed with error code: $?"
fi

# Print final summary
echo "Job completed at: $(date)"