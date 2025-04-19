#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=GREEN-EEG-Mixed-Data
#SBATCH --output=outputs/GREEN_EEG_Mixed_%j.out

# ===== CONFIGURABLE PARAMETERS =====
# Set the percentage of real data (e.g., 90 for 90% real, 10% synthetic)
REAL_PERCENT=90
# ===================================

# Create output directories
mkdir -p outputs
mkdir -p results/mixed_${REAL_PERCENT}_$(( 100 - REAL_PERCENT ))_training

# Activate environment (adjust based on your system)
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate green-env
# pip install wandb
# pip install geotorch
# pip install lightning

# Set paths
REAL_DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2"
SYNTH_DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/dataset/SYNTH-CAUEEG2"
OUTPUT_DIR="results/mixed_${REAL_PERCENT}_$(( 100 - REAL_PERCENT ))_training"
RUN_NAME="MIXED_${REAL_PERCENT}_$(( 100 - REAL_PERCENT ))_$(date +%Y%m%d_%H%M%S)"

echo "Creating mixed dataset with ${REAL_PERCENT}% real data and $(( 100 - REAL_PERCENT ))% synthetic data"

# Create a temporary directory to combine the datasets
TEMP_DIR="temp_mixed_dataset_${REAL_PERCENT}_$(( 100 - REAL_PERCENT ))"
mkdir -p $TEMP_DIR
mkdir -p $TEMP_DIR/Feature
mkdir -p $TEMP_DIR/Label

# Count available real and synthetic samples
AVAILABLE_REAL_COUNT=$(ls $REAL_DATA_DIR/Feature/feature_*.npy | wc -l)
AVAILABLE_SYNTH_COUNT=$(ls $SYNTH_DATA_DIR/Feature/feature_*.npy | wc -l)
echo "Available real samples: $AVAILABLE_REAL_COUNT"
echo "Available synthetic samples: $AVAILABLE_SYNTH_COUNT"

# Calculate how many samples of each type to use
if [ $REAL_PERCENT -eq 100 ]; then
    # 100% real data, 0% synthetic
    REAL_TO_USE=$AVAILABLE_REAL_COUNT
    SYNTH_TO_USE=0
elif [ $REAL_PERCENT -eq 0 ]; then
    # 0% real data, 100% synthetic
    REAL_TO_USE=0
    SYNTH_TO_USE=$AVAILABLE_SYNTH_COUNT
else
    # We want a specific ratio
    # For example, if we want 90/10 and we have 1000 real samples and 500 synthetic,
    # we'll use 900 real and 100 synthetic to maintain the ratio
    
    # First figure out if we're limited by real or synthetic data
    MAX_POSSIBLE_REAL=$(echo "scale=0; $AVAILABLE_REAL_COUNT * 100 / $REAL_PERCENT" | bc)
    MAX_POSSIBLE_SYNTH=$(echo "scale=0; $AVAILABLE_SYNTH_COUNT * 100 / (100 - $REAL_PERCENT)" | bc)
    
    if [ $MAX_POSSIBLE_REAL -lt $MAX_POSSIBLE_SYNTH ]; then
        # Limited by real data
        REAL_TO_USE=$AVAILABLE_REAL_COUNT
        SYNTH_TO_USE=$(echo "scale=0; $REAL_TO_USE * (100 - $REAL_PERCENT) / $REAL_PERCENT" | bc)
    else
        # Limited by synthetic data
        SYNTH_TO_USE=$AVAILABLE_SYNTH_COUNT
        REAL_TO_USE=$(echo "scale=0; $SYNTH_TO_USE * $REAL_PERCENT / (100 - $REAL_PERCENT)" | bc)
    fi
    
    # Ensure we don't exceed available counts
    if [ $REAL_TO_USE -gt $AVAILABLE_REAL_COUNT ]; then
        REAL_TO_USE=$AVAILABLE_REAL_COUNT
    fi
    
    if [ $SYNTH_TO_USE -gt $AVAILABLE_SYNTH_COUNT ]; then
        SYNTH_TO_USE=$AVAILABLE_SYNTH_COUNT
    fi
fi

echo "Using $REAL_TO_USE real samples and $SYNTH_TO_USE synthetic samples"
echo "Final ratio: $(echo "scale=1; $REAL_TO_USE * 100 / ($REAL_TO_USE + $SYNTH_TO_USE)" | bc)% real, $(echo "scale=1; $SYNTH_TO_USE * 100 / ($REAL_TO_USE + $SYNTH_TO_USE)" | bc)% synthetic"

# Get real samples
echo "Selecting and copying real samples..."
REAL_SAMPLES=($(ls $REAL_DATA_DIR/Feature/feature_*.npy))
if [ $REAL_TO_USE -lt $AVAILABLE_REAL_COUNT ]; then
    # Randomly select subset of real samples
    for i in $(shuf -i 0-$((AVAILABLE_REAL_COUNT-1)) -n $REAL_TO_USE); do
        SAMPLE=${REAL_SAMPLES[$i]}
        cp $SAMPLE $TEMP_DIR/Feature/
    done
else
    # Use all real samples
    cp $REAL_DATA_DIR/Feature/feature_*.npy $TEMP_DIR/Feature/
fi

# Get synthetic samples
if [ $SYNTH_TO_USE -gt 0 ]; then
    echo "Selecting and copying synthetic samples..."
    SYNTH_SAMPLES=($(ls $SYNTH_DATA_DIR/Feature/feature_*.npy))
    
    for i in $(shuf -i 0-$((AVAILABLE_SYNTH_COUNT-1)) -n $SYNTH_TO_USE); do
        SAMPLE=${SYNTH_SAMPLES[$i]}
        SAMPLE_BASENAME=$(basename $SAMPLE)
        # Add "synth_" prefix to avoid name conflicts
        cp $SAMPLE $TEMP_DIR/Feature/synth_$SAMPLE_BASENAME
    done
fi

# Copy label files
cp $REAL_DATA_DIR/Label/label.npy $TEMP_DIR/Label/

if [ $SYNTH_TO_USE -gt 0 ]; then
    cp $SYNTH_DATA_DIR/Label/label.npy $TEMP_DIR/Label/synth_label.npy
fi

# Create script to combine labels
cat > $TEMP_DIR/combine_labels.py << 'EOF'
import numpy as np
import os
import sys

# Load real labels
real_labels_file = 'Label/label.npy'
if os.path.exists(real_labels_file):
    real_labels = np.load(real_labels_file)
else:
    real_labels = np.empty((0, 2), dtype=int)

# Look for real samples actually present (in case we're using a subset)
real_files = [f for f in os.listdir('Feature') if f.startswith('feature_')]
real_ids = [int(f.split('_')[1].split('.')[0]) for f in real_files]

# Filter real labels to only include present files
filtered_real_labels = []
for row in real_labels:
    if row[1] in real_ids:
        filtered_real_labels.append(row)
if filtered_real_labels:
    filtered_real_labels = np.array(filtered_real_labels)
else:
    filtered_real_labels = np.empty((0, 2), dtype=int)

# Process synthetic labels if they exist
synth_labels_file = 'Label/synth_label.npy'
if os.path.exists(synth_labels_file):
    synth_labels = np.load(synth_labels_file)
    
    # Get the synth sample IDs from the filenames
    synth_files = [f for f in os.listdir('Feature') if f.startswith('synth_feature_')]
    synth_ids = [int(f.split('_')[2].split('.')[0]) for f in synth_files]

    # Find these IDs in the synth_labels file and create a new array
    selected_synth_labels = []
    
    # Get the highest ID from real data to avoid conflicts
    max_id = 0
    if len(filtered_real_labels) > 0:
        max_id = np.max(filtered_real_labels[:, 1])

    for synth_id in synth_ids:
        for row in synth_labels:
            if row[1] == synth_id:
                # Assign a new ID to the synthetic sample to avoid conflicts
                new_id = max_id + 1
                max_id += 1
                selected_synth_labels.append([row[0], new_id])
                
                # Also rename the feature file to match the new ID
                old_name = f"Feature/synth_feature_{synth_id}.npy"
                new_name = f"Feature/feature_{new_id}.npy"
                if os.path.exists(old_name):
                    os.rename(old_name, new_name)

    # Convert to numpy array
    if selected_synth_labels:
        selected_synth_labels = np.array(selected_synth_labels)
    else:
        selected_synth_labels = np.empty((0, 2), dtype=int)
    
    # Combine real and synthetic labels
    if len(filtered_real_labels) > 0 and len(selected_synth_labels) > 0:
        combined_labels = np.vstack((filtered_real_labels, selected_synth_labels))
    elif len(filtered_real_labels) > 0:
        combined_labels = filtered_real_labels
    else:
        combined_labels = selected_synth_labels
else:
    # No synthetic data, just use the filtered real labels
    combined_labels = filtered_real_labels

# Save the combined labels
np.save('Label/label.npy', combined_labels)

# Print statistics
print(f"Combined dataset statistics:")
real_count = len(filtered_real_labels) if len(filtered_real_labels.shape) > 1 else 0
synth_count = len(combined_labels) - real_count if len(combined_labels.shape) > 1 else 0
total_count = len(combined_labels) if len(combined_labels.shape) > 1 else 0

print(f"  Real samples: {real_count}")
print(f"  Synthetic samples: {synth_count}")
print(f"  Total samples: {total_count}")
print(f"  Ratio: {real_count/total_count*100:.1f}% real, {synth_count/total_count*100:.1f}% synthetic")

# Count class distribution
if total_count > 0:
    unique, counts = np.unique(combined_labels[:, 0], return_counts=True)
    print(f"  Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"    Class {cls}: {count} samples ({count/total_count*100:.1f}%)")

# Cleanup
if os.path.exists('Label/synth_label.npy'):
    os.remove('Label/synth_label.npy')
EOF

# Run the label combination script
cd $TEMP_DIR
python combine_labels.py
cd ..

# W&B Authentication - using API key
export WANDB_API_KEY="16232e63f53b8b502555cea8afc019f0dfc5b5ee"

# Print information about the run
echo "Starting GREEN training on mixed dataset"
echo "Run name: $RUN_NAME"
echo "Mixed data directory: $TEMP_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "W&B enabled: Yes"

# Run the training script
python neuro-green/train_green_model.py \
    --data_dir $TEMP_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size 32 \
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
    --wandb_project "green-caueeg" \
    --wandb_name "$RUN_NAME" \
    --wandb_tags "caueeg2" "synthetic" "mixed_data" "ratio_${REAL_PERCENT}_$(( 100 - REAL_PERCENT ))"

# Save information about the completed job
echo "Job completed at $(date)"
echo "Results saved to $OUTPUT_DIR"

# Compress results for easy downloading
tar -czvf ${OUTPUT_DIR}_results.tar.gz $OUTPUT_DIR

# Clean up temporary directory
echo "Cleaning up temporary dataset..."
rm -rf $TEMP_DIR