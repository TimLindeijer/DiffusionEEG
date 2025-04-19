#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=GREEN-EEG-Transfer-CAUEEG2-to-SYNTH
#SBATCH --output=outputs/GREEN_EEG_Transfer_CAUEEG2_to_SYNTH_%j.out

# Create output directories
mkdir -p outputs
mkdir -p results/transfer_caueeg2_to_synth

# Activate environment (adjust based on your system)
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate green-env
# pip install wandb
# pip install geotorch
# pip install lightning

# Set paths
DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2"
TEST_DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/dataset/SYNTH-CAUEEG2"
OUTPUT_DIR="results/transfer_caueeg2_to_synth"
RUN_NAME="CAUEEG2_to_SYNTH_$(date +%Y%m%d_%H%M%S)"

# W&B Authentication - using API key
export WANDB_API_KEY="16232e63f53b8b502555cea8afc019f0dfc5b5ee"

# Print information about the run
echo "Starting GREEN transfer learning from CAUEEG2 to SYNTH-CAUEEG2"
echo "Run name: $RUN_NAME"
echo "Training data directory: $DATA_DIR"
echo "Testing data directory: $TEST_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "W&B enabled: Yes"

# Run the training script
python neuro-green/train_green_model.py \
    --data_dir $DATA_DIR \
    --test_data_dir $TEST_DATA_DIR \
    --use_separate_test \
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
    --wandb_tags "caueeg2" "synthetic" "transfer_learning"

# Save information about the completed job
echo "Job completed at $(date)"
echo "Results saved to $OUTPUT_DIR"

# Compress results for easy downloading
tar -czvf ${OUTPUT_DIR}_results.tar.gz $OUTPUT_DIR