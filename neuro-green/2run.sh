#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=GREEN-EEG-Classification-CAUEEG2
#SBATCH --output=outputs/GREEN_EEG_Classification_CAUEEG2_%j.out

# Create output directories
mkdir -p outputs
mkdir -p results/caueeg2_classification

# Activate environment (adjust based on your system)
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate green-env
pip install seaborn
# Set paths
DATA_DIR="/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2"
OUTPUT_DIR="results/caueeg2_classification"

# Run the training script
python neuro-green/train_green_model.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size 32 \
    --learning_rate 0.0003 \
    --weight_decay 1e-5 \
    --max_epochs 200 \
    --patience 20 \
    --n_freqs 8 \
    --kernel_width_s 0.5 \
    --hidden_dim 64 32 \
    --dropout 0.5 \
    --num_workers 4 \
    --detailed_evaluation \
    --sfreq 200 \
    --seed 42

# Save information about the completed job
echo "Job completed at $(date)"
echo "Results saved to $OUTPUT_DIR"

# Compress results for easy downloading
tar -czvf ${OUTPUT_DIR}_results.tar.gz $OUTPUT_DIR