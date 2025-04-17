#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=GREEN-EEG-Classification-Synthetic
#SBATCH --output=outputs/GREEN_EEG_Classification_Synthetic_%j.out
# Activate environment (adjust based on your system)
uenv verbose cuda-12.1.0 cudnn-12.x-9.0.0
uenv miniconda3-py38
conda activate green-env

# Use just one GPU
export CUDA_VISIBLE_DEVICES=0

# Set paths
DATASET_ROOT="/home/stud/timlin/bhome/DiffusionEEG/dataset/SYNTHETIC_CAUEEG_MULTICLASS"
FEATURE_PATH="${DATASET_ROOT}/Feature"
LABEL_PATH="${DATASET_ROOT}/Label"
CHECKPOINT_DIR="./checkpoints/green_model_$(date +'%Y%m%d_%H%M%S')"

# Create output directory
mkdir -p outputs
mkdir -p ${CHECKPOINT_DIR}

# Make sure we're in the project root directory
cd /mnt/beegfs/home/timlin/DiffusionEEG/neuro-green

# Run the classification script with fixed batching and forward method
echo "Starting GREEN EEG Classification at $(date)"
echo "Using feature path: ${FEATURE_PATH}"
echo "Using label path: ${LABEL_PATH}"
echo "Saving checkpoints to: ${CHECKPOINT_DIR}"

# Use srun as recommended by the warning message in the logs
srun python -u run_eeg_classification.py \
  --feature-path ${FEATURE_PATH} \
  --label-path ${LABEL_PATH} \
  --checkpoint-dir ${CHECKPOINT_DIR} \
  --n-freqs 8 \
  --kernel-width 0.5 \
  --sampling-freq 200 \
  --dropout 0.5 \
  --hidden-dims 64 32 \
  --bimap-dims 16 \
  --epochs 50 \
  --batch-size 16 \
  --cv-folds 5 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --max-epochs 10 \
  --pad-to-length 1000

# Save experiment summary
echo "Experiment completed at $(date)" > ${CHECKPOINT_DIR}/experiment_summary.txt
echo "Parameters:" >> ${CHECKPOINT_DIR}/experiment_summary.txt
echo "- n_freqs: 8" >> ${CHECKPOINT_DIR}/experiment_summary.txt
echo "- kernel_width: 0.5" >> ${CHECKPOINT_DIR}/experiment_summary.txt
echo "- sampling_freq: 250" >> ${CHECKPOINT_DIR}/experiment_summary.txt
echo "- dropout: 0.5" >> ${CHECKPOINT_DIR}/experiment_summary.txt
echo "- hidden_dims: 64 32" >> ${CHECKPOINT_DIR}/experiment_summary.txt
echo "- bimap_dims: 16" >> ${CHECKPOINT_DIR}/experiment_summary.txt
echo "- epochs: 50" >> ${CHECKPOINT_DIR}/experiment_summary.txt
echo "- batch_size: 16" >> ${CHECKPOINT_DIR}/experiment_summary.txt
echo "- cv_folds: 5" >> ${CHECKPOINT_DIR}/experiment_summary.txt
echo "- weight_decay: 1e-4" >> ${CHECKPOINT_DIR}/experiment_summary.txt
echo "- max_epochs: 10" >> ${CHECKPOINT_DIR}/experiment_summary.txt
echo "- pad_to_length: 1000" >> ${CHECKPOINT_DIR}/experiment_summary.txt
echo "- num_workers: 2" >> ${CHECKPOINT_DIR}/experiment_summary.txt

echo "GREEN EEG Classification job completed"