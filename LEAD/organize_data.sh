#!/bin/bash

# Easy script to organize multi-class synthetic data
# Save this file as organize_multiclass_data.sh and make it executable with: chmod +x organize_multiclass_data.sh

# Update these paths to match your data locations
HC_DIR="dataset/LEAD_SYNTH/synthetic_eeg_data"
MCI_DIR="dataset/LEAD_SYNTH/synthetic_eeg_data_lead_mci"
DEMENTIA_DIR="dataset/LEAD_SYNTH/synthetic_eeg_data_lead_dementia"
OUTPUT_DIR="dataset/SYNTHETIC_CAUEEG_MULTICLASS"

# Create and activate the appropriate environment if needed
# conda activate your_environment

# Run the Python script
python LEAD/organize_synthetic_data.py \
  --hc_dir "$HC_DIR" \
  --mci_dir "$MCI_DIR" \
  --dementia_dir "$DEMENTIA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --start_id 1

echo "Multi-class data organization complete!"