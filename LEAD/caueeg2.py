import os
import numpy as np
import mne
import pandas as pd
import glob
import traceback

# Save channel information
SAVE_CHANNEL_INFO = True

# ---------------------------
# Configuration and Constants
# ---------------------------
# Standard set of 19 EEG channels (10-20 system)
STANDARD_CHANNELS = ['T6', 'T5', 'CZ', 'PZ', 'P3', 'T3', 'O2', 'C4', 'P4', 'FP2', 'F8', 'F3', 'O1', 'FP1', 'T4', 'FZ', 'F7', 'F4', 'C3']

# Root paths for input and output data
root = '/home/stud/timlin/bhome/DiffusionEEG/data/caueeg_bids'
derivatives_path = os.path.join(root, 'derivatives', 'sovaharmony')
participants_path = os.path.join(root, 'participants.tsv')
output_root = '/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2'
label_path = os.path.join(output_root, 'Label')
feature_path = os.path.join(output_root, 'Feature')
channel_info_dir = os.path.join(output_root, 'ChannelInfo')

# Create output directories
os.makedirs(output_root, exist_ok=True)
os.makedirs(label_path, exist_ok=True)
os.makedirs(feature_path, exist_ok=True)
if SAVE_CHANNEL_INFO:
    os.makedirs(channel_info_dir, exist_ok=True)

# ---------------------------
# Participants Labeling
# ---------------------------
participants = pd.read_csv(participants_path, sep='\t')

# Initialize labels with -1
participants['label'] = -1
for idx, row in participants.iterrows():
    # Assuming the 5th column (index 4) contains diagnosis info
    diagnosis = row[4]
    if pd.isna(diagnosis):
        continue
    diagnosis_str = str(diagnosis).lower()
    if 'hc (+smc)' in diagnosis_str:
        participants.at[idx, 'label'] = 0
    elif 'mci' in diagnosis_str:
        participants.at[idx, 'label'] = 1
    elif 'dementia' in diagnosis_str:
        participants.at[idx, 'label'] = 2

# Filter valid participants and assign a sequential subject_id
participants_valid = participants[participants['label'].isin([0, 1, 2])].copy()
participants_valid['subject_id'] = range(1, len(participants_valid) + 1)

# Save labels array
labels = participants_valid[['label', 'subject_id']].to_numpy(dtype='int32')
np.save(os.path.join(label_path, 'label.npy'), labels)

# Mapping from original participant IDs to new subject_ids
subject_id_map = dict(zip(participants_valid.iloc[:, 0], participants_valid['subject_id']))
valid_subjects = participants_valid.iloc[:, 0].tolist()

def process_epochs_data(epochs, sub_id, fif_path, file_suffix=""):
    """Process epochs data and save it to a file"""
    # Print channel information
    print(f"  Available channels ({len(epochs.ch_names)}):")
    print(f"  {epochs.ch_names}")
    
    # Print information about the data
    print(f"  Epochs info: {len(epochs)} epochs, {epochs.times.shape[0]} time points per epoch")
    print(f"  Sampling rate: {epochs.info['sfreq']} Hz")
    
    # Get data from all epochs (epochs, channels, times)
    epochs_data = epochs.get_data(copy=True)
    
    # Transpose to (epochs, times, channels)
    # Original: (epochs, channels, times) -> Transpose to: (epochs, times, channels)
    transposed_data = np.transpose(epochs_data, (0, 2, 1))
    
    # Define the output filename
    output_filename = os.path.join(feature_path, f'feature_{sub_id:02d}{file_suffix}.npy')
    
    # Save the data
    print(f"  Saving data with shape {transposed_data.shape} to {output_filename}")
    np.save(output_filename, transposed_data)
    print(f"  Data successfully saved to {output_filename}")
    
    # Save channel information if requested
    if SAVE_CHANNEL_INFO:
        channel_info_file = os.path.join(channel_info_dir, f'channels_{sub_id:02d}{file_suffix}.txt')
        with open(channel_info_file, 'w') as f:
            f.write(f"File: {os.path.basename(fif_path)}\n")
            f.write(f"Channels found ({len(epochs.ch_names)}):\n")
            for ch in epochs.ch_names:
                f.write(f"  {ch}\n")
            
            f.write("\nStandard channels present:\n")
            for ch in STANDARD_CHANNELS:
                if ch.upper() in [c.upper() for c in epochs.ch_names]:
                    f.write(f"  {ch} - Found as {[c for c in epochs.ch_names if c.upper() == ch.upper()][0]}\n")
                else:
                    f.write(f"  {ch} - Not found\n")
            
            f.write(f"\nEpochs info: {len(epochs)} epochs, {epochs.times.shape[0]} time points per epoch\n")
            f.write(f"Sampling rate: {epochs.info['sfreq']} Hz\n")
            f.write(f"Original data shape: {epochs_data.shape} (epochs, channels, times)\n")
            f.write(f"Saved data shape: {transposed_data.shape} (epochs, times, channels)\n")
    
    return transposed_data.shape

# ---------------------------
# Main Processing Loop
# ---------------------------
for participant_id in valid_subjects:
    try:
        sub_id = subject_id_map[participant_id]
        
        # Use participant_id directly if it's already in "sub-XXXXX" format
        if participant_id.startswith('sub-'):
            sub_name = participant_id
        else:
            # Try to convert to the right format if it's a numeric ID
            try:
                sub_name = f"sub-{int(participant_id):05d}"
            except ValueError:
                sub_name = participant_id  # Just use as is if conversion fails
        
        # Look for .fif files in the derivatives path for this subject
        fif_search_path = os.path.join(derivatives_path, sub_name, 'eeg', f'{sub_name}_task-*_desc-reject*_eeg.fif')
        fif_files = glob.glob(fif_search_path)
        
        if not fif_files:
            print(f"No .fif files found for {sub_name}")
            continue
        
        # Process each .fif file for this subject
        for i, fif_path in enumerate(fif_files):
            file_suffix = f"_{i}" if len(fif_files) > 1 else ""
            try:
                print(f"Processing {fif_path}")
                
                # Try to read as raw data first
                try:
                    print(f"  Trying to read as raw data...")
                    raw = mne.io.read_raw_fif(fif_path, preload=True)
                    print(f"  Raw data successfully loaded")
                    
                    # Convert to epochs with fixed length
                    # This is a fallback and likely won't be used, but included for completeness
                    data, times = raw.get_data(return_times=True)
                    data = data.T  # Transpose to (time, channels)
                    
                    # For raw data, we'd need to create artificial epochs
                    # This is just a placeholder - adjust as needed
                    print(f"  Converting raw data to epoched format...")
                    raise NotImplementedError("Raw data processing not fully implemented")
                
                except Exception as e:
                    print(f"  Failed to read as raw data: {e}")
                    print(f"  Trying to read as epochs data...")
                    
                    # Read as epochs
                    epochs = mne.read_epochs(fif_path, preload=True)
                    
                    # Process and save the epochs data
                    data_shape = process_epochs_data(epochs, sub_id, fif_path, file_suffix)
                    print(f"  Final data shape: {data_shape}")
            
            except Exception as e:
                print(f"Error processing {fif_path}:")
                print(traceback.format_exc())
    
    except Exception as e:
        print(f"Error processing participant {participant_id}:")
        print(traceback.format_exc())

print("Processing complete. Labels and features are now saved in CAUEEG2.")