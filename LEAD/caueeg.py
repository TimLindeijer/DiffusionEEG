import os
import numpy as np
from scipy.signal import resample
import mne
import pandas as pd

# ---------------------------
# Configuration and Constants
# ---------------------------
SAMPLE_RATE = 128   # Target sampling rate (Hz)
SAMPLE_LEN = 128    # Segment length in time points (1 second at 128 Hz)
LOW_FREQ = 0.5      # Lower bound for bandpass filtering (Hz)
HIGH_FREQ = 45      # Upper bound for bandpass filtering (Hz)

# Standard set of 19 EEG channels (10-20 system)
STANDARD_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3',
    'Pz', 'P4', 'T6', 'O1', 'O2'
]

# Root paths for input and output data
root = '/home/stud/timlin/bhome/DiffusionEEG/data/caueeg_bids'
participants_path = os.path.join(root, 'participants.tsv')
label_path = '/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG/Label'
feature_path = '/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG/Feature'

os.makedirs(label_path, exist_ok=True)
os.makedirs(feature_path, exist_ok=True)

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

# ---------------------------
# Utility Functions
# ---------------------------
def resample_time_series(data, original_fs, target_fs):
    """
    Resample time series data along the time axis.
    """
    new_length = int(data.shape[0] * target_fs / original_fs)
    return resample(data, new_length, axis=0)

def split_eeg_segments(data, segment_length=SAMPLE_LEN, half_overlap=False):
    """
    Split continuous EEG data into segments of fixed length.
    Optionally use half-overlapping segments.
    """
    T, C = data.shape
    step = segment_length // 2 if half_overlap else segment_length
    num_segments = (T - segment_length) // step + 1
    segments = np.stack([data[i * step : i * step + segment_length] for i in range(num_segments)])
    return segments

def standard_normalize(segment):
    """
    Perform standard normalization on each channel of a segment.
    Each channel is zero-meaned and scaled to unit variance.
    """
    norm_segment = np.empty_like(segment)
    for ch in range(segment.shape[1]):
        col = segment[:, ch]
        mean = np.mean(col)
        std = np.std(col)
        # Avoid division by zero
        if std < 1e-6:
            norm_segment[:, ch] = col - mean
        else:
            norm_segment[:, ch] = (col - mean) / std
    return norm_segment

def align_channels(raw, standard_channels):
    """
    Align channels to a fixed standard of 19 EEG channels.
    
    This function:
      - Sets non-EEG channels (e.g., EKG, Photic) to 'misc'
      - Applies the standard 10-20 montage using on_missing='warn'
      - Renames channels to match the expected montage nomenclature if needed.
      - If any standard channels remain missing, adds zero-filled channels and interpolates them.
      - Finally, selects only the standard channels.
    """
    # Create standard montage
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Set non-EEG channels to 'misc' to avoid montage conflicts
    for extra in ['EKG', 'Photic']:
        if extra in raw.ch_names:
            raw.set_channel_types({extra: 'misc'})
    
    # Apply the montage with on_missing='warn' (match_case=False allows case-insensitive matching)
    raw.set_montage(montage, on_missing='warn', match_case=False)
    
    # Build a renaming mapping for channels that differ in case
    available_channels = raw.ch_names
    rename_mapping = {}
    for ch in standard_channels:
        if ch not in available_channels:
            # Look for a case-insensitive match
            for avail in available_channels:
                if avail.lower() == ch.lower():
                    rename_mapping[avail] = ch
                    break
    if rename_mapping:
        raw.rename_channels(rename_mapping)
    
    # After renaming, check for any still-missing standard channels
    missing = [ch for ch in standard_channels if ch not in raw.ch_names]
    if missing:
        print(f"Warning: Missing channels after renaming: {missing}. Adding zero data and interpolating.")
        n_times = raw.n_times
        missing_data = np.zeros((len(missing), n_times))
        info_missing = mne.create_info(ch_names=missing, sfreq=raw.info['sfreq'], ch_types='eeg')
        raw_missing = mne.io.RawArray(missing_data, info_missing, verbose=False)
        raw.add_channels([raw_missing], force_update_info=True)
        raw.info['bads'].extend(missing)
        raw.interpolate_bads(reset_bads=True, verbose=False)
    
    # Use the new pick() method. Some MNE versions support the 'ordered' parameter.
    try:
        raw.pick(standard_channels, ordered=False)
    except TypeError:
        raw.pick(standard_channels)
    
    return raw

# ---------------------------
# Main Processing Loop
# ---------------------------
for participant_id in valid_subjects:
    sub_id = subject_id_map[participant_id]
    eeg_dir = os.path.join(root, participant_id, 'eeg')
    
    if not os.path.exists(eeg_dir):
        continue
    
    for file in os.listdir(eeg_dir):
        if file.endswith('.vhdr'):
            file_path = os.path.join(eeg_dir, file)
            try:
                # Read raw BrainVision EEG data
                raw = mne.io.read_raw_brainvision(file_path, preload=True)
                
                # ---- Channel Alignment ----
                raw = align_channels(raw, STANDARD_CHANNELS)
                
                # ---- Frequency Filtering ----
                raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, verbose=False)
                
                # ---- Frequency Alignment (Resampling) ----
                if raw.info['sfreq'] != SAMPLE_RATE:
                    raw.resample(SAMPLE_RATE, verbose=False)
                
                # Convert the raw data to a NumPy array with shape (time, channels)
                data = raw.get_data().T
                
                # ---- Sample Segmentation ----
                segments = split_eeg_segments(data, segment_length=SAMPLE_LEN, half_overlap=False)
                
                # ---- Standard Normalization ----
                segments_normalized = np.array([standard_normalize(segment) for segment in segments])
                
                # Save the processed features for the subject
                np.save(os.path.join(feature_path, f'feature_{sub_id:02d}.npy'), segments_normalized)
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

print("Processing complete. Labels and features are now properly aligned.")
