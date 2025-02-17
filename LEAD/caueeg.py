import os
import numpy as np
from scipy.signal import resample
import mne
import pandas as pd

SAMPLE_RATE = 128  # Target sampling rate (Hz)
SAMPLE_LEN = 128   # Segment length (in time points)

STANDARD_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3',
    'PZ', 'P4', 'T6', 'O1', 'O2'
]

root = '/home/stud/timlin/bhome/DiffusionEEG/data/caueeg_bids'
participants_path = os.path.join(root, 'participants.tsv')
participants = pd.read_csv(participants_path, sep='\t')

# Assign labels and filter valid participants
participants['label'] = -1
for idx, row in participants.iterrows():
    diagnosis = row[4]  # Assuming the 5th column contains diagnosis info
    if pd.isna(diagnosis):
        continue
    diagnosis_str = str(diagnosis).lower()
    if 'hc (+smc)' in diagnosis_str:
        participants.at[idx, 'label'] = 0
    elif 'mci' in diagnosis_str:
        participants.at[idx, 'label'] = 1
    elif 'dementia' in diagnosis_str:
        participants.at[idx, 'label'] = 2

participants_valid = participants[participants['label'].isin([0, 1, 2])].copy()
participants_valid['subject_id'] = range(1, len(participants_valid) + 1)

# Create and save labels array
labels = participants_valid[['label', 'subject_id']].to_numpy(dtype='int32')
label_path = '/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG/Label'
os.makedirs(label_path, exist_ok=True)
np.save(os.path.join(label_path, 'label.npy'), labels)

# Feature extraction setup
feature_path = '/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG/Feature'
os.makedirs(feature_path, exist_ok=True)

subject_id_map = dict(zip(participants_valid.iloc[:, 0], participants_valid['subject_id']))
valid_subjects = participants_valid.iloc[:, 0].tolist()

def resample_time_series(data, original_fs, target_fs):
    new_length = int(data.shape[0] * target_fs / original_fs)
    return resample(data, new_length, axis=0)

def split_eeg_segments(data, segment_length=SAMPLE_LEN, half_overlap=False):
    T, C = data.shape
    step = segment_length // 2 if half_overlap else segment_length
    num_segments = (T - segment_length) // step + 1
    segments = np.stack([data[i*step:i*step+segment_length] for i in range(num_segments)])
    return segments

# Process EEG data for valid subjects
for participant_id in valid_subjects:
    sub_id = subject_id_map[participant_id]
    eeg_dir = os.path.join(root, participant_id, 'eeg')
    
    if not os.path.exists(eeg_dir):
        continue
    
    for file in os.listdir(eeg_dir):
        if file.endswith('.vhdr'):
            file_path = os.path.join(eeg_dir, file)
            try:
                raw = mne.io.read_raw_brainvision(file_path, preload=True)
                raw.pick_channels(STANDARD_CHANNELS)
                data = raw.get_data().T
                
                if raw.info['sfreq'] != SAMPLE_RATE:
                    data = resample_time_series(data, raw.info['sfreq'], SAMPLE_RATE)
                
                features = split_eeg_segments(data, SAMPLE_LEN, half_overlap=True)
                file_base = os.path.splitext(file)[0]
                np.save(os.path.join(feature_path, f'feature_{sub_id:02d}.npy'), features)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

print("Processing complete. Labels and features are now properly aligned.")