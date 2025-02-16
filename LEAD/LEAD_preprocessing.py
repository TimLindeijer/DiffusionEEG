import sovaharmony
from sovaharmony.preprocessing import harmonize
import mne
import numpy as np
import os
import pandas as pd
import bids
from mne_bids import BIDSPath, read_raw_bids, get_entities_from_fname

# Configuration
user_path = "/home/stud/timlin/bhome/DiffusionEEG"
verbose = True
data_path = os.path.join(user_path, "data")
bids_root = os.path.join(data_path, "caueeg_bids")
output_dir = os.path.join(bids_root, "derivatives", "preprocessed")
os.makedirs(output_dir, exist_ok=True)

# Standard 10-20 system channels (19 channels)
STANDARD_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3',
    'PZ', 'P4', 'T6', 'O1', 'O2'
]

def debug_print(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)

# 1. Dataset configuration and initial harmonization
THE_DATASET = {
    'name': 'korea',
    'input_path': bids_root,
    'layout': {'extension': '.vhdr', 'suffix': 'eeg', 'task': 'eyesClosed', 'return_type': 'filename'},
    'args': {'line_freqs': [60]},
    'channels': STANDARD_CHANNELS,
    'group_regex': None,
    'events_to_keep': None
}

korea={
'layout':{'extension':'.fif', 'suffix':'eeg', 'return_type':'filename', 'task':'eyesClosed',},
    'ch_names':STANDARD_CHANNELS,
    'path': bids_root
}
DATASET=korea #DEFINE DATASET


debug_print("Performing initial harmonization...")
sovaharmony.preprocessing.harmonize(THE_DATASET, fast_mode=False)

# 2. Locate processed files
# layout = bids.BIDSLayout(bids_root, derivatives=True)
# eegs = layout.get(
#     extension='.fif',
#     suffix='eeg',
#     task='eyesClosed',
#     # processing='sovaharmony',
#     scope='derivatives'
# )
layoutd = DATASET.get('layout', None)

layout = bids.BIDSLayout(DATASET.get('path', None), derivatives=True)
eegs = layout.get(**layoutd)
eegs = [k for k in eegs if 'eyesClosed_desc-reject' in k]

debug_print(f"\nFound {len(eegs)} processed files")

# 3. Process each file
dict_list = []
for eeg_file in eegs:
    # Get subject info
    entities = layout.parse_file_entities(eeg_file)
    debug_print(f"\nProcessing subject {entities['subject']}")
    
    # Load harmonized epochs
    epochs = mne.read_epochs(eeg_file, preload=True)
    
    # Convert epochs to continuous raw data
    data = epochs.get_data()
    data_continuous = np.concatenate(data, axis=-1)
    info = epochs.info.copy()
    raw = mne.io.RawArray(data_continuous, info)
    
    # Channel verification
    current_channels = set(raw.ch_names)
    missing = set(STANDARD_CHANNELS) - current_channels
    extra = current_channels - set(STANDARD_CHANNELS)
    
    if missing:
        debug_print(f"Interpolating missing channels: {missing}")
        raw.set_montage('standard_1005')
        raw.info['bads'].extend(list(missing))
        raw.interpolate_bads(reset_bads=True)
    
    if extra:
        debug_print(f"Removing extra channels: {extra}")
        raw.pick_channels(STANDARD_CHANNELS)
    
    # Frequency alignment
    if raw.info['sfreq'] != 128:
        debug_print(f"Resampling from {raw.info['sfreq']}Hz to 128Hz")
        raw.resample(128)
    
    # Frequency filtering
    debug_print("Applying bandpass filter (0.5-45Hz)")
    raw.filter(0.5, 45, method='iir', verbose='WARNING')
    
    # # Create new 1-second epochs
    debug_print("Creating 1-second epochs")
    new_epochs = mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0.0, preload=True)
    # Standard normalization
    debug_print("Applying z-score normalization")
    def zscore_epoch(epoch):
        return (epoch - np.mean(epoch, axis=-1, keepdims=True)) / np.std(epoch, axis=-1, keepdims=True)
    
    new_epochs.apply_function(zscore_epoch, picks='eeg', verbose='WARNING')
    
    # Save preprocessed epochs
    output_fname = f"sub-{entities['subject']}_task-eyesClosed_preproc-epo.fif"
    output_path = os.path.join(output_dir, output_fname)
    new_epochs.save(output_path, overwrite=True)
    
    # Collect metadata
    features = {
        'subject': f"k_{entities['subject']}",
        # 'session': entities['session'],
        'num_epochs': len(epochs),
        'channels': len(epochs.ch_names),
        'sfreq': epochs.info['sfreq']
    }
    dict_list.append(features)

# Save processing report
description = pd.DataFrame(dict_list)
description.to_csv(os.path.join(output_dir, "preprocessing_report.csv"), index=False)
debug_print("\nPreprocessing complete!")