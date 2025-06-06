import os
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
from autoreject import AutoReject
import mne

# === Set up ===
bids_root = "/bhome/tordmy/Master/data/caueeg_bids"
deriv_root = os.path.join(bids_root, "derivatives", "preprocessed")
os.makedirs(deriv_root, exist_ok=True)

# Get all subjects in the dataset
subjects = get_entity_vals(bids_root, entity_key='subject')

# === Loop through each subject ===
for subject in subjects:
    try:
        print(f"\nProcessing subject: {subject}")

        # Set BIDS path
        bids_path = BIDSPath(root=bids_root, subject=subject, task="eyesClosed", datatype="eeg")

        # Load raw EEG data
        raw = read_raw_bids(bids_path, verbose=False)
        raw.load_data()
        
        #Rename to proper case
        mapping = {ch: ch.capitalize() for ch in raw.ch_names}
        raw.rename_channels(mapping)

        #Pick only EEG channels (exclude EKG, Photic)
        raw = raw.copy().pick_types(eeg=True, exclude=['Ekg', 'Photic'])

        #Set montage on EEG-only data
        raw.set_montage('standard_1020')

        # Step 1: Average reference
        raw.set_eeg_reference('average', projection=False)

        # Step 2: Band-pass filter (1–100 Hz)
        raw.filter(l_freq=1., h_freq=99.)

        # Step 3: Epoching (10s fixed-length)
        events = mne.make_fixed_length_events(raw, duration=10.0)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=10.0, baseline=None, preload=True)

        # Step 4: Artifact rejection with Autoreject
        ar = AutoReject()
        epochs_clean = ar.fit_transform(epochs)

        # Step 5: Resample to 140 Hz
        epochs_clean.resample(140)

        # Step 6: Save cleaned epochs
        output_fname = os.path.join(deriv_root, f"sub-{subject}_task-eyesClosed_clean_140hz-epo.fif")
        epochs_clean.save(output_fname, overwrite=True)

        print(f"Saved cleaned data for subject {subject} to {output_fname}")

    except Exception as e:
        print(f"❌ Failed to process subject {subject}: {e}")
