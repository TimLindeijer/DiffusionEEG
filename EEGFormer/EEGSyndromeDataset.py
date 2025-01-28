import os
import glob
import mne
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EEGSyndromeDataset(Dataset):
    def __init__(self, root_dir, participants_tsv, l_freq=0.1, h_freq=60.0):
        """
        Args:
            root_dir (str): Path to derivatives/sovaharmony directory.
            participants_tsv (str): Path to participants.tsv file.
            l_freq (float): Low cutoff frequency for bandpass filter.
            h_freq (float): High cutoff frequency for bandpass filter.
        """
        self.root_dir = root_dir
        self.l_freq = l_freq
        self.h_freq = h_freq
        
        # Load and filter participants.tsv
        self.participants_df = pd.read_csv(participants_tsv, sep='\t')
        self.participants_df = self.participants_df.dropna(subset=['ad_syndrome_3'])
        
        # Map syndrome categories to integer labels
        self.syndrome_mapping = {
            'mci': 0,
            'hc (+smc)': 1,
            'dementia': 2
        }
        
        # Find FIF files only for categorized participants
        self.fif_files = []
        self.labels = []
        for sub_id in self.participants_df['participant_id']:
            fif_pattern = os.path.join(
                root_dir, 
                f"{sub_id}/eeg/{sub_id}_task-eyesClosed_desc-reject[]_eeg.fif"
            )
            matched_files = glob.glob(fif_pattern)
            if matched_files:
                syndrome = self.participants_df.loc[
                    self.participants_df['participant_id'] == sub_id,
                    'ad_syndrome_3'
                ].iloc[0].strip().lower()
                
                if syndrome in self.syndrome_mapping:
                    self.fif_files.extend(matched_files)
                    self.labels.append(self.syndrome_mapping[syndrome])
        
        if not self.fif_files:
            raise FileNotFoundError("No valid FIF files found with syndrome categorization.")
            
        # Load all data into memory
        self.data = self._load_data()

    def _load_data(self):
        all_data = []
        for fif_file in self.fif_files:
            epochs = mne.read_epochs(fif_file, verbose=False)
            epochs = epochs.filter(
                l_freq=self.l_freq, 
                h_freq=self.h_freq, 
                method="iir", 
                verbose=False
            )
            all_data.append(epochs.get_data())
        
        return np.vstack(all_data)  # Shape: (n_epochs, n_channels, n_times)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.data[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return eeg, label

    def get_class_distribution(self):
        return {
            syndrome: list(self.labels).count(label)
            for syndrome, label in self.syndrome_mapping.items()
        }

if __name__ == "__main__":
    participants_tsv = "/home/stud/timlin/bhome/DiffusionEEG/data/caueeg_bids/participants.tsv"
    root_dir = "/home/stud/timlin/bhome/DiffusionEEG/data/caueeg_bids/derivatives/sovaharmony"
    
    try:
        print("Creating syndrome-categorized dataset...")
        dataset = EEGSyndromeDataset(root_dir, participants_tsv)
        
        print("\nClass distribution:")
        for syndrome, count in dataset.get_class_distribution().items():
            print(f"{syndrome}: {count} samples")
            
        sample_idx = 0
        eeg_sample, label_sample = dataset[sample_idx]
        print(f"\nSample {sample_idx} label: {list(dataset.syndrome_mapping.keys())[label_sample]}")
        
    except Exception as e:
        print(f"Error: {e}")