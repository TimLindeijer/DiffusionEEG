import numpy as np
import mne
import matplotlib.pyplot as plt
import os

# File path
path = '/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2/Feature/feature_01.npy'

# Output directory
output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)

# Load data
data = np.load(path)
sfreq = 200  # Sampling frequency

print(f"Original data shape: {data.shape}")

# MNE expects data in shape (n_epochs, n_channels, n_times)
# Your data is (n_epochs, n_channels, n_timepoints)
if data.shape[2] == 19:
    data = data.transpose(0, 2, 1) 
    print(f"Transposed data shape: {data.shape}")
n_epochs, n_channels, n_times = data.shape

# Confirm data dimensions
print(f"Epochs: {n_epochs}, Channels: {n_channels}, Timepoints: {n_times}")

# Generate dummy channel names just for plotting
channel_names = [f"EEG {i+1}" for i in range(n_channels)]
channel_types = ["eeg"] * n_channels

# Create MNE info structure
info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)

# Create MNE Epochs object - data is already in the right shape (n_epochs, n_channels, n_times)
epochs = mne.EpochsArray(data, info)

# Plot PSD for all channels
print("Computing and plotting PSD...")
psd_fig = epochs.compute_psd(method="welch", fmin=1, fmax=30).plot(average=True)
psd_fig.savefig(os.path.join(output_dir, 'caueeg2_eeg_psd_all_channels_feature01.png'))
print(f"Saved PSD plot to {os.path.join(output_dir, 'caueeg2_eeg_psd_all_channels.png')}")

# # Plot PSDs for individual channels
# print("Plotting individual channel PSDs...")
# for ch_idx in range(min(n_channels, 6)):  # Plot first 6 channels (or fewer if there are less)
#     ch_name = channel_names[ch_idx]
#     ch_fig = epochs.compute_psd(method="welch", fmin=1, fmax=30, picks=[ch_idx]).plot(average=True)
#     ch_fig.savefig(os.path.join(output_dir, f'synthetic_eeg_psd_channel_{ch_idx+1}.png'))
#     print(f"Saved PSD plot for channel {ch_name} to {os.path.join(output_dir, f'synthetic_eeg_psd_channel_{ch_idx+1}.png')}")

# Plot a sample of the raw data
print("Plotting raw data sample...")
raw_fig = epochs[0].plot(scalings='auto')
plt.savefig(os.path.join(output_dir, 'caueeg2_eeg_raw_sample.png'))
print(f"Saved raw data plot to {os.path.join(output_dir, 'caueeg2_eeg_raw_sample.png')}")

print("All plots generated successfully!")