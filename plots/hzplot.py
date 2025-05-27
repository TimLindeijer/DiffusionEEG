import numpy as np
import mne
import matplotlib.pyplot as plt
import os
from scipy import signal

# File path
path = 'dataset/discrimination_ftsurrogate/all_augmented_train/Feature/feature_02.npy'

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

# Check for problematic values
print(f"NaN values: {np.isnan(data).any()}")
print(f"Inf values: {np.isinf(data).any()}")
print(f"Data range: {np.min(data)} to {np.max(data)}")
print(f"Zero values only: {np.allclose(data, 0, atol=1e-10)}")

# Fix data if needed
if np.isnan(data).any() or np.isinf(data).any():
    # Replace NaN/Inf with small values
    data = np.nan_to_num(data, nan=1e-6, posinf=1.0, neginf=-1.0)
    print("Replaced NaN/Inf values")

# Add tiny jitter if data is all zeros or very small
if np.allclose(data, 0, atol=1e-6):
    data = data + np.random.normal(0, 1e-5, data.shape)
    print("Added small jitter to avoid all-zero data")

# Generate dummy channel names just for plotting
channel_names = [f"EEG {i+1}" for i in range(n_channels)]
channel_types = ["eeg"] * n_channels

# Create MNE info structure
info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)

# Create MNE Epochs object
epochs = mne.EpochsArray(data, info)

# Function to plot PSD using scipy's welch method (more robust)
def plot_psd_scipy(data, sfreq, fmin=1, fmax=30):
    n_epochs, n_channels, n_times = data.shape
    
    # Calculate frequencies
    freqs, _ = signal.welch(data[0, 0], fs=sfreq, nperseg=min(256, n_times))
    
    # Filter frequencies
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    
    # Calculate PSD for all channels and epochs
    psd_all = np.zeros((n_epochs, n_channels, len(freqs)))
    
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            _, psd = signal.welch(data[epoch_idx, ch_idx], fs=sfreq, nperseg=min(256, n_times))
            psd_all[epoch_idx, ch_idx] = psd[mask]
    
    # Average across epochs
    psd_avg = np.mean(psd_all, axis=0)
    
    # Plot
    plt.figure(figsize=(12, 8))
    for ch_idx in range(n_channels):
        plt.semilogy(freqs, psd_avg[ch_idx], label=f'Channel {ch_idx+1}')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (µV²/Hz)')
    plt.title('Power Spectral Density')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()

# Plot PSD using scipy (more robust method)
print("Computing and plotting PSD using scipy...")
psd_fig = plot_psd_scipy(data, sfreq=sfreq, fmin=1, fmax=30)
psd_fig.savefig(os.path.join(output_dir, 'augmentedeeg_psd_all_channels_feature01_1000TP.png'))
print(f"Saved PSD plot to {os.path.join(output_dir, 'augmentedeeg_psd_all_channels_feature01_1000TP.png')}")

# Try MNE's method as a fallback
try:
    print("Attempting MNE's PSD calculation...")
    # Use multitaper method instead of welch
    psd_mne_fig = epochs.compute_psd(method="multitaper", fmin=1, fmax=30).plot(average=True)
    psd_mne_fig.savefig(os.path.join(output_dir, 'augmentedeeg_psd_mne_method.png'))
    print(f"Saved MNE PSD plot")
except Exception as e:
    print(f"MNE PSD calculation failed: {e}")

# Plot a sample of the raw data
print("Plotting raw data sample...")
fig, ax = plt.subplots(n_channels, 1, figsize=(12, 2*n_channels), sharex=True)
time = np.arange(n_times) / sfreq

for ch_idx in range(n_channels):
    if n_channels > 1:
        ax[ch_idx].plot(time, data[0, ch_idx])
        ax[ch_idx].set_ylabel(f"Ch {ch_idx+1}")
    else:
        ax.plot(time, data[0, ch_idx])
        ax.set_ylabel(f"Ch {ch_idx+1}")

# Add labels
if n_channels > 1:
    ax[-1].set_xlabel("Time (s)")
else:
    ax.set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'augmentedeeg_raw_sample_feat01_1000TP.png'))
print(f"Saved raw data plot")

print("All plots generated successfully!")