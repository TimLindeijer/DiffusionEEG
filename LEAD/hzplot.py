import numpy as np
import mne
import matplotlib.pyplot as plt

path = '/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2/Feature/feature_01.npy'

data = np.load(path)
sfreq = 200  # Sampling freq

n_epochs, n_times, n_channels = data.shape

# Generate dummy channel names just for plotting
channel_names = [f"EEG {i+1}" for i in range(n_channels)]
channel_types = ["eeg"] * n_channels

# Create MNE info structure
info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
# Transpose data to match MNE's expected shape: (n_epochs, n_channels, n_times)
data_mne = data.transpose(0, 2, 1)  # (546, 19, 128)
# Create MNE Epochs object
print(data_mne.shape)
epochs = mne.EpochsArray(data_mne, info)

# Compute and plot PSD
psd_fig = epochs.compute_psd(method="welch", fmin=1, fmax=30).plot(average=True)

# Make sure the plot is displayed (if running in a notebook or interactive environment)
# plt.show()

# Alternative: save the plot to a file
psd_fig.savefig('eeg_psd_plot2.png')