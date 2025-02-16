import mne
import matplotlib.pyplot as plt
import os

# Define the correct path
subject_id = "00001"  # Adjust subject ID as needed
file_path = f"/home/stud/timlin/bhome/DiffusionEEG/data/caueeg_bids/derivatives/preprocessed/sub-00001_task-eyesClosed_preproc-epo.fif"

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load epochs data
epochs = mne.read_epochs(file_path)

# # Apply a bandpass filter (0.1–60 Hz to avoid DC offset)
# epochs_filtered = epochs.copy().filter(
#     l_freq=0.1,  # Lower cutoff (avoids DC)
#     h_freq=60.0,  # Upper cutoff (removes high-frequency noise)
#     method="iir",  # Use IIR filter for efficiency
#     verbose=True
# )

# Plot filtered epochs (first 10 epochs for visualization)
plt.figure(figsize=(12, 6))
epochs.plot(
    picks="eeg",  # Plot only EEG channels
    n_epochs=10,  # Show first 10 epochs
    title="LEAD EEG",
    show=False
)

# Save the plot
plt.savefig("LEAD_eeg_epochs3.png", dpi=300)
plt.close()

# Optional: Save filtered epochs to a new FIF file
# epochs_filtered.save("filtered_epochs.fif", overwrite=True)