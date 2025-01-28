import mne
import matplotlib.pyplot as plt

# Define the path to your FIF file
subject_id = "00001"  # Example subject ID (adjust this)
file_path = f"/home/stud/timlin/bhome/DiffusionEEG/data/caueeg_bids/derivatives/sovaharmony/sub-{subject_id}/eeg/sub-{subject_id}_task-eyesClosed_desc-reject[]_eeg.fif"

# Load the FIF file (try as Raw, then Evoked/Epochs)
try:
    # Attempt to read as Raw data
    raw = mne.io.read_raw_fif(file_path, preload=True)
    data_type = "raw"
except:
    try:
        # Attempt to read as Evoked data (average)
        evoked_list = mne.read_evokeds(file_path)
        evoked = evoked_list[0]  # Use the first evoked response
        data_type = "evoked"
    except:
        try:
            # Attempt to read as Epochs
            epochs = mne.read_epochs(file_path)
            data_type = "epochs"
        except Exception as e:
            raise ValueError(f"Could not read FIF file: {e}")

# Plot and save based on data type
plt.figure(figsize=(12, 6))

if data_type == "raw":
    # Plot 10 seconds of EEG data (first 10 channels)
    raw.pick_types(eeg=True).crop(tmax=10).load_data()
    data, times = raw[:, :int(10 * raw.info["sfreq"])]
    plt.plot(times, data.T * 1e6)  # Convert to microvolts
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("Raw EEG Data (First 10 Seconds)")
    
elif data_type == "evoked":
    # Plot evoked response
    evoked.plot(show=False)
    plt.title("Evoked Response")
    
elif data_type == "epochs":
    # Plot epochs
    epochs.plot(show=False)
    plt.title("Epochs")

# Save the plot
plt.tight_layout()
plt.savefig("eeg_plot.png")
plt.close()