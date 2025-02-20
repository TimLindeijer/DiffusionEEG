import numpy as np
import matplotlib.pyplot as plt

# Load the feature file (adjust the file path as needed)
feature_file = '/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG/Feature/feature_01.npy'
features = np.load(feature_file)

# Check the shape of the loaded data
print("Features shape:", features.shape)  # Expecting (n_segments, 128, 19)

# Select a segment to plot (e.g., the first segment)
segment = features[0]  # shape: (128, 19)

# Create a plot: plot each channel with a vertical offset for clarity
plt.figure(figsize=(12, 8))
offset = 10  # Adjust offset based on signal amplitude

for ch in range(segment.shape[1]):
    # Offset each channel's signal so they do not overlap
    plt.plot(segment[:, ch] + ch * offset, label=f"Channel {ch + 1}")

plt.xlabel("Time (samples)")
plt.ylabel("Amplitude + Offset")
plt.title("EEG Segment Visualization")
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.savefig("LEAD_eeg.png", dpi=300)
plt.close()