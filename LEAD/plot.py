import numpy as np
import matplotlib.pyplot as plt

# Load the feature file (adjust the file path as needed)
# feature_file = '/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG/Feature/feature_01.npy'
# feature_file = '/home/stud/timlin/bhome/DiffusionEEG/results/LEAD/diffusion/LEAD/D-1-CAUEEG-Diff-Simple/generated_samples.npy'
feature_file = '/home/stud/timlin/bhome/DiffusionEEG/results/LEAD/diffusion/LEAD/D-1-CAUEEG-Diff-Advanced/generated_samples.npy'
features = np.load(feature_file)

# Check the shape of the loaded data
print("Features shape:", features.shape)

# Select a segment to plot (e.g., the first segment)
segment = features[0]  # Get the first segment
print("Segment shape:", segment.shape)

# Determine if we need to transpose
# We want format (time_points, channels) for plotting
# If shape is (19, 128) - transpose it to (128, 19)
# If shape is (128, 19) - keep as is
if segment.shape[0] == 19 and segment.shape[1] == 128:
    print("Transposing segment from (19, 128) to (128, 19)")
    segment = segment.T  # Transpose to get (128, 19)
    
# Now segment should be (128, 19) or (time_points, channels)
time_points, n_channels = segment.shape
print("Using shape for plotting:", segment.shape)

# Create a plot: plot each channel with a vertical offset for clarity
plt.figure(figsize=(12, 8))
offset = 10  # Adjust offset based on signal amplitude

for ch in range(n_channels):
    # Offset each channel's signal so they do not overlap
    plt.plot(segment[:, ch] + ch * offset, label="Channel {}".format(ch + 1))

plt.xlabel("Time (samples)")
plt.ylabel("Amplitude + Offset")
plt.title("EEG Segment Visualization")
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.savefig("Diff_LEAD_eeg_ADV.png", dpi=300)
plt.close()