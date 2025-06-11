import numpy as np
import os

# Define the paths
label_path = "/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2/Label/label.npy"
feature_dir = "/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2/Feature/"

# Load the labels to get the IDs
labels = np.load(label_path)
ids = labels[:, 1]
unique_ids = np.unique(ids)

# Check epochs for each feature file
epoch_counts = []
missing_files = []
short_files = []  # Files with < 10 epochs

for feature_id in unique_ids:
    feature_file = os.path.join(feature_dir, f"feature_{feature_id:02d}.npy")
    
    if os.path.exists(feature_file):
        feature_data = np.load(feature_file)
        num_epochs = feature_data.shape[0]
        epoch_counts.append(num_epochs)
        
        if num_epochs < 10:
            short_files.append((feature_id, num_epochs))
        
        print(f"feature_{feature_id:02d}.npy: {num_epochs} epochs")
    else:
        missing_files.append(feature_id)
        print(f"feature_{feature_id:02d}.npy: MISSING")

if epoch_counts:
    avg_epochs = np.mean(epoch_counts)
    min_epochs = np.min(epoch_counts)
    max_epochs = np.max(epoch_counts)
    
    print(f"\n--- Summary ---")
    print(f"Average epochs: {avg_epochs:.2f}")
    print(f"Min epochs: {min_epochs}")
    print(f"Max epochs: {max_epochs}")
    print(f"Total files checked: {len(epoch_counts)}")
    print(f"Files with < 10 epochs: {len(short_files)}")
    
    if short_files:
        print(f"\nFiles with fewer than 10 epochs:")
        for file_id, epochs in short_files:
            print(f"  feature_{file_id:02d}.npy: {epochs} epochs")
    
    if missing_files:
        print(f"\nMissing files: {len(missing_files)}")
else:
    print("No feature files found!")