import os
import numpy as np
import glob
import shutil

# Define paths
output_root = "dataset/SYNTH-CAUEEG2"
label_path = os.path.join(output_root, 'Label')
feature_path = os.path.join(output_root, 'Feature')
metadata_path = os.path.join(output_root, 'Metadata')  # Add metadata directory

# Source paths
hc_path = "dataset/SYNTH-CAUEEG2-HC"
mci_path = "dataset/SYNTH-CAUEEG2-MCI"
dementia_path = "dataset/SYNTH-CAUEEG2-dementia"

# Class distribution (as provided)
class_0_count = 439  # HC
class_1_count = 328  # MCI
class_2_count = 219  # Dementia

# Create output directories
os.makedirs(output_root, exist_ok=True)
os.makedirs(label_path, exist_ok=True)
os.makedirs(feature_path, exist_ok=True)
os.makedirs(metadata_path, exist_ok=True)  # Create metadata directory

# Initialize labels array
# Format: [[label, subject_id], [label, subject_id], ...]
labels = []
subject_id = 1  # Start subject IDs from 1

def process_files(source_dir, label, count, start_subject_id):
    """
    Process files from source directory, assign labels, and copy to feature directory
    Returns: next_subject_id, list of [label, subject_id] pairs
    """
    current_id = start_subject_id
    processed_count = 0
    
    # Get all .npy files but separate data and metadata
    all_files = glob.glob(os.path.join(source_dir, "*.npy"))
    
    # Filter files properly: data files shouldn't have '_metadata' in their name
    data_files = []
    metadata_files = []
    
    for file_path in all_files:
        if "_metadata" in os.path.basename(file_path):
            metadata_files.append(file_path)
        else:
            data_files.append(file_path)
    
    # Create mapping from original subject IDs to metadata files for easy lookup
    metadata_map = {}
    for meta_file in metadata_files:
        base_name = os.path.basename(meta_file).replace("_metadata.npy", "")
        metadata_map[base_name] = meta_file
    
    if not data_files:
        print(f"No data files found in {source_dir}")
        return current_id, []
    
    print(f"Processing {len(data_files)} data files and {len(metadata_files)} metadata files from {source_dir}")
    print(f"Target count for label {label}: {count}")
    
    # Process files up to the count limit
    new_labels = []
    for file_path in data_files:
        if processed_count >= count:
            break
            
        try:
            # Load the data to check format
            data = np.load(file_path)
            file_basename = os.path.basename(file_path)
            print(f"  File: {file_basename}, Shape: {data.shape}")
            
            # Create new feature filename
            new_feature_name = f"feature_{current_id:02d}.npy"
            new_feature_path = os.path.join(feature_path, new_feature_name)
            
            # Save the data to the new location
            np.save(new_feature_path, data)
            
            # Check if there's corresponding metadata and save it
            # First remove .npy extension to get base filename
            base_name = file_basename.rsplit('.', 1)[0]
            
            # Check if metadata exists for this file
            meta_file_path = os.path.join(source_dir, f"{base_name}_metadata.npy")
            if os.path.exists(meta_file_path):
                try:
                    # Load metadata with allow_pickle=True
                    metadata = np.load(meta_file_path, allow_pickle=True)
                    
                    # Save metadata with new name
                    meta_output_path = os.path.join(metadata_path, f"metadata_{current_id:02d}.npy")
                    np.save(meta_output_path, metadata)
                    print(f"  Saved metadata for subject {current_id}")
                except Exception as e:
                    print(f"  Error processing metadata {meta_file_path}: {e}")
            
            # Add label
            new_labels.append([label, current_id])
            
            # Increment counters
            current_id += 1
            processed_count += 1
            
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    print(f"Processed {processed_count} files with label {label}")
    return current_id, new_labels

# Process HC files (Class 0)
print("\nProcessing HC (Class 0) files...")
next_id, hc_labels = process_files(hc_path, 0, class_0_count, subject_id)
labels.extend(hc_labels)

# Process MCI files (Class 1)
print("\nProcessing MCI (Class 1) files...")
next_id, mci_labels = process_files(mci_path, 1, class_1_count, next_id)
labels.extend(mci_labels)

# Process Dementia files (Class 2)
print("\nProcessing Dementia (Class 2) files...")
_, dementia_labels = process_files(dementia_path, 2, class_2_count, next_id)
labels.extend(dementia_labels)

# Convert labels to numpy array and save
labels_array = np.array(labels, dtype='int32')
np.save(os.path.join(label_path, 'label.npy'), labels_array)

# Print summary
print("\nSynthetic dataset creation complete.")
print(f"Class distribution in the synthetic dataset:")
c0 = np.sum(labels_array[:, 0] == 0)
c1 = np.sum(labels_array[:, 0] == 1)
c2 = np.sum(labels_array[:, 0] == 2)
print(f"  Class 0 (HC): {c0} subjects (target: {class_0_count})")
print(f"  Class 1 (MCI): {c1} subjects (target: {class_1_count})")
print(f"  Class 2 (Dementia): {c2} subjects (target: {class_2_count})")
print(f"Total: {len(labels_array)} subjects")

# Check if we met the targets
if c0 < class_0_count:
    print(f"WARNING: Not enough HC subjects (missing {class_0_count - c0})")
if c1 < class_1_count:
    print(f"WARNING: Not enough MCI subjects (missing {class_1_count - c1})")
if c2 < class_2_count:
    print(f"WARNING: Not enough Dementia subjects (missing {class_2_count - c2})")