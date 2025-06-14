import numpy as np
import os
from pathlib import Path

def normalize_to_reference(synthetic_data, reference_data, per_channel=True):
    """
    Normalize synthetic data to match the statistical properties of reference data.
    
    Args:
        synthetic_data: Numpy array of synthetic data of shape [epochs, channels, timepoints]
        reference_data: Numpy array of reference data of shape [samples, epochs, channels, timepoints]
        per_channel: Whether to normalize each channel separately
    
    Returns:
        Normalized synthetic data with same statistical properties as reference
    """
    print(f"Synthetic data shape: {synthetic_data.shape}, Reference data shape: {reference_data.shape}")
    print("Applying statistical normalization to match real data properties...")
    
    # Create a copy of synthetic data to modify
    normalized_data = synthetic_data.copy()
    
    if per_channel:
        # Normalize each channel separately
        print("Normalizing each EEG channel separately...")
        for c in range(synthetic_data.shape[1]):
            # Calculate statistics for this channel
            ref_mean = reference_data[:, c].mean()
            ref_std = reference_data[:, c].std()
            syn_mean = synthetic_data[:, c].mean()
            syn_std = synthetic_data[:, c].std()
            
            # Check for extremely small standard deviations
            if syn_std < 1e-10:
                print(f"Warning: Very small std in synthetic data channel {c}: {syn_std}")
                syn_std = 1e-10
            
            # Normalize synthetic data to match reference statistics
            normalized_data[:, c] = ((synthetic_data[:, c] - syn_mean) / syn_std) * ref_std + ref_mean
            
            # Log the changes
            print(f"Channel {c}: Changed mean from {syn_mean:.2f} to {ref_mean:.2f}, " +
                  f"std from {syn_std:.2f} to {ref_std:.2f}")
    else:
        # Normalize the entire data at once
        print("Normalizing all EEG channels together...")
        ref_mean = reference_data.mean()
        ref_std = reference_data.std()
        syn_mean = synthetic_data.mean()
        syn_std = synthetic_data.std()
        
        # Check for extremely small standard deviations
        if syn_std < 1e-10:
            print(f"Warning: Very small std in synthetic data: {syn_std}")
            syn_std = 1e-10
        
        # Normalize synthetic data to match reference statistics
        normalized_data = ((synthetic_data - syn_mean) / syn_std) * ref_std + ref_mean
        
        # Log the changes
        print(f"Overall: Changed mean from {syn_mean:.2f} to {ref_mean:.2f}, " +
              f"std from {syn_std:.2f} to {ref_std:.2f}")
    
    return normalized_data

def normalize_existing_synth_dataset(synth_dataset_path, reference_data_path, output_path=None):
    """
    Normalize an existing synthetic dataset to match reference data properties.
    
    Args:
        synth_dataset_path: Path to synthetic dataset directory (contains Feature and Label folders)
        reference_data_path: Path to reference CAUEEG2 data (.npy file or directory)
        output_path: Path for normalized dataset (if None, overwrites original)
    """
    
    # Set up paths
    synth_path = Path(synth_dataset_path)
    feature_path = synth_path / "Feature"
    label_path = synth_path / "Label"
    
    if output_path is None:
        # Create backup first
        backup_path = synth_path.parent / f"{synth_path.name}_backup"
        print(f"Creating backup at: {backup_path}")
        import shutil
        if not backup_path.exists():
            shutil.copytree(synth_path, backup_path)
        output_feature_path = feature_path
        output_label_path = label_path
    else:
        output_path = Path(output_path)
        output_feature_path = output_path / "Feature"
        output_label_path = output_path / "Label"
        output_feature_path.mkdir(parents=True, exist_ok=True)
        output_label_path.mkdir(parents=True, exist_ok=True)
    
    # Load reference data
    print("Loading reference CAUEEG2 data...")
    reference_path = Path(reference_data_path)
    
    if reference_path.is_file():
        # Single file
        print(f"Loading single reference file: {reference_path}")
        reference_data = np.load(reference_path)
        print(f"Loaded reference data shape: {reference_data.shape}")
    elif reference_path.is_dir():
        # Directory with multiple files - concatenate them
        ref_files = sorted(list(reference_path.glob("*.npy")))
        print(f"Found {len(ref_files)} reference files in directory")
        
        if len(ref_files) == 0:
            raise ValueError(f"No .npy files found in reference directory: {reference_path}")
        
        print("Loading and concatenating reference files...")
        ref_data_list = []
        for i, ref_file in enumerate(ref_files):
            data = np.load(ref_file)
            ref_data_list.append(data)
            print(f"  {ref_file.name}: {data.shape}")
        
        reference_data = np.concatenate(ref_data_list, axis=0)
        print(f"Combined reference data shape: {reference_data.shape}")
    else:
        raise ValueError(f"Reference path does not exist: {reference_path}")
    
    print(f"Final reference data shape: {reference_data.shape}")
    
    # Ensure reference data is in the right format for normalization
    # If it's 3D [samples, channels, timepoints], reshape to [samples, channels]
    if len(reference_data.shape) == 3:
        # Flatten timepoints dimension for statistics calculation
        reference_data = reference_data.reshape(-1, reference_data.shape[1])
    
    # Get list of synthetic feature files
    feature_files = sorted(list(feature_path.glob("*.npy")))
    print(f"Found {len(feature_files)} synthetic feature files to normalize")
    
    # Process each synthetic file
    for i, feature_file in enumerate(feature_files):
        print(f"\nProcessing {feature_file.name} ({i+1}/{len(feature_files)})")
        
        # Load synthetic data
        synthetic_data = np.load(feature_file)
        print(f"Original synthetic data shape: {synthetic_data.shape}")
        
        # Normalize the data
        if len(synthetic_data.shape) == 3:
            # Data is [epochs, channels, timepoints] - normalize using all timepoints
            flattened_synth = synthetic_data.reshape(-1, synthetic_data.shape[1])
            normalized_flat = normalize_to_reference(flattened_synth, reference_data, per_channel=True)
            normalized_data = normalized_flat.reshape(synthetic_data.shape)
        else:
            # Data is already 2D
            normalized_data = normalize_to_reference(synthetic_data, reference_data, per_channel=True)
        
        # Save normalized feature data
        output_file = output_feature_path / feature_file.name
        np.save(output_file, normalized_data)
        print(f"Saved normalized data to: {output_file}")
        
        # Copy corresponding label file (no normalization needed for labels)
        label_file = label_path / feature_file.name
        if label_file.exists():
            output_label_file = output_label_path / feature_file.name
            if output_label_path != label_path:  # Don't copy to same location
                import shutil
                shutil.copy2(label_file, output_label_file)
    
    print(f"\nNormalization complete! Processed {len(feature_files)} files.")
    if output_path is None:
        print("Original dataset has been normalized (backup created).")
    else:
        print(f"Normalized dataset saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Paths - adjust these to your actual paths
    synth_dataset = "/home/stud/timlin/bhome/DiffusionEEG/dataset/SYNTH-CAUEEG2"
    reference_data = "/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2/Feature"  # Adjust this path
    
    # Option 1: Normalize in place (creates backup)
    # normalize_existing_synth_dataset(synth_dataset, reference_data)
    
    # Option 2: Create new normalized dataset
    output_dir = "/home/stud/timlin/bhome/DiffusionEEG/dataset/SYNTH-CAUEEG2-NORMALIZED"
    normalize_existing_synth_dataset(synth_dataset, reference_data, output_dir)