#!/usr/bin/env python
"""
Create datasets for evaluating genuine vs synthetic EEG discrimination.
This script:
1. Creates balanced datasets (equal genuine and synthetic samples)
2. Creates separate datasets for each dementia class (HC, MCI, Dementia)
3. Creates a combined dataset with all classes
4. Relabels samples as genuine (0) or synthetic (1)
5. Splits each dataset into train and validation sets
"""

import os
import argparse
import numpy as np
import shutil
from collections import defaultdict
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Create genuine vs synthetic discrimination datasets")
    
    # Path to datasets
    parser.add_argument("--genuine_dataset", type=str, required=True,
                      help="Path to the genuine dataset directory")
    parser.add_argument("--synthetic_dataset", type=str, required=True,
                      help="Path to the synthetic dataset directory")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save the generated datasets")
    
    # Split parameters
    parser.add_argument("--val_size", type=float, default=0.2,
                      help="Proportion of data to use for validation (default: 0.2)")
    
    # Generation parameters
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    return parser.parse_args()

def read_labels(label_path):
    """
    Read label.npy file and parse the samples by category.
    
    Returns:
        Dictionary mapping label (0, 1, 2) to list of feature file IDs
        Full label array
    """
    print(f"Reading labels from: {label_path}")
    
    # Load the labels
    labels = np.load(label_path)
    print(f"Loaded label file with shape: {labels.shape}")
    
    # Group by label
    label_to_features = defaultdict(list)
    for entry in labels:
        label = int(entry[0])  # First column is label
        feature_id = int(entry[1])  # Second column is subject_id
        label_to_features[label].append(feature_id)
    
    # Print summary
    print(f"Found {len(label_to_features[0])} HC samples, " +
          f"{len(label_to_features[1])} MCI samples, " +
          f"{len(label_to_features[2])} Dementia samples")
    
    return label_to_features, labels

def copy_feature(src_dir, feature_id, dst_dir, new_id, prefix="feature_"):
    """Copy a feature file with a new ID."""
    src_file = os.path.join(src_dir, f"{prefix}{feature_id:02d}.npy")
    dst_file = os.path.join(dst_dir, f"{prefix}{new_id:02d}.npy")
    
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
        return True
    else:
        print(f"Warning: Source file not found: {src_file}")
        return False

def create_discrimination_dataset(genuine_path, synthetic_path, output_dir, 
                                 dementia_class, val_size=0.2, seed=42):
    """
    Create a balanced dataset for genuine vs synthetic discrimination for a specific dementia class.
    
    Args:
        genuine_path: Path to genuine dataset
        synthetic_path: Path to synthetic dataset
        output_dir: Path to save the dataset
        dementia_class: Dementia class to use (0=HC, 1=MCI, 2=Dementia, None=All)
        val_size: Proportion of data to use for validation
        seed: Random seed for reproducibility
    """
    # Set class name for output
    if dementia_class == 0:
        class_name = "hc"
    elif dementia_class == 1:
        class_name = "mci"
    elif dementia_class == 2:
        class_name = "dementia"
    else:
        class_name = "all"
    
    print(f"\n{'='*80}")
    print(f"Creating discrimination dataset for class: {class_name}")
    print(f"{'='*80}")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    train_dir = os.path.join(output_dir, f"{class_name}_train")
    val_dir = os.path.join(output_dir, f"{class_name}_val")
    
    train_feature_dir = os.path.join(train_dir, "Feature")
    train_label_dir = os.path.join(train_dir, "Label")
    val_feature_dir = os.path.join(val_dir, "Feature")
    val_label_dir = os.path.join(val_dir, "Label")
    
    os.makedirs(train_feature_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_feature_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Read labels from both datasets
    genuine_label_path = os.path.join(genuine_path, "Label", "label.npy")
    synthetic_label_path = os.path.join(synthetic_path, "Label", "label.npy")
    
    genuine_label_dict, genuine_labels = read_labels(genuine_label_path)
    synthetic_label_dict, synthetic_labels = read_labels(synthetic_label_path)
    
    # Get feature IDs for the specified dementia class (or all classes)
    if dementia_class is not None:
        genuine_features = genuine_label_dict[dementia_class]
        synthetic_features = synthetic_label_dict[dementia_class]
        class_list = [dementia_class]
    else:
        # Combine all classes
        genuine_features = []
        synthetic_features = []
        for cls in [0, 1, 2]:
            genuine_features.extend(genuine_label_dict[cls])
            synthetic_features.extend(synthetic_label_dict[cls])
        class_list = [0, 1, 2]
    
    print(f"Found {len(genuine_features)} genuine and {len(synthetic_features)} synthetic samples")
    
    # Balance the dataset by taking the minimum number of samples from each source
    min_samples = min(len(genuine_features), len(synthetic_features))
    if min_samples == 0:
        print(f"Error: No samples available for one source, skipping")
        return
    
    # Randomly select balanced samples
    if len(genuine_features) > min_samples:
        genuine_features = random.sample(genuine_features, min_samples)
    if len(synthetic_features) > min_samples:
        synthetic_features = random.sample(synthetic_features, min_samples)
    
    print(f"Using {min_samples} samples from each source for a balanced dataset")
    
    # Create arrays for labels and features
    features = []  # Will store tuples (feature_id, is_synthetic, original_class)
    
    # Add genuine samples (label 0)
    for feature_id in genuine_features:
        original_class = None
        for label, feature_ids in genuine_label_dict.items():
            if feature_id in feature_ids:
                original_class = label
                break
        features.append((feature_id, 0, original_class, "genuine"))  # 0 = genuine
    
    # Add synthetic samples (label 1)
    for feature_id in synthetic_features:
        original_class = None
        for label, feature_ids in synthetic_label_dict.items():
            if feature_id in feature_ids:
                original_class = label
                break
        features.append((feature_id, 1, original_class, "synthetic"))  # 1 = synthetic
    
    # Shuffle the features
    random.shuffle(features)
    
    # Split into train and validation sets
    train_features, val_features = train_test_split(
        features, test_size=val_size, random_state=seed, 
        stratify=[f[1] for f in features]  # Stratify by genuine/synthetic
    )
    
    print(f"Split into {len(train_features)} train and {len(val_features)} validation samples")
    
    # Copy files and create labels
    train_labels = []
    val_labels = []
    
    # Process training set
    for i, (feature_id, is_synthetic, original_class, source) in enumerate(train_features):
        src_dir = os.path.join(synthetic_path if is_synthetic else genuine_path, "Feature")
        if copy_feature(src_dir, feature_id, train_feature_dir, i):
            train_labels.append([is_synthetic, i, original_class])
    
    # Process validation set
    for i, (feature_id, is_synthetic, original_class, source) in enumerate(val_features):
        src_dir = os.path.join(synthetic_path if is_synthetic else genuine_path, "Feature")
        if copy_feature(src_dir, feature_id, val_feature_dir, i):
            val_labels.append([is_synthetic, i, original_class])
    
    # Save label files
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    
    np.save(os.path.join(train_label_dir, "label.npy"), train_labels)
    np.save(os.path.join(val_label_dir, "label.npy"), val_labels)
    
    # Save metadata about the dataset
    metadata = {
        "description": f"Discrimination dataset for {class_name}",
        "label_meaning": "0=genuine, 1=synthetic",
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "genuine_count": len([1 for l in train_labels if l[0] == 0]) + len([1 for l in val_labels if l[0] == 0]),
        "synthetic_count": len([1 for l in train_labels if l[0] == 1]) + len([1 for l in val_labels if l[0] == 1]),
    }
    
    # Print class distribution
    train_genuine = len([1 for l in train_labels if l[0] == 0])
    train_synthetic = len([1 for l in train_labels if l[0] == 1])
    val_genuine = len([1 for l in val_labels if l[0] == 0])
    val_synthetic = len([1 for l in val_labels if l[0] == 1])
    
    print(f"Train set: {train_genuine} genuine, {train_synthetic} synthetic")
    print(f"Val set: {val_genuine} genuine, {val_synthetic} synthetic")
    
    # Save metadata
    with open(os.path.join(output_dir, f"{class_name}_metadata.txt"), "w") as f:
        f.write(f"Discrimination dataset for {class_name}\n")
        f.write(f"Label meaning: 0=genuine, 1=synthetic\n")
        f.write(f"Train samples: {len(train_labels)}\n")
        f.write(f"Val samples: {len(val_labels)}\n")
        f.write(f"Train set: {train_genuine} genuine, {train_synthetic} synthetic\n")
        f.write(f"Val set: {val_genuine} genuine, {val_synthetic} synthetic\n")
    
    print(f"Dataset created successfully: {output_dir}/{class_name}_train and {output_dir}/{class_name}_val")

def main():
    args = parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Creating genuine vs synthetic discrimination datasets")
    
    # Create individual class datasets
    for dementia_class in [0, 1, 2]:  # HC, MCI, Dementia
        create_discrimination_dataset(
            args.genuine_dataset,
            args.synthetic_dataset,
            args.output_dir,
            dementia_class,
            args.val_size,
            args.seed
        )
    
    # Create combined dataset with all classes
    create_discrimination_dataset(
        args.genuine_dataset,
        args.synthetic_dataset,
        args.output_dir,
        None,  # None means all classes
        args.val_size,
        args.seed
    )
    
    print("\nAll discrimination datasets created successfully!")
    print("\nEach dataset is split into train and validation sets with labels:")
    print("  - 0: genuine samples")
    print("  - 1: synthetic samples")
    print("\nAvailable datasets:")
    print(f"  - HC only: {args.output_dir}/hc_train and {args.output_dir}/hc_val")
    print(f"  - MCI only: {args.output_dir}/mci_train and {args.output_dir}/mci_val")
    print(f"  - Dementia only: {args.output_dir}/dementia_train and {args.output_dir}/dementia_val")
    print(f"  - All classes: {args.output_dir}/all_train and {args.output_dir}/all_val")

if __name__ == "__main__":
    main()