#!/usr/bin/env python
"""
Create datasets for evaluating genuine vs non-genuine EEG discrimination.
This script:
1. Creates balanced datasets (equal genuine and non-genuine samples)
2. Creates separate datasets for each dementia class (HC, MCI, Dementia)
3. Creates a combined dataset with all classes
4. Relabels samples as genuine (0) or non-genuine (1)
5. Splits each dataset into train and validation sets

The non-genuine data can be synthetic, augmented, or any other type.
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
    parser = argparse.ArgumentParser(description="Create genuine vs non-genuine discrimination datasets")
    
    # Path to datasets
    parser.add_argument("--genuine_dataset", type=str, required=True,
                      help="Path to the genuine dataset directory")
    parser.add_argument("--comparison_dataset", type=str, required=True,
                      help="Path to the comparison dataset directory (synthetic, augmented, etc.)")
    
    # Data type specification
    parser.add_argument("--comparison_type", type=str, default="synthetic",
                      choices=["synthetic", "augmented", "generated", "modified"],
                      help="Type of comparison data (default: synthetic)")
    
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

def create_discrimination_dataset(genuine_path, comparison_path, output_dir, 
                                 dementia_class, comparison_type, val_size=0.2, seed=42):
    """
    Create a balanced dataset for genuine vs comparison discrimination for a specific dementia class.
    
    Args:
        genuine_path: Path to genuine dataset
        comparison_path: Path to comparison dataset (synthetic, augmented, etc.)
        output_dir: Path to save the dataset
        dementia_class: Dementia class to use (0=HC, 1=MCI, 2=Dementia, None=All)
        comparison_type: Type of comparison data ("synthetic", "augmented", etc.)
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
    print(f"Comparing genuine vs {comparison_type} data")
    print(f"{'='*80}")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories with comparison type in name
    train_dir = os.path.join(output_dir, f"{class_name}_{comparison_type}_train")
    val_dir = os.path.join(output_dir, f"{class_name}_{comparison_type}_val")
    
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
    comparison_label_path = os.path.join(comparison_path, "Label", "label.npy")
    
    genuine_label_dict, genuine_labels = read_labels(genuine_label_path)
    comparison_label_dict, comparison_labels = read_labels(comparison_label_path)
    
    # Get feature IDs for the specified dementia class (or all classes)
    if dementia_class is not None:
        genuine_features = genuine_label_dict[dementia_class]
        comparison_features = comparison_label_dict[dementia_class]
        class_list = [dementia_class]
    else:
        # Combine all classes
        genuine_features = []
        comparison_features = []
        for cls in [0, 1, 2]:
            genuine_features.extend(genuine_label_dict[cls])
            comparison_features.extend(comparison_label_dict[cls])
        class_list = [0, 1, 2]
    
    print(f"Found {len(genuine_features)} genuine and {len(comparison_features)} {comparison_type} samples")
    
    # Balance the dataset by taking the minimum number of samples from each source
    min_samples = min(len(genuine_features), len(comparison_features))
    if min_samples == 0:
        print(f"Error: No samples available for one source, skipping")
        return
    
    # Randomly select balanced samples
    if len(genuine_features) > min_samples:
        genuine_features = random.sample(genuine_features, min_samples)
    if len(comparison_features) > min_samples:
        comparison_features = random.sample(comparison_features, min_samples)
    
    print(f"Using {min_samples} samples from each source for a balanced dataset")
    
    # Create arrays for labels and features
    features = []  # Will store tuples (feature_id, is_comparison, original_class, source)
    
    # Add genuine samples (label 0)
    for feature_id in genuine_features:
        original_class = None
        for label, feature_ids in genuine_label_dict.items():
            if feature_id in feature_ids:
                original_class = label
                break
        features.append((feature_id, 0, original_class, "genuine"))  # 0 = genuine
    
    # Add comparison samples (label 1)
    for feature_id in comparison_features:
        original_class = None
        for label, feature_ids in comparison_label_dict.items():
            if feature_id in feature_ids:
                original_class = label
                break
        features.append((feature_id, 1, original_class, comparison_type))  # 1 = comparison
    
    # Shuffle the features
    random.shuffle(features)
    
    # Split into train and validation sets
    train_features, val_features = train_test_split(
        features, test_size=val_size, random_state=seed, 
        stratify=[f[1] for f in features]  # Stratify by genuine/comparison
    )
    
    print(f"Split into {len(train_features)} train and {len(val_features)} validation samples")
    
    # Copy files and create labels
    train_labels = []
    val_labels = []
    
    # Process training set
    for i, (feature_id, is_comparison, original_class, source) in enumerate(train_features):
        src_dir = os.path.join(comparison_path if is_comparison else genuine_path, "Feature")
        if copy_feature(src_dir, feature_id, train_feature_dir, i):
            train_labels.append([is_comparison, i, original_class])
    
    # Process validation set
    for i, (feature_id, is_comparison, original_class, source) in enumerate(val_features):
        src_dir = os.path.join(comparison_path if is_comparison else genuine_path, "Feature")
        if copy_feature(src_dir, feature_id, val_feature_dir, i):
            val_labels.append([is_comparison, i, original_class])
    
    # Save label files
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    
    np.save(os.path.join(train_label_dir, "label.npy"), train_labels)
    np.save(os.path.join(val_label_dir, "label.npy"), val_labels)
    
    # Print class distribution
    train_genuine = len([1 for l in train_labels if l[0] == 0])
    train_comparison = len([1 for l in train_labels if l[0] == 1])
    val_genuine = len([1 for l in val_labels if l[0] == 0])
    val_comparison = len([1 for l in val_labels if l[0] == 1])
    
    print(f"Train set: {train_genuine} genuine, {train_comparison} {comparison_type}")
    print(f"Val set: {val_genuine} genuine, {val_comparison} {comparison_type}")
    
    # Save metadata
    with open(os.path.join(output_dir, f"{class_name}_{comparison_type}_metadata.txt"), "w") as f:
        f.write(f"Discrimination dataset for {class_name}\n")
        f.write(f"Comparing genuine vs {comparison_type} data\n")
        f.write(f"Label meaning: 0=genuine, 1={comparison_type}\n")
        f.write(f"Train samples: {len(train_labels)}\n")
        f.write(f"Val samples: {len(val_labels)}\n")
        f.write(f"Train set: {train_genuine} genuine, {train_comparison} {comparison_type}\n")
        f.write(f"Val set: {val_genuine} genuine, {val_comparison} {comparison_type}\n")
        f.write(f"\nDataset structure:\n")
        f.write(f"  - {train_dir}/\n")
        f.write(f"  - {val_dir}/\n")
        f.write(f"\nUsage: Train a binary classifier to distinguish between genuine and {comparison_type} data\n")
    
    print(f"Dataset created successfully: {output_dir}/{class_name}_{comparison_type}_train and {output_dir}/{class_name}_{comparison_type}_val")

def main():
    args = parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Creating genuine vs {args.comparison_type} discrimination datasets")
    print(f"Genuine dataset: {args.genuine_dataset}")
    print(f"Comparison dataset: {args.comparison_dataset}")
    print(f"Comparison type: {args.comparison_type}")
    
    # Create individual class datasets
    for dementia_class in [0, 1, 2]:  # HC, MCI, Dementia
        create_discrimination_dataset(
            args.genuine_dataset,
            args.comparison_dataset,
            args.output_dir,
            dementia_class,
            args.comparison_type,
            args.val_size,
            args.seed
        )
    
    # Create combined dataset with all classes
    create_discrimination_dataset(
        args.genuine_dataset,
        args.comparison_dataset,
        args.output_dir,
        None,  # None means all classes
        args.comparison_type,
        args.val_size,
        args.seed
    )
    
    print(f"\nAll discrimination datasets created successfully!")
    print(f"\nEach dataset is split into train and validation sets with labels:")
    print(f"  - 0: genuine samples")
    print(f"  - 1: {args.comparison_type} samples")
    print(f"\nAvailable datasets:")
    print(f"  - HC only: {args.output_dir}/hc_{args.comparison_type}_train and {args.output_dir}/hc_{args.comparison_type}_val")
    print(f"  - MCI only: {args.output_dir}/mci_{args.comparison_type}_train and {args.output_dir}/mci_{args.comparison_type}_val")
    print(f"  - Dementia only: {args.output_dir}/dementia_{args.comparison_type}_train and {args.output_dir}/dementia_{args.comparison_type}_val")
    print(f"  - All classes: {args.output_dir}/all_{args.comparison_type}_train and {args.output_dir}/all_{args.comparison_type}_val")
    
    # Create a summary file
    summary_file = os.path.join(args.output_dir, f"discrimination_summary_{args.comparison_type}.txt")
    with open(summary_file, "w") as f:
        f.write(f"Discrimination Dataset Summary\n")
        f.write(f"============================\n\n")
        f.write(f"Comparison type: {args.comparison_type}\n")
        f.write(f"Genuine dataset: {args.genuine_dataset}\n")
        f.write(f"Comparison dataset: {args.comparison_dataset}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"Validation split: {args.val_size}\n")
        f.write(f"Random seed: {args.seed}\n\n")
        f.write(f"Created datasets:\n")
        f.write(f"  - hc_{args.comparison_type}_train/val\n")
        f.write(f"  - mci_{args.comparison_type}_train/val\n")
        f.write(f"  - dementia_{args.comparison_type}_train/val\n")
        f.write(f"  - all_{args.comparison_type}_train/val\n\n")
        f.write(f"Label encoding: 0=genuine, 1={args.comparison_type}\n")
    
    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    main()