#!/usr/bin/env python
"""
Create train/test splits from genuine data and augment the training set with synthetic data.
This script:
1. Splits genuine data into train/test sets (keeping test set 100% genuine)
2. Augments only the training set with synthetic data by specified percentages
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
    parser = argparse.ArgumentParser(description="Create train/test splits and augment training data")
    
    # Path to datasets
    parser.add_argument("--genuine_dataset", type=str, required=True,
                      help="Path to the genuine dataset directory")
    parser.add_argument("--synthetic_dataset", type=str, required=True,
                      help="Path to the synthetic dataset directory")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save the split and augmented datasets")
    
    # Split parameters
    parser.add_argument("--test_size", type=float, default=0.2,
                      help="Proportion of genuine data to use for testing (default: 0.2)")
    parser.add_argument("--stratify", action="store_true", default=True,
                      help="Whether to stratify the split by class labels")
    
    # Augmentation percentages
    parser.add_argument("--percentages", type=str, default="20,40,60,100",
                      help="Comma-separated list of augmentation percentages (e.g., '20,40,60,100')")
    
    # Generation parameters
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    return parser.parse_args()

def get_available_feature_files(feature_dir, prefix="feature_"):
    """
    Get list of available feature file IDs in the directory.
    
    Args:
        feature_dir: Directory containing feature files
        prefix: Prefix of feature files (default: "feature_")
    
    Returns:
        Set of available feature IDs
    """
    available_ids = set()
    
    if not os.path.exists(feature_dir):
        print(f"Warning: Feature directory does not exist: {feature_dir}")
        return available_ids
    
    for filename in os.listdir(feature_dir):
        if filename.startswith(prefix) and filename.endswith('.npy'):
            try:
                # Extract ID from filename (e.g., "feature_01.npy" -> 1)
                id_str = filename[len(prefix):-4]  # Remove prefix and .npy
                feature_id = int(id_str)
                available_ids.add(feature_id)
            except ValueError:
                # Skip files that don't match the expected pattern
                continue
    
    print(f"Found {len(available_ids)} available feature files in {feature_dir}")
    return available_ids

def read_labels(label_path, feature_dir=None):
    """
    Read label.npy file and parse the samples by category.
    Optionally filter by available feature files.
    
    Args:
        label_path: Path to label.npy file
        feature_dir: Optional path to feature directory for validation
    
    Returns:
        Dictionary mapping label (0, 1, 2) to list of feature file numbers
        Full label array (potentially filtered)
    """
    print(f"Reading labels from: {label_path}")
    
    # Load the labels
    labels = np.load(label_path)
    print(f"Loaded label file with shape: {labels.shape}")
    
    # Get available feature files if feature_dir is provided
    available_features = None
    if feature_dir:
        available_features = get_available_feature_files(feature_dir)
    
    # Group by label and filter by available files
    label_to_features = defaultdict(list)
    filtered_labels = []
    
    for entry in labels:
        label = int(entry[0])  # First column is label
        subject_id = int(entry[1])  # Second column is subject_id
        
        # Only include if feature file exists (when validation is enabled)
        if available_features is None or subject_id in available_features:
            label_to_features[label].append(subject_id)
            filtered_labels.append(entry)
        else:
            print(f"Warning: Feature file feature_{subject_id:02d}.npy not found, skipping")
    
    # Convert back to numpy array
    filtered_labels = np.array(filtered_labels) if filtered_labels else np.array([])
    
    # Print summary
    if filtered_labels.size > 0:
        print(f"Found {len(label_to_features[0])} HC samples, " +
              f"{len(label_to_features[1])} MCI samples, " +
              f"{len(label_to_features[2])} Dementia samples")
        
        if available_features is not None:
            original_count = len(labels)
            filtered_count = len(filtered_labels)
            if original_count != filtered_count:
                print(f"Filtered from {original_count} to {filtered_count} samples based on available files")
    else:
        print("Warning: No valid samples found after filtering")
    
    return label_to_features, filtered_labels

def copy_features(feature_ids, src_dir, dst_dir, prefix="feature_"):
    """Copy feature files with the given IDs from source to destination."""
    os.makedirs(dst_dir, exist_ok=True)
    
    copied_count = 0
    for feature_id in feature_ids:
        src_file = os.path.join(src_dir, f"{prefix}{feature_id:02d}.npy")
        dst_file = os.path.join(dst_dir, f"{prefix}{feature_id:02d}.npy")
        
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            copied_count += 1
        else:
            print(f"Warning: Source file not found: {src_file}")
    
    return copied_count

def create_train_test_split(genuine_path, output_dir, test_size=0.2, stratify=True, seed=42):
    """
    Split genuine dataset into train and test sets.
    
    Args:
        genuine_path: Path to genuine dataset
        output_dir: Base path to save split datasets
        test_size: Proportion of data to use for testing
        stratify: Whether to stratify the split by class labels
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping labels to feature IDs for the training set
        List of all feature IDs in the training set
    """
    print(f"\n{'='*80}")
    print(f"Creating train/test split (test_size={test_size}, stratify={stratify})")
    print(f"{'='*80}")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train_genuine")
    test_dir = os.path.join(output_dir, "test_genuine")
    
    train_feature_dir = os.path.join(train_dir, "Feature")
    train_label_dir = os.path.join(train_dir, "Label")
    test_feature_dir = os.path.join(test_dir, "Feature")
    test_label_dir = os.path.join(test_dir, "Label")
    
    os.makedirs(train_feature_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_feature_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    
    # Read genuine labels with feature file validation
    genuine_label_path = os.path.join(genuine_path, "Label", "label.npy")
    genuine_feature_dir = os.path.join(genuine_path, "Feature")
    genuine_label_dict, genuine_labels = read_labels(genuine_label_path, genuine_feature_dir)
    
    # Create arrays for sklearn's train_test_split
    features = []
    labels = []
    
    for idx, entry in enumerate(genuine_labels):
        label = int(entry[0])
        feature_id = int(entry[1])
        features.append(feature_id)
        labels.append(label)
    
    # Perform train/test split
    if stratify:
        features_train, features_test, labels_train, labels_test = train_test_split(
            features, labels, test_size=test_size, random_state=seed, stratify=labels
        )
    else:
        features_train, features_test, labels_train, labels_test = train_test_split(
            features, labels, test_size=test_size, random_state=seed
        )
    
    print(f"Split genuine dataset: {len(features_train)} train samples, {len(features_test)} test samples")
    
    # Create label arrays
    train_labels = []
    test_labels = []
    
    for i, feature_id in enumerate(features_train):
        train_labels.append([labels_train[i], feature_id])
    
    for i, feature_id in enumerate(features_test):
        test_labels.append([labels_test[i], feature_id])
    
    # Save label files
    np.save(os.path.join(train_label_dir, "label.npy"), np.array(train_labels))
    np.save(os.path.join(test_label_dir, "label.npy"), np.array(test_labels))
    
    # Copy feature files
    train_copied = copy_features(features_train, genuine_feature_dir, train_feature_dir)
    test_copied = copy_features(features_test, genuine_feature_dir, test_feature_dir)
    
    print(f"Copied {train_copied} feature files to train set and {test_copied} feature files to test set")
    
    # Create train set label dictionary for augmentation
    train_label_dict = defaultdict(list)
    for i, feature_id in enumerate(features_train):
        label = labels_train[i]
        train_label_dict[label].append(feature_id)
    
    # Print class distribution
    train_dist = {label: len(ids) for label, ids in train_label_dict.items()}
    test_dist = defaultdict(int)
    for label in labels_test:
        test_dist[label] += 1
    
    print(f"Train set class distribution: {train_dist}")
    print(f"Test set class distribution: {dict(test_dist)}")
    
    return train_label_dict, features_train

def augment_training_set(genuine_train_dict, genuine_train_ids, synthetic_path, output_dir, 
                        percentage, seed=42):
    """
    Augment the training set with synthetic data.
    
    Args:
        genuine_train_dict: Dictionary mapping labels to feature IDs for the genuine training set
        genuine_train_ids: List of all feature IDs in the genuine training set
        synthetic_path: Path to synthetic dataset
        output_dir: Path to save the augmented training set
        percentage: Percentage of synthetic data to add
        seed: Random seed for reproducibility
    """
    print(f"\n{'='*80}")
    print(f"Augmenting training set with +{percentage}% synthetic data")
    print(f"{'='*80}")
    
    # Set random seed
    random.seed(seed)
    
    # Create output directories
    output_feature_dir = os.path.join(output_dir, "Feature")
    output_label_dir = os.path.join(output_dir, "Label")
    os.makedirs(output_feature_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Read synthetic labels with feature file validation
    synthetic_label_path = os.path.join(synthetic_path, "Label", "label.npy")
    synthetic_feature_dir = os.path.join(synthetic_path, "Feature")
    synthetic_label_dict, synthetic_labels = read_labels(synthetic_label_path, synthetic_feature_dir)
    
    # Calculate number of samples to add for each category
    samples_to_add = {}
    for label in [0, 1, 2]:  # HC, MCI, Dementia
        genuine_count = len(genuine_train_dict[label])
        add_count = int(genuine_count * percentage / 100)
        available_count = len(synthetic_label_dict[label])
        
        if add_count > available_count:
            print(f"Warning: Requested {add_count} samples for category {label}, but only {available_count} available")
            add_count = min(add_count, available_count)
        
        samples_to_add[label] = add_count
        print(f"Category {label}: Adding {add_count} synthetic samples to {genuine_count} genuine samples")
    
    # First, copy all genuine training files to the output directory
    print("Copying genuine training files...")
    genuine_train_dir = os.path.join(output_dir, "..", "train_genuine", "Feature")
    
    for feature_id in genuine_train_ids:
        src_file = os.path.join(genuine_train_dir, f"feature_{feature_id:02d}.npy")
        dst_file = os.path.join(output_feature_dir, f"feature_{feature_id:02d}.npy")
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
    
    # Find the highest feature_id in the genuine training set
    max_feature_id = max(genuine_train_ids) if genuine_train_ids else 0
    print(f"Highest feature_id in genuine training set: {max_feature_id}")
    
    # Initialize new labels array with genuine training labels
    augmented_labels = []
    for label, feature_ids in genuine_train_dict.items():
        for feature_id in feature_ids:
            augmented_labels.append([label, feature_id])
    
    # Add synthetic samples for each category
    for label in [0, 1, 2]:
        # Get available synthetic feature IDs for this category
        available_features = synthetic_label_dict[label].copy()
        
        # Randomly select the required number of samples
        num_to_add = samples_to_add[label]
        if num_to_add > 0 and len(available_features) > 0:
            selected_features = random.sample(available_features, num_to_add)
            
            print(f"Adding {len(selected_features)} synthetic samples for category {label}")
            
            # Copy selected synthetic features with new IDs
            next_id = max_feature_id + 1
            for i, feature_id in enumerate(selected_features):
                # Source synthetic file
                src_filename = f"feature_{feature_id:02d}.npy"
                src_path = os.path.join(synthetic_feature_dir, src_filename)
                
                # Destination with new ID
                new_id = next_id + i
                dst_filename = f"feature_{new_id:02d}.npy"
                dst_path = os.path.join(output_feature_dir, dst_filename)
                
                # Double-check that source file exists before copying
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    # Add to augmented labels
                    augmented_labels.append([label, new_id])
                else:
                    print(f"Error: Source file {src_path} does not exist, skipping")
            
            # Update max_feature_id
            max_feature_id += len(selected_features)
        elif num_to_add > 0:
            print(f"Warning: No synthetic samples available for category {label}")
    
    # Save augmented labels
    augmented_labels = np.array(augmented_labels)
    np.save(os.path.join(output_label_dir, "label.npy"), augmented_labels)
    
    # Print summary
    original_count = len(genuine_train_ids)
    added_count = len(augmented_labels) - original_count
    print(f"Augmented training set: {original_count} genuine + {added_count} synthetic = {len(augmented_labels)} total samples")
    print(f"Files saved to: {output_dir}")

def main():
    args = parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse percentages
    try:
        percentages = [int(p) for p in args.percentages.split(',')]
    except ValueError:
        print(f"Error: Invalid percentages format. Using default: 20,40,60,100")
        percentages = [20, 40, 60, 100]
    
    print(f"Will create splits and augmented training sets with percentages: {percentages}")
    
    # First, create train/test split with only genuine data
    genuine_train_dict, genuine_train_ids = create_train_test_split(
        args.genuine_dataset,
        args.output_dir,
        args.test_size,
        args.stratify,
        args.seed
    )
    
    # Then, create augmented training sets for each percentage
    for percentage in percentages:
        output_path = os.path.join(args.output_dir, f"train_augmented_{percentage}pct")
        augment_training_set(
            genuine_train_dict,
            genuine_train_ids,
            args.synthetic_dataset,
            output_path,
            percentage,
            args.seed
        )
    
    print("\nAll datasets created successfully!")
    print("\nTo use these datasets for training and evaluation:")
    print(f"1. Train on any augmented set: {args.output_dir}/train_augmented_XXpct")
    print(f"2. Test on the genuine test set: {args.output_dir}/test_genuine")

if __name__ == "__main__":
    main()