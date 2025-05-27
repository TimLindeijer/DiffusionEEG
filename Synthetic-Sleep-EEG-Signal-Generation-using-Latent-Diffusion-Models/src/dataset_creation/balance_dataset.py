#!/usr/bin/env python
"""
Create train/test splits from genuine data and balance the training set with synthetic data.
This script:
1. Splits genuine data into train/test sets (keeping test set 100% genuine)
2. Balances the training set by adding synthetic data to make all classes equal in size
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
    parser = argparse.ArgumentParser(description="Create train/test splits and balance training data")
    
    # Path to datasets
    parser.add_argument("--genuine_dataset", type=str, required=True,
                      help="Path to the genuine dataset directory")
    parser.add_argument("--synthetic_dataset", type=str, required=True,
                      help="Path to the synthetic dataset directory")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save the split and balanced datasets")
    
    # Split parameters
    parser.add_argument("--test_size", type=float, default=0.2,
                      help="Proportion of genuine data to use for testing (default: 0.2)")
    parser.add_argument("--stratify", action="store_true", default=True,
                      help="Whether to stratify the split by class labels")
    
    # Balance strategy
    parser.add_argument("--balance_to", type=str, default="max",
                      help="Balance strategy: 'max' (to largest class), 'mean' (to mean size), or a number")
    
    # Generation parameters
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    return parser.parse_args()

def read_labels(label_path):
    """
    Read label.npy file and parse the samples by category.
    
    Returns:
        Dictionary mapping label (0, 1, 2) to list of feature file numbers
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
        subject_id = int(entry[1])  # Second column is subject_id
        label_to_features[label].append(subject_id)
    
    # Print summary
    print(f"Found {len(label_to_features[0])} HC samples, " +
          f"{len(label_to_features[1])} MCI samples, " +
          f"{len(label_to_features[2])} Dementia samples")
    
    return label_to_features, labels

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
        Dictionary mapping labels to feature IDs for the test set
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
    
    # Read genuine labels
    genuine_label_path = os.path.join(genuine_path, "Label", "label.npy")
    genuine_label_dict, genuine_labels = read_labels(genuine_label_path)
    
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
    genuine_feature_dir = os.path.join(genuine_path, "Feature")
    train_copied = copy_features(features_train, genuine_feature_dir, train_feature_dir)
    test_copied = copy_features(features_test, genuine_feature_dir, test_feature_dir)
    
    print(f"Copied {train_copied} feature files to train set and {test_copied} feature files to test set")
    
    # Create train and test set label dictionaries
    train_label_dict = defaultdict(list)
    for i, feature_id in enumerate(features_train):
        label = labels_train[i]
        train_label_dict[label].append(feature_id)
    
    test_label_dict = defaultdict(list)
    for i, feature_id in enumerate(features_test):
        label = labels_test[i]
        test_label_dict[label].append(feature_id)
    
    # Print class distribution
    train_dist = {label: len(ids) for label, ids in train_label_dict.items()}
    test_dist = {label: len(ids) for label, ids in test_label_dict.items()}
    
    print(f"Train set class distribution: {train_dist}")
    print(f"Test set class distribution: {test_dist}")
    
    return train_label_dict, features_train, test_label_dict

def balance_training_set(genuine_train_dict, genuine_train_ids, synthetic_path, output_dir, 
                        balance_to="max", seed=42):
    """
    Balance the training set with synthetic data.
    
    Args:
        genuine_train_dict: Dictionary mapping labels to feature IDs for the genuine training set
        genuine_train_ids: List of all feature IDs in the genuine training set
        synthetic_path: Path to synthetic dataset
        output_dir: Path to save the balanced training set
        balance_to: Strategy for balancing - "max", "mean", or a specific number
        seed: Random seed for reproducibility
    """
    print(f"\n{'='*80}")
    print(f"Balancing training set with synthetic data (strategy: {balance_to})")
    print(f"{'='*80}")
    
    # Set random seed
    random.seed(seed)
    
    # Create output directories
    output_feature_dir = os.path.join(output_dir, "Feature")
    output_label_dir = os.path.join(output_dir, "Label")
    os.makedirs(output_feature_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Read synthetic labels
    synthetic_label_path = os.path.join(synthetic_path, "Label", "label.npy")
    synthetic_label_dict, synthetic_labels = read_labels(synthetic_label_path)
    
    # Determine target size for each class
    class_sizes = {label: len(ids) for label, ids in genuine_train_dict.items()}
    
    if balance_to == "max":
        target_size = max(class_sizes.values())
    elif balance_to == "mean":
        target_size = int(sum(class_sizes.values()) / len(class_sizes))
    else:
        try:
            target_size = int(balance_to)
        except ValueError:
            print(f"Invalid balance_to value: {balance_to}. Using 'max' instead.")
            target_size = max(class_sizes.values())
    
    print(f"Target size for each class: {target_size}")
    print(f"Current class sizes in training set: {class_sizes}")
    
    # Calculate number of samples to add for each category
    samples_to_add = {}
    for label in [0, 1, 2]:  # HC, MCI, Dementia
        genuine_count = class_sizes.get(label, 0)
        add_count = target_size - genuine_count
        available_count = len(synthetic_label_dict[label])
        
        if add_count > available_count:
            print(f"Warning: Need {add_count} samples for category {label}, but only {available_count} available")
            add_count = min(add_count, available_count)
        
        samples_to_add[label] = max(0, add_count)  # Ensure non-negative
        print(f"Category {label}: Adding {samples_to_add[label]} synthetic samples to {genuine_count} genuine samples")
    
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
    balanced_labels = []
    for label, feature_ids in genuine_train_dict.items():
        for feature_id in feature_ids:
            balanced_labels.append([label, feature_id])
    
    # Add synthetic samples for each category
    for label in [0, 1, 2]:
        # Get available synthetic feature IDs for this category
        available_features = synthetic_label_dict[label].copy()
        
        # Randomly select the required number of samples
        num_to_add = samples_to_add[label]
        if num_to_add > 0:
            selected_features = random.sample(available_features, num_to_add)
            
            print(f"Adding {len(selected_features)} synthetic samples for category {label}")
            
            # Copy selected synthetic features with new IDs
            next_id = max_feature_id + 1
            for i, feature_id in enumerate(selected_features):
                # Source synthetic file
                src_filename = f"feature_{feature_id:02d}.npy"
                src_path = os.path.join(synthetic_path, "Feature", src_filename)
                
                # Destination with new ID
                new_id = next_id + i
                dst_filename = f"feature_{new_id:02d}.npy"
                dst_path = os.path.join(output_feature_dir, dst_filename)
                
                # Copy file with new ID
                shutil.copy2(src_path, dst_path)
                
                # Add to balanced labels
                balanced_labels.append([label, new_id])
            
            # Update max_feature_id
            max_feature_id += len(selected_features)
    
    # Save balanced labels
    balanced_labels = np.array(balanced_labels)
    np.save(os.path.join(output_label_dir, "label.npy"), balanced_labels)
    
    # Print summary
    original_count = len(genuine_train_ids)
    added_count = len(balanced_labels) - original_count
    print(f"Balanced training set: {original_count} genuine + {added_count} synthetic = {len(balanced_labels)} total samples")
    
    # Print final class distribution
    final_dist = defaultdict(int)
    for label, _ in balanced_labels:
        final_dist[int(label)] += 1
    print(f"Final balanced class distribution: {dict(final_dist)}")
    print(f"Files saved to: {output_dir}")

def main():
    args = parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Will create balanced training set using '{args.balance_to}' strategy")
    
    # First, create train/test split with only genuine data
    genuine_train_dict, genuine_train_ids, test_dict = create_train_test_split(
        args.genuine_dataset,
        args.output_dir,
        args.test_size,
        args.stratify,
        args.seed
    )
    
    # Then, create balanced training set
    output_path = os.path.join(args.output_dir, "train_balanced")
    balance_training_set(
        genuine_train_dict,
        genuine_train_ids,
        args.synthetic_dataset,
        output_path,
        args.balance_to,
        args.seed
    )
    
    print("\nDatasets created successfully!")
    print("\nTo use these datasets for training and evaluation:")
    print(f"1. Train on the balanced set: {args.output_dir}/train_balanced")
    print(f"2. Test on the genuine test set: {args.output_dir}/test_genuine")
    
    # Print test set info for reference
    print(f"\nTest set class distribution (100% genuine):")
    test_dist = {label: len(ids) for label, ids in test_dict.items()}
    print(f"  HC: {test_dist.get(0, 0)} samples")
    print(f"  MCI: {test_dist.get(1, 0)} samples")
    print(f"  Dementia: {test_dist.get(2, 0)} samples")

if __name__ == "__main__":
    main()