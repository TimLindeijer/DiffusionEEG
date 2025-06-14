#!/usr/bin/env python
"""
Modified version to handle augmented datasets that only contain augmented samples
"""

import os
import argparse
import numpy as np
import pandas as pd
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
                      help="Path to the synthetic/augmented dataset directory")
    
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
    
    # Dataset type
    parser.add_argument("--dataset_type", type=str, default="auto", 
                      choices=["auto", "synthetic", "augmented"],
                      help="Type of secondary dataset: synthetic (includes originals) or augmented (only augmented)")
    
    # Generation parameters
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    return parser.parse_args()

def detect_dataset_type(dataset_path):
    """
    Detect if the dataset is synthetic (includes originals) or augmented (only augmented)
    """
    subject_mapping_path = os.path.join(dataset_path, "Label", "subject_mapping.csv")
    
    if os.path.exists(subject_mapping_path):
        try:
            df = pd.read_csv(subject_mapping_path)
            
            # Check if there are entries marked as "Not present"
            if 'file_in_output' in df.columns:
                not_present_count = df['file_in_output'].str.contains('Not present', na=False).sum()
                total_count = len(df)
                
                if not_present_count > total_count * 0.5:  # More than 50% not present
                    print(f"Detected AUGMENTED dataset (only augmented samples present)")
                    return "augmented"
                else:
                    print(f"Detected SYNTHETIC dataset (includes original samples)")
                    return "synthetic"
            
        except Exception as e:
            print(f"Warning: Could not read subject mapping: {e}")
    
    # Fallback: assume synthetic
    print("Could not detect dataset type, assuming SYNTHETIC")
    return "synthetic"

def read_labels_augmented(label_path, subject_mapping_path):
    """
    Read labels for augmented dataset, filtering out original references
    """
    print(f"Reading augmented labels from: {label_path}")
    
    # Load the labels
    labels = np.load(label_path)
    print(f"Loaded label file with shape: {labels.shape}")
    
    # Load subject mapping to filter out original references
    if os.path.exists(subject_mapping_path):
        df = pd.read_csv(subject_mapping_path)
        print(f"Loaded subject mapping with {len(df)} entries")
        
        # Filter to only include entries that are present in the output
        present_entries = df[~df['file_in_output'].str.contains('Not present', na=False)]
        present_subject_ids = set(present_entries['subject_id'].values)
        
        print(f"Found {len(present_subject_ids)} subjects with actual files (excluding original references)")
        
        # Filter labels to only include present subjects
        filtered_labels = []
        for entry in labels:
            subject_id = int(entry[1])
            if subject_id in present_subject_ids:
                filtered_labels.append(entry)
        
        filtered_labels = np.array(filtered_labels)
        print(f"Filtered labels shape: {filtered_labels.shape}")
        
    else:
        print("Warning: No subject mapping found, using all labels")
        filtered_labels = labels
    
    # Group by label
    label_to_features = defaultdict(list)
    for entry in filtered_labels:
        label = int(entry[0])  # First column is label
        subject_id = int(entry[1])  # Second column is subject_id
        label_to_features[label].append(subject_id)
    
    # Print summary
    print(f"Augmented dataset summary:")
    for label, features in label_to_features.items():
        print(f"  Class {label}: {len(features)} augmented samples")
    
    return label_to_features, filtered_labels

def read_labels_synthetic(label_path):
    """
    Original read_labels function for synthetic datasets
    """
    print(f"Reading synthetic labels from: {label_path}")
    
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
    print(f"Synthetic dataset summary:")
    for label, features in label_to_features.items():
        print(f"  Class {label}: {len(features)} samples")
    
    return label_to_features, labels

def read_labels(label_path, dataset_type="synthetic", dataset_path=None):
    """
    Read labels based on dataset type
    """
    if dataset_type == "augmented":
        subject_mapping_path = os.path.join(dataset_path, "Label", "subject_mapping.csv")
        return read_labels_augmented(label_path, subject_mapping_path)
    else:
        return read_labels_synthetic(label_path)

def copy_features(feature_ids, src_dir, dst_dir, prefix="feature_"):
    """Copy feature files with the given IDs from source to destination."""
    os.makedirs(dst_dir, exist_ok=True)
    
    copied_count = 0
    missing_files = []
    
    for feature_id in feature_ids:
        src_file = os.path.join(src_dir, f"{prefix}{feature_id:02d}.npy")
        dst_file = os.path.join(dst_dir, f"{prefix}{feature_id:02d}.npy")
        
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            copied_count += 1
        else:
            missing_files.append(src_file)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} source files not found")
        if len(missing_files) <= 5:
            for f in missing_files:
                print(f"  Missing: {f}")
    
    return copied_count

def create_train_test_split(genuine_path, output_dir, test_size=0.2, stratify=True, seed=42):
    """
    Split genuine dataset into train and test sets.
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
    genuine_label_dict, genuine_labels = read_labels(genuine_label_path, "synthetic")
    
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
                        balance_to="max", dataset_type="synthetic", seed=42):
    """
    Balance the training set with synthetic/augmented data.
    """
    print(f"\n{'='*80}")
    print(f"Balancing training set with {dataset_type} data (strategy: {balance_to})")
    print(f"{'='*80}")
    
    # Set random seed
    random.seed(seed)
    
    # Create output directories
    output_feature_dir = os.path.join(output_dir, "Feature")
    output_label_dir = os.path.join(output_dir, "Label")
    os.makedirs(output_feature_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Read synthetic/augmented labels
    synthetic_label_path = os.path.join(synthetic_path, "Label", "label.npy")
    synthetic_label_dict, synthetic_labels = read_labels(
        synthetic_label_path, dataset_type, synthetic_path
    )
    
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
        available_count = len(synthetic_label_dict.get(label, []))
        
        if add_count > available_count:
            print(f"Warning: Need {add_count} samples for category {label}, but only {available_count} available")
            add_count = min(add_count, available_count)
        
        samples_to_add[label] = max(0, add_count)  # Ensure non-negative
        print(f"Category {label}: Adding {samples_to_add[label]} {dataset_type} samples to {genuine_count} genuine samples")
    
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
    
    # Add synthetic/augmented samples for each category
    for label in [0, 1, 2]:
        # Get available synthetic/augmented feature IDs for this category
        available_features = synthetic_label_dict.get(label, []).copy()
        
        # Randomly select the required number of samples
        num_to_add = samples_to_add[label]
        if num_to_add > 0 and available_features:
            selected_features = random.sample(available_features, num_to_add)
            
            print(f"Adding {len(selected_features)} {dataset_type} samples for category {label}")
            
            # Copy selected synthetic/augmented features with new IDs
            next_id = max_feature_id + 1
            for i, feature_id in enumerate(selected_features):
                # Source synthetic/augmented file
                src_filename = f"feature_{feature_id:02d}.npy"
                src_path = os.path.join(synthetic_path, "Feature", src_filename)
                
                # Destination with new ID
                new_id = next_id + i
                dst_filename = f"feature_{new_id:02d}.npy"
                dst_path = os.path.join(output_feature_dir, dst_filename)
                
                # Copy file with new ID
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    # Add to balanced labels
                    balanced_labels.append([label, new_id])
                else:
                    print(f"Warning: {dataset_type} file not found: {src_path}")
            
            # Update max_feature_id
            max_feature_id += len(selected_features)
    
    # Save balanced labels
    balanced_labels = np.array(balanced_labels)
    np.save(os.path.join(output_label_dir, "label.npy"), balanced_labels)
    
    # Print summary
    original_count = len(genuine_train_ids)
    added_count = len(balanced_labels) - original_count
    print(f"Balanced training set: {original_count} genuine + {added_count} {dataset_type} = {len(balanced_labels)} total samples")
    
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
    
    # Auto-detect dataset type if needed
    if args.dataset_type == "auto":
        dataset_type = detect_dataset_type(args.synthetic_dataset)
    else:
        dataset_type = args.dataset_type
    
    print(f"Using dataset type: {dataset_type}")
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
        dataset_type,
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