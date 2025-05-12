#!/usr/bin/env python
"""
Augment genuine EEG dataset with synthetic data by a specified percentage.
This script combines real data with synthetic data to create augmented datasets
of different sizes (e.g., +20%, +40%, etc.)
"""

import os
import argparse
import numpy as np
import shutil
from collections import defaultdict
from pathlib import Path
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Augment genuine EEG dataset with synthetic data")
    
    # Path to datasets
    parser.add_argument("--genuine_dataset", type=str, required=True,
                      help="Path to the genuine dataset directory")
    parser.add_argument("--synthetic_dataset", type=str, required=True,
                      help="Path to the synthetic dataset directory")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save the augmented datasets")
    
    # Augmentation percentages
    parser.add_argument("--percentages", type=str, default="20,40,60,100",
                      help="Comma-separated list of augmentation percentages (e.g., '20,40,60,100')")
    
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

def create_augmented_dataset(genuine_path, synthetic_path, output_path, percentage, seed=42):
    """
    Create an augmented dataset by combining genuine data with synthetic data.
    
    Args:
        genuine_path: Path to genuine dataset
        synthetic_path: Path to synthetic dataset
        output_path: Path to save augmented dataset
        percentage: Percentage of synthetic data to add (e.g., 20 for 20%)
        seed: Random seed for reproducibility
    """
    print(f"\n{'='*80}")
    print(f"Creating augmented dataset with +{percentage}% synthetic data")
    print(f"{'='*80}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directories
    output_feature_dir = os.path.join(output_path, "Feature")
    output_label_dir = os.path.join(output_path, "Label")
    os.makedirs(output_feature_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Read genuine and synthetic labels
    genuine_label_path = os.path.join(genuine_path, "Label", "label.npy")
    synthetic_label_path = os.path.join(synthetic_path, "Label", "label.npy")
    
    genuine_label_dict, genuine_labels = read_labels(genuine_label_path)
    synthetic_label_dict, synthetic_labels = read_labels(synthetic_label_path)
    
    # Calculate number of samples to add for each category
    samples_to_add = {}
    for label in [0, 1, 2]:  # HC, MCI, Dementia
        genuine_count = len(genuine_label_dict[label])
        add_count = int(genuine_count * percentage / 100)
        available_count = len(synthetic_label_dict[label])
        
        if add_count > available_count:
            print(f"Warning: Requested {add_count} samples for category {label}, but only {available_count} available")
            add_count = min(add_count, available_count)
        
        samples_to_add[label] = add_count
        print(f"Category {label}: Adding {add_count} synthetic samples to {genuine_count} genuine samples")
    
    # First, copy all genuine files to the output directory
    print("Copying genuine dataset files...")
    
    # Copy all genuine features
    genuine_feature_path = os.path.join(genuine_path, "Feature")
    for feature_file in os.listdir(genuine_feature_path):
        if feature_file.endswith(".npy"):
            src_path = os.path.join(genuine_feature_path, feature_file)
            dst_path = os.path.join(output_feature_dir, feature_file)
            shutil.copy2(src_path, dst_path)
    
    # Find the highest feature_id in the genuine dataset
    max_feature_id = 0
    for label in genuine_label_dict.values():
        if label:
            max_feature_id = max(max_feature_id, max(label))
    
    print(f"Highest feature_id in genuine dataset: {max_feature_id}")
    
    # Initialize new labels array with genuine labels
    augmented_labels = genuine_labels.tolist()
    
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
                
                # Add to augmented labels
                augmented_labels.append([label, new_id])
            
            # Update max_feature_id
            max_feature_id += len(selected_features)
    
    # Save augmented labels
    augmented_labels = np.array(augmented_labels)
    np.save(os.path.join(output_label_dir, "label.npy"), augmented_labels)
    
    # Print summary
    original_count = len(genuine_labels)
    added_count = len(augmented_labels) - original_count
    print(f"Augmented dataset created: {original_count} genuine + {added_count} synthetic = {len(augmented_labels)} total samples")
    print(f"Files saved to: {output_path}")

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
    
    print(f"Will create augmented datasets with the following percentages: {percentages}")
    
    # Create augmented datasets for each percentage
    for percentage in percentages:
        output_path = os.path.join(args.output_dir, f"augmented_{percentage}pct")
        create_augmented_dataset(
            args.genuine_dataset,
            args.synthetic_dataset,
            output_path,
            percentage,
            args.seed
        )
    
    print("\nAll augmented datasets created successfully!")

if __name__ == "__main__":
    main()