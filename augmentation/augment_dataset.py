#!/usr/bin/env python3
"""
FTSurrogate Dataset Augmentation Script
======================================

This script augments EEG datasets using FTSurrogate augmentation.
Designed to run on SLURM clusters with configurable paths.

Usage:
    python augment_dataset.py --input_dataset /path/to/input --output_dataset /path/to/output

Author: Based on "Data augmentation for learning predictive models on EEG" by Rommel et al., 2022
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
import numpy as np
import glob
import re
from typing import Dict, Tuple, List, Optional
import traceback

def setup_logging(log_file: str) -> logging.Logger:
    """Setup logging for the augmentation process."""
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def ft_surrogate_augmentation(
    data: np.ndarray,
    delta_phi_max: float = 0.9 * np.pi,
    independent_channels: bool = True,
    probability: float = 1.0
) -> np.ndarray:
    """
    Apply Fourier Transform Surrogate augmentation to EEG data.
    
    Parameters:
    -----------
    data : np.ndarray
        Input EEG data with shape (n_epochs, n_times, n_channels)
    delta_phi_max : float, default=0.9*π
        Maximum phase perturbation in radians
    independent_channels : bool, default=True
        If True, apply independent phase perturbations to each channel
    probability : float, default=1.0
        Probability of applying augmentation to each epoch
        
    Returns:
    --------
    np.ndarray
        Augmented data with same shape as input
    """
    
    if data.ndim != 3:
        raise ValueError(f"Data must be 3D (epochs, times, channels), got shape {data.shape}")
    
    n_epochs, n_times, n_channels = data.shape
    augmented_data = data.copy()
    
    for epoch_idx in range(n_epochs):
        # Apply augmentation with given probability
        if np.random.random() > probability:
            continue
            
        epoch_data = data[epoch_idx]  # Shape: (n_times, n_channels)
        
        # Apply FFT to each channel
        fft_data = np.fft.fft(epoch_data, axis=0)  # FFT along time axis
        
        # Generate phase perturbations
        if independent_channels:
            # Independent phase perturbations for each channel and frequency
            n_freqs = fft_data.shape[0]
            delta_phi = np.random.uniform(
                0, delta_phi_max, 
                size=(n_freqs, n_channels)
            )
        else:
            # Same phase perturbation for all channels (preserves cross-channel correlations)
            n_freqs = fft_data.shape[0]
            delta_phi = np.random.uniform(
                0, delta_phi_max, 
                size=(n_freqs, 1)
            )
            delta_phi = np.broadcast_to(delta_phi, (n_freqs, n_channels))
        
        # Apply phase perturbations
        # F[FTSurrogate(X)](f) = F[X](f) * e^(i*Δφ)
        phase_multiplier = np.exp(1j * delta_phi)
        fft_augmented = fft_data * phase_multiplier
        
        # Apply inverse FFT to get augmented signal
        augmented_epoch = np.fft.ifft(fft_augmented, axis=0).real
        
        augmented_data[epoch_idx] = augmented_epoch
    
    return augmented_data

def augment_complete_dataset(
    input_dataset_root: str,
    output_dataset_root: str,
    delta_phi_max: float = 0.9 * np.pi,
    independent_channels: bool = True,
    augmentation_suffix: str = "_ftsurrogate",
    logger: logging.Logger = None
) -> None:
    """
    Augment the complete dataset including features and labels for all patients.
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Starting dataset augmentation")
    logger.info(f"Input dataset: {input_dataset_root}")
    logger.info(f"Output dataset: {output_dataset_root}")
    logger.info(f"Delta phi max: {delta_phi_max}")
    logger.info(f"Independent channels: {independent_channels}")
    
    # Define paths
    feature_dir = os.path.join(input_dataset_root, 'Feature')
    label_dir = os.path.join(input_dataset_root, 'Label')
    
    output_feature_dir = os.path.join(output_dataset_root, 'Feature')
    output_label_dir = os.path.join(output_dataset_root, 'Label')
    
    # Verify input directories exist
    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    
    # Create output directories
    os.makedirs(output_feature_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Load original labels
    label_file = os.path.join(label_dir, 'label.npy')
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")
    
    original_labels = np.load(label_file)  # Shape: (n_subjects, 2) [label, subject_id]
    logger.info(f"Loaded original labels: {original_labels.shape}")
    logger.info(f"Original labels preview:\n{original_labels[:5]}")
    
    # Find all feature files
    feature_files = glob.glob(os.path.join(feature_dir, 'feature_*.npy'))
    feature_files.sort()  # Ensure consistent ordering
    
    logger.info(f"Found {len(feature_files)} feature files")
    
    # Parse subject IDs from filenames
    subject_ids = []
    for file_path in feature_files:
        filename = os.path.basename(file_path)
        # Extract subject ID from filename (e.g., feature_01.npy -> 1)
        match = re.search(r'feature_(\d+)', filename)
        if match:
            subject_ids.append(int(match.group(1)))
        else:
            logger.warning(f"Could not parse subject ID from {filename}")
    
    logger.info(f"Parsed subject IDs: {subject_ids}")
    
    # Prepare augmented labels
    augmented_labels = []
    max_original_subject_id = np.max(original_labels[:, 1])
    next_subject_id = max_original_subject_id + 1
    
    # Process each subject
    processed_subjects = 0
    failed_subjects = 0
    
    for file_path, subject_id in zip(feature_files, subject_ids):
        filename = os.path.basename(file_path)
        logger.info(f"Processing subject {subject_id}: {filename}")
        
        try:
            # Load original feature data
            original_data = np.load(file_path)
            logger.info(f"  Loaded data shape: {original_data.shape}")
            
            # Find corresponding label
            label_row = original_labels[original_labels[:, 1] == subject_id]
            if len(label_row) == 0:
                logger.warning(f"  No label found for subject {subject_id}")
                failed_subjects += 1
                continue
            elif len(label_row) > 1:
                logger.warning(f"  Multiple labels found for subject {subject_id}, using first")
            
            original_label = label_row[0, 0]  # Get the actual label (0, 1, or 2)
            logger.info(f"  Original label: {original_label}")
            
            # Apply FTSurrogate augmentation
            augmented_data = ft_surrogate_augmentation(
                original_data,
                delta_phi_max=delta_phi_max,
                independent_channels=independent_channels,
                probability=1.0
            )
            logger.info(f"  Augmented data shape: {augmented_data.shape}")
            
            # Save only augmented data with original filename
            augmented_output_file = os.path.join(output_feature_dir, filename)
            np.save(augmented_output_file, augmented_data)
            logger.info(f"  Saved augmented data as: {augmented_output_file}")
            
            # Add augmented label entry
            augmented_labels.append([original_label, next_subject_id])
            logger.info(f"  Augmented subject ID: {next_subject_id}")
            
            next_subject_id += 1
            processed_subjects += 1
            
        except Exception as e:
            logger.error(f"  Error processing subject {subject_id}: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            failed_subjects += 1
            continue
    
    # Combine original and augmented labels
    augmented_labels_array = np.array(augmented_labels, dtype=np.int32)
    combined_labels = np.vstack([original_labels, augmented_labels_array])
    
    logger.info(f"Label Summary:")
    logger.info(f"Original subjects: {len(original_labels)}")
    logger.info(f"Augmented subjects: {len(augmented_labels_array)}")
    logger.info(f"Total subjects: {len(combined_labels)}")
    logger.info(f"Combined labels shape: {combined_labels.shape}")
    logger.info(f"Successfully processed: {processed_subjects}")
    logger.info(f"Failed subjects: {failed_subjects}")
    
    # Show label distribution
    unique_labels, counts = np.unique(combined_labels[:, 0], return_counts=True)
    logger.info(f"Label distribution:")
    for label, count in zip(unique_labels, counts):
        logger.info(f"  Label {label}: {count} subjects")
    
    # Save combined labels
    output_label_file = os.path.join(output_label_dir, 'label.npy')
    np.save(output_label_file, combined_labels)
    logger.info(f"Saved combined labels to: {output_label_file}")
    
    # Create a mapping file for reference
    create_subject_mapping_file(
        original_labels, 
        augmented_labels_array, 
        output_label_dir, 
        augmentation_suffix,
        logger
    )
    
    logger.info(f"Dataset augmentation completed!")

def create_subject_mapping_file(
    original_labels: np.ndarray,
    augmented_labels: np.ndarray,
    output_dir: str,
    augmentation_suffix: str,
    logger: logging.Logger = None
) -> None:
    """
    Create a CSV file mapping original subjects to their augmented counterparts.
    Note: The output directory contains only augmented files with original filenames.
    """
    try:
        import pandas as pd
        
        if logger is None:
            logger = logging.getLogger(__name__)
        
        # Create mapping dataframe
        mapping_data = []
        
        # Add original subjects (these are references, actual files in output are augmented)
        for label, subject_id in original_labels:
            mapping_data.append({
                'subject_id': subject_id,
                'label': label,
                'type': 'original_reference',
                'original_subject_id': subject_id,
                'augmentation': 'none',
                'file_in_output': f'Not present (original data)',
                'note': 'Original data not saved in output directory'
            })
        
        # Add augmented subjects (these are the actual files in the output directory)
        for i, (label, aug_subject_id) in enumerate(augmented_labels):
            original_subject_id = original_labels[i, 1]  # Corresponding original subject
            mapping_data.append({
                'subject_id': aug_subject_id,
                'label': label,
                'type': 'augmented',
                'original_subject_id': original_subject_id,
                'augmentation': augmentation_suffix.replace('_', ''),
                'file_in_output': f'feature_{original_subject_id:02d}.npy',
                'note': 'Augmented data saved with original filename'
            })
        
        mapping_df = pd.DataFrame(mapping_data)
        
        # Save mapping file
        mapping_file = os.path.join(output_dir, 'subject_mapping.csv')
        mapping_df.to_csv(mapping_file, index=False)
        logger.info(f"Created subject mapping file: {mapping_file}")
        
        # Print summary
        logger.info(f"Subject mapping summary:")
        summary = mapping_df.groupby(['type', 'label']).size().unstack(fill_value=0)
        logger.info(f"\n{summary}")
        
        # Create additional info file explaining the structure
        info_file = os.path.join(output_dir, 'dataset_structure_info.txt')
        with open(info_file, 'w') as f:
            f.write("Dataset Structure Information\n")
            f.write("============================\n\n")
            f.write("This dataset contains ONLY AUGMENTED data.\n")
            f.write("The Feature/ directory contains augmented versions of the original data,\n")
            f.write("but saved with the original filenames (e.g., feature_01.npy).\n\n")
            f.write("The label.npy file contains BOTH original and augmented subject entries:\n")
            f.write("- Original subjects (1-N): Reference labels, no corresponding files in Feature/\n")
            f.write(f"- Augmented subjects ({np.max(original_labels[:, 1])+1}-{np.max(augmented_labels[:, 1])}): Actual files in Feature/\n\n")
            f.write("To use this dataset:\n")
            f.write("1. Load the feature files from Feature/ directory (these are augmented)\n")
            f.write("2. Use the augmented subject labels from label.npy\n")
            f.write("3. See subject_mapping.csv for detailed mapping information\n")
        
        logger.info(f"Created dataset info file: {info_file}")
        
    except ImportError:
        logger.warning("pandas not available, skipping mapping file creation")
    except Exception as e:
        logger.error(f"Error creating mapping file: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Augment EEG dataset using FTSurrogate augmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--input_dataset',
        type=str,
        required=True,
        help='Path to input dataset root directory (containing Feature/ and Label/ subdirs)'
    )
    
    parser.add_argument(
        '--output_dataset',
        type=str,
        required=True,
        help='Path to output dataset root directory'
    )
    
    # Optional arguments
    parser.add_argument(
        '--delta_phi_max',
        type=float,
        default=0.9 * np.pi,
        help='Maximum phase perturbation in radians (best value: 0.9*pi)'
    )
    
    parser.add_argument(
        '--independent_channels',
        action='store_true',
        default=True,
        help='Apply independent phase perturbations to each channel (use --no-independent_channels for BCI data)'
    )
    
    parser.add_argument(
        '--no-independent_channels',
        action='store_false',
        dest='independent_channels',
        help='Apply same phase perturbation to all channels (for BCI data to preserve cross-channel correlations)'
    )
    
    parser.add_argument(
        '--augmentation_suffix',
        type=str,
        default='_ftsurrogate',
        help='Suffix to add to augmented feature files'
    )
    
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Path to log file (if not specified, creates one in output directory)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Setup logging
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = os.path.join(args.output_dataset, f'augmentation_{timestamp}.log')
    
    logger = setup_logging(args.log_file)
    
    # Log job information
    logger.info("=" * 60)
    logger.info("FTSurrogate Dataset Augmentation")
    logger.info("=" * 60)
    logger.info(f"Job started at: {datetime.now()}")
    logger.info(f"Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 60)
    
    try:
        # Record start time
        start_time = time.time()
        
        # Run augmentation
        augment_complete_dataset(
            input_dataset_root=args.input_dataset,
            output_dataset_root=args.output_dataset,
            delta_phi_max=args.delta_phi_max,
            independent_channels=args.independent_channels,
            augmentation_suffix=args.augmentation_suffix,
            logger=logger
        )
        
        # Record completion
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info(f"Job completed successfully at: {datetime.now()}")
        logger.info(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Job failed with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()