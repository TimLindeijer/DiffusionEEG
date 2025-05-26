"""
Utility functions for CAUEEG GREEN classifier
"""

import yaml
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    
    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_paths(config: Dict) -> Dict[str, str]:
    """
    Build full paths from configuration
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    paths : dict
        Dictionary with full paths
    """
    paths = config['paths']
    
    full_paths = {
        'data_path': paths['user_path'] + paths['data_subdir'],
        'bids_root': paths['user_path'] + paths['data_subdir'] + paths['bids_subdir'],
        'derivatives_path': paths['user_path'] + paths['data_subdir'] + 
                           paths['bids_subdir'] + paths['derivatives_subdir'],
        'participants_path': paths['user_path'] + paths['data_subdir'] + 
                            paths['bids_subdir'] + '/' + paths['participants_file']
    }
    
    return full_paths

def validate_data(epochs_list: List, labels: List, config: Dict) -> Tuple[List, List]:
    """
    Validate loaded data against configuration
    
    Parameters:
    -----------
    epochs_list : list
        List of MNE epochs
    labels : list
        List of labels
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    epochs_list : list
        Validated epochs list
    labels : list
        Validated labels
    """
    # Check if we have data
    if len(epochs_list) == 0:
        raise ValueError("No data loaded!")
    
    # Check channels
    expected_channels = config['eeg']['channels']
    for i, epochs in enumerate(epochs_list):
        if not set(expected_channels).issubset(set(epochs.ch_names)):
            print(f"Warning: Subject {i} missing expected channels")
            print(f"Expected: {expected_channels}")
            print(f"Found: {epochs.ch_names}")
    
    # Check labels
    unique_labels = set(labels)
    expected_labels = set(config['labels']['expected_classes'])
    
    if unique_labels != expected_labels:
        print(f"Warning: Label mismatch")
        print(f"Expected: {expected_labels}")
        print(f"Found: {unique_labels}")
    
    return epochs_list, labels

def get_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch dtype
    
    Parameters:
    -----------
    dtype_str : str
        String representation of dtype
    
    Returns:
    --------
    dtype : torch.dtype
        PyTorch dtype
    """
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
    }
    
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    
    return dtype_map[dtype_str]

def create_results_summary(cv_results: List, label_encoder, output_path: str):
    """
    Create a summary of cross-validation results
    
    Parameters:
    -----------
    cv_results : list
        Results from cross-validation
    label_encoder : LabelEncoder
        Fitted label encoder
    output_path : str
        Path to save summary
    """
    if not cv_results or cv_results[0] is None:
        print("No results to summarize")
        return
    
    # Extract scores
    scores = [r[0]['test_score'] for r in cv_results if r]
    
    # Create summary
    summary = {
        'n_folds': len(scores),
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores),
        'classes': list(label_encoder.classes_)
    }
    
    # Save as JSON
    import json
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Also create a text summary
    text_summary = f"""
CAUEEG GREEN Classifier Results
==============================

Cross-Validation Configuration:
- Number of folds: {summary['n_folds']}
- Classes: {', '.join(summary['classes'])}

Results:
- Mean Score: {summary['mean_score']:.4f} ± {summary['std_score']:.4f}
- Min Score: {summary['min_score']:.4f}
- Max Score: {summary['max_score']:.4f}

Per-Fold Scores:
"""
    
    for i, score in enumerate(scores):
        text_summary += f"- Fold {i}: {score:.4f}\n"
    
    # Save text summary
    text_path = output_path.replace('.json', '.txt')
    with open(text_path, 'w') as f:
        f.write(text_summary)
    
    print(f"Results saved to {output_path} and {text_path}")

def plot_training_curves(checkpoint_dir: str, output_dir: str):
    """
    Plot training curves from checkpoint directory
    
    Parameters:
    -----------
    checkpoint_dir : str
        Directory containing checkpoints
    output_dir : str
        Directory to save plots
    """
    import matplotlib.pyplot as plt
    import glob
    
    # Find all metrics files
    metrics_files = glob.glob(os.path.join(checkpoint_dir, '*/metrics.csv'))
    
    if not metrics_files:
        print("No metrics files found")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for metrics_file in metrics_files:
        fold_name = os.path.basename(os.path.dirname(metrics_file))
        
        # Load metrics
        metrics_df = pd.read_csv(metrics_file)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        if 'train_loss' in metrics_df.columns:
            ax1.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train')
        if 'val_loss' in metrics_df.columns:
            ax1.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Loss Curve - {fold_name}')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        if 'val_acc' in metrics_df.columns:
            ax2.plot(metrics_df['epoch'], metrics_df['val_acc'], label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Balanced Accuracy')
        ax2.set_title(f'Accuracy Curve - {fold_name}')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'training_curves_{fold_name}.png'))
        plt.close()
        
        print(f"Saved plot for {fold_name}")

# Example usage function
def run_with_config(config_path: str):
    """
    Example of how to use the configuration system
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Build paths
    paths = build_paths(config)
    
    # Print configuration summary
    print("Configuration loaded:")
    print(f"- Data path: {paths['data_path']}")
    print(f"- BIDS root: {paths['bids_root']}")
    print(f"- Model frequencies: {config['model']['n_freqs']}")
    print(f"- Hidden dimensions: {config['model']['hidden_dim']}")
    print(f"- Training epochs: {config['training']['n_epochs']}")
    print(f"- Cross-validation folds: {config['cross_validation']['n_splits']}")
    
    return config, paths

# Data augmentation utilities
def augment_epochs(epochs, augmentation_factor=2, noise_level=0.1):
    """
    Simple data augmentation for EEG epochs
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Original epochs
    augmentation_factor : int
        How many augmented versions to create
    noise_level : float
        Standard deviation of Gaussian noise to add
    
    Returns:
    --------
    augmented_epochs : mne.Epochs
        Augmented epochs
    """
    import mne
    
    data = epochs.get_data()
    augmented_data = []
    
    for _ in range(augmentation_factor):
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, data.shape)
        augmented = data + noise
        augmented_data.append(augmented)
    
    # Concatenate all data
    all_data = np.concatenate([data] + augmented_data, axis=0)
    
    # Create new epochs object
    augmented_epochs = mne.EpochsArray(
        all_data, 
        epochs.info, 
        tmin=epochs.times[0]
    )
    
    return augmented_epochs