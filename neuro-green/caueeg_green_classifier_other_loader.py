#!/usr/bin/env python
"""
Train GREEN model on CAUEEG dataset with command-line interface
Fixed version that handles participant ID mismatches
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import mne
from pathlib import Path
from datetime import datetime
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# Add the green module paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from green.data_utils import EpochsDataset
from green.wavelet_layers import RealCovariance
from green.research_code.pl_utils import get_green, GreenClassifierLM
from green.research_code.crossval_utils import pl_crossval

# Optional: Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available. Install with: pip install wandb")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train GREEN model on CAUEEG dataset')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CAUEEG BIDS directory')
    parser.add_argument('--output_dir', type=str, default='results/caueeg_classification',
                        help='Output directory for results')
    parser.add_argument('--derivatives_subdir', type=str, default='/derivatives/sovaharmony',
                        help='Subdirectory for preprocessed data')
    
    # Model architecture
    parser.add_argument('--n_freqs', type=int, default=10,
                        help='Number of frequency bands (NOT channels!)')
    parser.add_argument('--kernel_width_s', type=float, default=0.5,
                        help='Wavelet kernel width in seconds')
    parser.add_argument('--hidden_dim', type=int, nargs='+', default=[32, 16],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--bi_out', type=int, nargs='+', default=[10],
                        help='BiMap output dimensions')
    parser.add_argument('--logref', type=str, default='logeuclid',
                        choices=['identity', 'logeuclid'],
                        help='Reference for log mapping')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Dataset arguments
    parser.add_argument('--n_epochs_per_subject', type=int, default=20,
                        help='Number of epochs to use per subject')
    parser.add_argument('--sfreq', type=float, default=None,
                        help='Sampling frequency (auto-detected if not specified)')
    
    # Cross-validation arguments
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Evaluation arguments
    parser.add_argument('--detailed_evaluation', action='store_true',
                        help='Perform detailed evaluation and save predictions')
    
    # W&B arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='green-caueeg',
                        help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=['caueeg'],
                        help='W&B tags')
    
    return parser.parse_args()

def load_caueeg_data(data_dir, derivatives_subdir):
    """Load CAUEEG data and labels with flexible ID matching"""
    # Set paths
    bids_root = data_dir
    # Remove leading slash and use proper path joining
    derivatives_subdir_clean = derivatives_subdir.lstrip('/')
    derivatives_path = os.path.join(bids_root, derivatives_subdir_clean)
    participants_path = os.path.join(bids_root, 'participants.tsv')
    
    print(f"BIDS root: {bids_root}")
    print(f"Derivatives path: {derivatives_path}")
    print(f"Checking if derivatives exists: {os.path.exists(derivatives_path)}")
    print(f"Loading data from {derivatives_path}")
    
    # Load participants data
    participants_df = pd.read_csv(participants_path, sep='\t')
    participants_df = participants_df.dropna(subset=['ad_syndrome_3'])
    
    epochs_list = []
    labels = []
    subjects = []
    
    # Expected channels
    expected_channels = ['FP1', 'F3', 'C3', 'P3', 'O1', 'FP2', 'F4', 
                        'C4', 'P4', 'O2', 'F7', 'T3', 'T5', 'F8', 
                        'T4', 'T6', 'FZ', 'CZ', 'PZ']
    
    # Track loaded subjects
    found_count = 0
    
    for _, row in participants_df.iterrows():
        participant_id = row['participant_id']
        label = row['ad_syndrome_3']
        
        # Construct file paths - try both with and without square brackets
        eeg_file_pattern = os.path.join(
                derivatives_path,
                f'{participant_id}_task-eyesClosed_clean_140hz-epo.fif'
            )
        
        eeg_file = None
        if os.path.exists(eeg_file_pattern):
            eeg_file = eeg_file_pattern
        
        if eeg_file:
            try:
                epochs = mne.read_epochs(eeg_file, preload=True, verbose=False)
                
                # Pick channels if needed
                if set(expected_channels).issubset(set(epochs.ch_names)):
                    epochs = epochs.pick(expected_channels)
                
                epochs_list.append(epochs)
                labels.append(label)
                subjects.append(participant_id)
                found_count += 1
                
                print(f"✓ Loaded {participant_id}: {epochs.get_data().shape[0]} epochs, label: {label}")
                
            except Exception as e:
                print(f"✗ Error loading {participant_id}: {e}")
        else:
            print(f"✗ EEG file not found for {participant_id}")
    
    print(f"\nSummary: Loaded {found_count} subjects out of {len(participants_df)}")
    
    return epochs_list, labels, subjects

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize W&B if requested
    if args.use_wandb and WANDB_AVAILABLE:
        run_name = args.wandb_name or f"caueeg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=args.wandb_tags,
            config=vars(args)
        )
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set MNE log level
    mne.set_log_level('ERROR')
    
    # Load data
    print("=" * 80)
    print("Loading CAUEEG data...")
    print("=" * 80)
    
    epochs_list, labels, subjects = load_caueeg_data(args.data_dir, args.derivatives_subdir)
    
    if len(epochs_list) == 0:
        print("ERROR: No data loaded. Check file paths.")
        return
    
    print(f"\nTotal subjects loaded: {len(epochs_list)}")
    
    # Prepare labels
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    n_classes = len(label_encoder.classes_)
    
    # One-hot encode
    one_hot_labels = np.zeros((len(labels), n_classes))
    one_hot_labels[np.arange(len(labels)), numeric_labels] = 1
    encoded_labels = torch.Tensor(one_hot_labels).to(torch.float32)
    
    print(f"\nLabel classes: {label_encoder.classes_}")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Get data properties
    example_epochs = epochs_list[0]
    n_channels = len(example_epochs.ch_names)
    sfreq = args.sfreq or example_epochs.info['sfreq']
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = EpochsDataset(
        epochs=epochs_list,
        targets=encoded_labels,
        subjects=subjects,
        n_epochs=args.n_epochs_per_subject,
        padding='repeat',
        shuffle=True,
        random_state=args.seed
    )
    
    print(f"Dataset created: {len(dataset)} subjects")
    print(f"Epochs per subject: {args.n_epochs_per_subject}")
    print(f"Number of channels: {n_channels}")
    print(f"Sampling frequency: {sfreq} Hz")
    
    # Create model
    print("\nCreating GREEN model...")
    model = get_green(
        n_freqs=args.n_freqs,
        kernel_width_s=args.kernel_width_s,
        n_ch=n_channels,
        sfreq=sfreq,
        orth_weights=True,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        logref=args.logref,
        pool_layer=RealCovariance(),
        bi_out=args.bi_out,
        dtype=torch.float32,
        out_dim=n_classes
    )
    
    # Create PyTorch Lightning module
    model_pl = GreenClassifierLM(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    stratify_labels = np.argmax(encoded_labels.numpy(), axis=1)
    
    train_splits = []
    test_splits = []
    
    for train, test in cv.split(np.arange(len(dataset)), stratify_labels):
        train_splits.append(train)
        test_splits.append(test)
    
    print(f"\nStarting {args.n_splits}-fold cross-validation...")
    print("=" * 80)
    
    # Run cross-validation
    ckpt_prefix = os.path.join(args.output_dir, 'checkpoints')
    
    results, _ = pl_crossval(
        model,
        dataset=dataset,
        n_epochs=args.max_epochs,
        save_preds=args.detailed_evaluation,
        ckpt_prefix=ckpt_prefix,
        train_splits=train_splits,
        test_splits=test_splits,
        batch_size=args.batch_size,
        pl_module=GreenClassifierLM,
        num_workers=args.num_workers,
        pl_params={'weight_decay': args.weight_decay}
    )
    
    print("\nCross-validation complete!")
    print("=" * 80)
    
    # Calculate and save results
    if results and results[0] is not None:
        scores = [r[0]['test_score'] for r in results if r]
        
        results_summary = {
            'args': vars(args),
            'n_subjects': len(dataset),
            'n_folds': len(scores),
            'scores': scores,
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'classes': list(label_encoder.classes_),
            'label_distribution': dict(zip(*np.unique(labels, return_counts=True)))
        }
        
        # Save results
        results_file = os.path.join(args.output_dir, 'results_summary.json')
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=4)
        
        # Save label encoder
        encoder_file = os.path.join(args.output_dir, 'label_encoder.pkl')
        with open(encoder_file, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Print results
        print("\nResults Summary:")
        print(f"Mean Score: {results_summary['mean_score']:.4f} ± {results_summary['std_score']:.4f}")
        print(f"Min Score: {results_summary['min_score']:.4f}")
        print(f"Max Score: {results_summary['max_score']:.4f}")
        
        for i, score in enumerate(scores):
            print(f"Fold {i}: {score:.4f}")
        
        # Log to W&B
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'mean_score': results_summary['mean_score'],
                'std_score': results_summary['std_score'],
                'scores': scores
            })
            wandb.finish()
        
        print(f"\nResults saved to {args.output_dir}")
    else:
        print("ERROR: No results obtained from cross-validation")

if __name__ == "__main__":
    main()