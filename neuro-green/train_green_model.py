#!/usr/bin/env python
"""
Training script for GREEN model on CAUEEG2 dataset.
Designed to be used with SLURM job submission.
"""

import os
import time
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import mne
import torch
from torch.utils.data import DataLoader, random_split
from green.data_utils import EpochsDataset
from green.wavelet_layers import RealCovariance
from green.research_code.pl_utils import get_green, GreenClassifierLM
from green.research_code.crossval_utils import pl_crossval
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Set MNE log level
mne.set_log_level('ERROR')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GREEN model on CAUEEG2 dataset')
    
    # Data and output directories
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CAUEEG2 dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping')
    
    # Model parameters
    parser.add_argument('--n_freqs', type=int, default=8,
                        help='Number of wavelet filters')
    parser.add_argument('--kernel_width_s', type=float, default=0.5,
                        help='Kernel width in seconds')
    parser.add_argument('--hidden_dim', type=int, nargs='+', default=[64, 32],
                        help='Hidden dimensions (multiple values for multiple layers)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--sfreq', type=float, default=100.0,
                        help='Sampling frequency of the data')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--detailed_evaluation', action='store_true',
                        help='Perform detailed evaluation and generate plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_caueeg_data(feature_path, label_path, ch_names=None):
    """
    Load CAUEEG2 dataset and convert to mne.Epochs
    """
    # Define default channel names if not provided
    if ch_names is None:
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'O1', 'FP2', 'F4', 'C4', 'P4', 'O2', 
                   'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'FZ', 'CZ', 'PZ']
    
    # Load labels
    labels_array = np.load(os.path.join(label_path, 'label.npy'))
    
    # Get mapping between subject_id and label
    subject_labels = {int(entry[1]): int(entry[0]) for entry in labels_array}
    
    # Get feature files
    feature_files = [f for f in os.listdir(feature_path) if f.startswith('feature_') and f.endswith('.npy')]
    feature_files.sort()
    
    epochs_list = []
    labels_list = []
    subjects_list = []
    
    print(f"Found {len(feature_files)} feature files")
    
    # Process each feature file
    for feature_file in feature_files:
        # Extract subject ID
        subject_id = int(feature_file.split('_')[1].split('.')[0])
        
        # Skip if no label for this subject
        if subject_id not in subject_labels:
            continue
        
        # Load feature data
        feature_data = np.load(os.path.join(feature_path, feature_file))
        
        # Feature data is (epochs, times, channels)
        # MNE expects (epochs, channels, times)
        data = np.transpose(feature_data, (0, 2, 1))
        
        # Create info object - make sure to only use available channels
        n_channels = data.shape[1]
        used_ch_names = ch_names[:n_channels] if n_channels <= len(ch_names) else [f"ch{i}" for i in range(n_channels)]
        
        info = mne.create_info(ch_names=used_ch_names, sfreq=100.0, ch_types='eeg')
        
        # Create epochs object
        epochs = mne.EpochsArray(data, info)
        
        # Get label
        label = subject_labels[subject_id]
        
        epochs_list.append(epochs)
        labels_list.append(label)
        subjects_list.append(f"subject_{subject_id}")
    
    print(f"Loaded {len(epochs_list)} subjects with labels")
    
    # Print class distribution
    classes, counts = np.unique(labels_list, return_counts=True)
    print("Class distribution:")
    for c, count in zip(classes, counts):
        print(f"  Class {c}: {count} subjects")
    
    return epochs_list, labels_list, subjects_list

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """
    Plot confusion matrix and save to output directory.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def evaluate_prediction_results(all_preds, output_dir):
    """
    Evaluate prediction results and save detailed reports.
    """
    # Extract true and predicted labels
    y_true = all_preds['y_true'].values
    y_pred = all_preds['y_pred'].values
    
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, output_dir)
    
    # Count number of examples per class
    class_counts = pd.Series(y_true).value_counts().sort_index()
    class_counts.to_csv(os.path.join(output_dir, 'class_counts.csv'))
    
    # Print summary
    print("\nClassification Report:")
    print(pd.DataFrame(report).transpose())
    
    # Return the overall accuracy
    return report['accuracy']

def main():
    """Main function to run the training and evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set paths
    feature_path = os.path.join(args.data_dir, 'Feature')
    label_path = os.path.join(args.data_dir, 'Label')
    
    # Check that paths exist
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature directory not found: {feature_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label directory not found: {label_path}")
    
    print(f"Loading data from {args.data_dir}")
    
    # Load data
    epochs_list, labels_list, subjects_list = load_caueeg_data(feature_path, label_path)
    
    # Convert labels to one-hot encoding
    targets = [torch.tensor([1 if i == label else 0 for i in range(3)], dtype=torch.float32) 
              for label in labels_list]
    
    # Create dataset
    dataset = EpochsDataset(
        epochs=epochs_list,
        targets=targets,
        subjects=subjects_list,
        n_epochs=10,  # Use 10 epochs per subject
        padding='repeat'
    )
    
    # Number of channels from the first epochs object
    n_ch = len(epochs_list[0].ch_names)
    print(f"Using {n_ch} EEG channels")
    
    # Create model for 3-class classification
    model = get_green(
        n_freqs=args.n_freqs,
        kernel_width_s=args.kernel_width_s,
        n_ch=n_ch,
        sfreq=args.sfreq,
        orth_weights=True,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        logref='logeuclid',
        pool_layer=RealCovariance(),
        bi_out=[args.hidden_dim[0] // 4],  # Use a fraction of the first hidden dim
        dtype=torch.float32,
        out_dim=3  # 3 classes: HC, MCI, Dementia
    )
    
    # Print model structure
    print(f"Model structure:")
    print(model)
    
    # Configure callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_acc",
            patience=args.patience,
            mode="max",
            verbose=True
        ),
    ]
    
    # Setup cross-validation
    # Split data into train/test (80/20)
    n_subjects = len(epochs_list)
    n_train = int(0.8 * n_subjects)
    
    indices = list(range(n_subjects))
    np.random.shuffle(indices)
    
    train_indices = [indices[:n_train]]
    test_indices = [indices[n_train:]]
    
    # Print split information
    print(f"Training on {len(train_indices[0])} subjects, testing on {len(test_indices[0])} subjects")
    
    # Start timing
    start_time = time.time()
    
    # Train model
    pl_crossval_output, _ = pl_crossval(
        model, 
        dataset=dataset,
        n_epochs=args.max_epochs,
        save_preds=True,
        ckpt_prefix=os.path.join(args.output_dir, 'checkpoints'),
        train_splits=train_indices,
        test_splits=test_indices,
        batch_size=args.batch_size,
        pl_module=GreenClassifierLM,
        num_workers=args.num_workers,
        callbacks=callbacks,
        pl_params={
            'weight_decay': args.weight_decay,
            'lr': args.learning_rate
        }
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Save training time
    with open(os.path.join(args.output_dir, 'training_time.txt'), 'w') as f:
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Training time: {training_time/60:.2f} minutes\n")
        f.write(f"Training time: {training_time/3600:.2f} hours\n")
    
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    # Perform detailed evaluation if requested
    if args.detailed_evaluation:
        print("Performing detailed evaluation...")
        
        # Load predictions
        preds_path = os.path.join(args.output_dir, 'checkpoints/fold0/preds.pkl')
        if os.path.exists(preds_path):
            all_preds = pd.read_pickle(preds_path)
            
            # Evaluate predictions
            accuracy = evaluate_prediction_results(all_preds, args.output_dir)
            
            # Save accuracy
            with open(os.path.join(args.output_dir, 'accuracy.txt'), 'w') as f:
                f.write(f"Accuracy: {accuracy:.4f}\n")
        else:
            print(f"Warning: Could not find predictions file at {preds_path}")
    
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main()