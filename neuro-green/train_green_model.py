#!/usr/bin/env python
"""
Training script for GREEN model on CAUEEG2 dataset with TRUE subject-based cross-validation.
This script ensures that all epochs from the same subject are either fully in the training set
or fully in the testing set, preventing data leakage.
"""

import os
import time
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import seaborn as sns

import mne
import torch
from torch.utils.data import DataLoader, random_split, Subset
from green.data_utils import EpochsDataset
from green.wavelet_layers import RealCovariance
from green.research_code.pl_utils import get_green, GreenClassifierLM
from green.research_code.crossval_utils import pl_crossval
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Import wandb
import wandb
from lightning.pytorch.callbacks import Callback

# Set MNE log level
mne.set_log_level('ERROR')

# Define class name mapping
CLASS_NAMES = {
    0: "HC (+SMC)",
    1: "MCI",
    2: "Dementia"
}

class WandbCallback(Callback):
    """Custom callback for wandb logging"""
    def __init__(self):
        super().__init__()
        self.best_val_acc = 0
        self.best_epoch = 0
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Log metrics from the trainer's callback_metrics
        metrics = trainer.callback_metrics
        
        # Only log if wandb is initialized
        if wandb.run is not None:
            # Log validation metrics
            epoch_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                epoch_metrics[key] = value
            
            # Add epoch number
            epoch_metrics['epoch'] = trainer.current_epoch
            
            # Track best validation accuracy
            val_acc = metrics.get('val_acc', 0)
            if isinstance(val_acc, torch.Tensor):
                val_acc = val_acc.item()
                
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = trainer.current_epoch
                epoch_metrics['best_val_acc'] = self.best_val_acc
                epoch_metrics['best_epoch'] = self.best_epoch
            
            # Log to wandb
            wandb.log(epoch_metrics)
            
    def on_test_epoch_end(self, trainer, pl_module):
        # Log test metrics
        metrics = trainer.callback_metrics
        
        # Only log if wandb is initialized
        if wandb.run is not None:
            test_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                test_metrics[f"test_{key}"] = value
            
            # Log to wandb
            wandb.log(test_metrics)

# Custom classifier that logs to wandb
class WandbGreenClassifierLM(GreenClassifierLM):
    """Extension of GreenClassifierLM that logs to wandb"""
    def training_step(self, batch, batch_idx):
        # Call parent method to get loss
        loss = super().training_step(batch, batch_idx)
        
        # Log to wandb
        if wandb.run is not None and batch_idx % 10 == 0:  # Log every 10 batches to avoid flooding
            wandb.log({"train_batch_loss": loss.item()})
        
        return loss

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GREEN model on CAUEEG2 dataset with true subject-based cross-validation')
    
    # Data and output directories
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training dataset directory')
    parser.add_argument('--test_data_dir', type=str, default=None,
                        help='Path to testing dataset directory (if different from training)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--use_separate_test', action='store_true',
                        help='Use separate dataset for testing')
    
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
    parser.add_argument('--sfreq', type=float, default=200.0,
                        help='Sampling frequency of the data')
    
    # Cross-validation parameters
    parser.add_argument('--test_fraction', type=float, default=0.2,
                        help='Fraction of subjects to use for testing')
    parser.add_argument('--n_folds', type=int, default=1,
                        help='Number of cross-validation folds (use 1 for simple train/test split)')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--detailed_evaluation', action='store_true',
                        help='Perform detailed evaluation and generate plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # W&B parameters
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='green-caueeg',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity name (username or team name)')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='W&B run name (defaults to timestamp if not specified)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[],
                        help='Tags for the W&B run')
    
    # Debug option
    parser.add_argument('--debug', action='store_true',
                        help='Enable verbose debug output')
    
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

def create_true_subject_cross_validation(all_subjects_list, n_folds=5, test_fraction=0.2, seed=42):
    """
    Create cross-validation splits where all epochs from the same subject are either in 
    training or testing, never split between both.
    
    Parameters
    ----------
    all_subjects_list : list
        List of subject identifiers, can contain duplicates (one per epoch)
    n_folds : int, optional
        Number of cross-validation folds, by default 5. If 1, creates a single 
        train/test split using test_fraction.
    test_fraction : float, optional
        Fraction of subjects to use for testing when n_folds=1, by default 0.2
    seed : int, optional
        Random seed for reproducibility, by default 42
    
    Returns
    -------
    train_splits : list of lists
        List containing training indices for each fold
    test_splits : list of lists
        List containing testing indices for each fold
    """
    # Set random seed
    np.random.seed(seed)
    
    # Get unique subjects
    unique_subjects = list(set(all_subjects_list))
    n_subjects = len(unique_subjects)
    
    # Create mapping from subject ID to indices in the original list
    subject_to_indices = {subject: [] for subject in unique_subjects}
    for i, subject in enumerate(all_subjects_list):
        subject_to_indices[subject].append(i)
    
    if n_folds == 1:
        # Simple train/test split
        n_test = int(n_subjects * test_fraction)
        
        # Randomly select test subjects
        np.random.shuffle(unique_subjects)
        test_subjects = set(unique_subjects[:n_test])
        train_subjects = set(unique_subjects[n_test:])
        
        # Get indices for train and test subjects
        train_indices = []
        for subject in train_subjects:
            train_indices.extend(subject_to_indices[subject])
        
        test_indices = []
        for subject in test_subjects:
            test_indices.extend(subject_to_indices[subject])
        
        # Return as lists of lists for consistency with the cross-validation format
        return [train_indices], [test_indices]
    else:
        # Multiple folds for cross-validation
        np.random.shuffle(unique_subjects)
        fold_size = n_subjects // n_folds
        remainder = n_subjects % n_folds
        
        train_splits = []
        test_splits = []
        
        for fold in range(n_folds):
            # Calculate start and end indices for test subjects in this fold
            start_idx = fold * fold_size + min(fold, remainder)
            end_idx = (fold + 1) * fold_size + min(fold + 1, remainder)
            
            # Get test subjects for this fold
            test_subjects = set(unique_subjects[start_idx:end_idx])
            
            # Get train subjects (all subjects except test subjects)
            train_subjects = set(unique_subjects) - test_subjects
            
            # Get indices for train and test subjects
            train_indices = []
            for subject in train_subjects:
                train_indices.extend(subject_to_indices[subject])
            
            test_indices = []
            for subject in test_subjects:
                test_indices.extend(subject_to_indices[subject])
            
            train_splits.append(train_indices)
            test_splits.append(test_indices)
        
        return train_splits, test_splits

def load_caueeg_data(feature_path, label_path, ch_names=None, label_prefix=""):
    """
    Load CAUEEG2 dataset and convert to mne.Epochs
    
    Parameters
    ----------
    feature_path : str
        Path to feature directory
    label_path : str
        Path to label directory
    ch_names : list, optional
        List of channel names
    label_prefix : str, optional
        Prefix to add to printed labels (e.g., "Training" or "Testing")
        
    Returns
    -------
    epochs_list : list
        List of mne.Epochs objects
    labels_list : list
        List of labels
    subjects_list : list
        List of subject identifiers
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
    
    prefix = f"{label_prefix} " if label_prefix else ""
    print(f"Found {len(feature_files)} {prefix.lower()}feature files")
    
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
        if feature_data.shape[2] == 19:
            data = feature_data.transpose(0, 2, 1) 
            print(f"Transposed data shape: {data.shape}")
        else:
            data = feature_data
            print(f"Data shape: {data.shape}")
        
        # Create info object - make sure to only use available channels
        n_channels = data.shape[1]
        used_ch_names = ch_names[:n_channels] if n_channels <= len(ch_names) else [f"ch{i}" for i in range(n_channels)]
        
        info = mne.create_info(ch_names=used_ch_names, sfreq=200.0, ch_types='eeg')  # Note: sfreq=200.0 to match your data
        
        # Create epochs object
        epochs = mne.EpochsArray(data, info)
        
        # Get label
        label = subject_labels[subject_id]
        
        epochs_list.append(epochs)
        labels_list.append(label)
        # Store the actual subject ID instead of a generic "subject_X" string
        subjects_list.append(subject_id)
    
    print(f"Loaded {len(epochs_list)} {prefix.lower()}subjects with labels")
    
    # Print class distribution with class names
    classes, counts = np.unique(labels_list, return_counts=True)
    print(f"{prefix}Class distribution:")
    for c, count in zip(classes, counts):
        print(f"  {CLASS_NAMES[c]}: {count} subjects")
    
    return epochs_list, labels_list, subjects_list

def plot_confusion_matrix(y_true, y_pred, output_dir, use_wandb=False):
    """
    Plot confusion matrix and save to output directory.
    Using class names instead of numeric labels.
    Optionally log to wandb.
    """
    # Get unique classes
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Get class names for the labels
    class_names = [CLASS_NAMES[c] for c in classes]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save locally
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    
    # Log to wandb if enabled
    if use_wandb and wandb.run is not None:
        wandb.log({"confusion_matrix": wandb.Image(cm_path)})
    
    plt.close()

def evaluate_prediction_results(all_preds, output_dir, use_wandb=False):
    """
    Evaluate prediction results and save detailed reports.
    Using class names instead of numeric labels.
    Optionally log to wandb.
    """
    # Extract true and predicted labels
    y_true = all_preds['y_true'].values
    y_pred = all_preds['y_pred'].values
    
    # Print raw prediction information
    print("\nPrediction information:")
    print(f"y_true shape: {y_true.shape}, unique values: {np.unique(y_true, return_counts=True)}")
    print(f"y_pred shape: {y_pred.shape}, unique values: {np.unique(y_pred, return_counts=True)}")
    
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Convert the report to use class names
    named_report = {}
    for key, value in report.items():
        if key.isdigit() or (key.replace('.', '', 1).isdigit() and key.count('.') < 2):
            # This is a class label, convert to class name
            try:
                class_label = int(float(key))
                if class_label in CLASS_NAMES:
                    named_report[CLASS_NAMES[class_label]] = value
                else:
                    named_report[key] = value
            except ValueError:
                named_report[key] = value
        else:
            # This is a metric like 'accuracy', keep as is
            named_report[key] = value
    
    # Add balanced accuracy to the report
    named_report['balanced_accuracy'] = balanced_acc
    
    # Convert to DataFrame and save
    report_df = pd.DataFrame(named_report).transpose()
    report_csv_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(report_csv_path)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, output_dir, use_wandb)
    
    # Count number of examples per class with names
    class_counts = pd.Series(y_true).value_counts().sort_index()
    named_counts = pd.Series({CLASS_NAMES[k]: v for k, v in class_counts.items()})
    named_counts.to_csv(os.path.join(output_dir, 'class_counts.csv'))
    
    # Print summary with class names
    print("\nClassification Report:")
    print(pd.DataFrame(named_report).transpose())
    
    # Log metrics to wandb
    if use_wandb and wandb.run is not None:
        # Log overall metrics
        wandb.log({
            "test_accuracy": report['accuracy'],
            "test_balanced_accuracy": balanced_acc,
            "test_weighted_f1": report['weighted avg']['f1-score'],
            "test_macro_f1": report['macro avg']['f1-score']
        })
        
        # Log per-class metrics
        for class_label, metrics in named_report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                class_name = class_label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
                wandb.log({
                    f"test_{class_name}_precision": metrics['precision'],
                    f"test_{class_name}_recall": metrics['recall'],
                    f"test_{class_name}_f1": metrics['f1-score']
                })
        
        # Log classification report as a table
        table = wandb.Table(dataframe=report_df.reset_index().rename(columns={"index": "class"}))
        wandb.log({"classification_report": table})
    
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
    
    # Initialize wandb if enabled
    if args.use_wandb:
        # Create a run name if not specified
        if args.wandb_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.use_separate_test:
                args.wandb_name = f"TRUE_CV_TRANSFER_{timestamp}"
            else:
                args.wandb_name = f"TRUE_CV_{timestamp}"
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags + (["true_subject_cv", "no_leakage"]),
            config=vars(args)
        )
        
        print(f"Initialized W&B run: {args.wandb_name}")
    
    # Set paths
    train_feature_path = os.path.join(args.data_dir, 'Feature')
    train_label_path = os.path.join(args.data_dir, 'Label')
    
    # Check that training paths exist
    if not os.path.exists(train_feature_path):
        raise FileNotFoundError(f"Training feature directory not found: {train_feature_path}")
    if not os.path.exists(train_label_path):
        raise FileNotFoundError(f"Training label directory not found: {train_label_path}")
    
    print(f"Loading training data from {args.data_dir}")
    
    # Load training data
    train_epochs_list, train_labels_list, train_subjects_list = load_caueeg_data(
        train_feature_path, train_label_path, label_prefix="Training")
    
    # Handle test dataset
    if args.use_separate_test:
        if args.test_data_dir is None:
            raise ValueError("--test_data_dir must be provided when --use_separate_test is set")
            
        test_feature_path = os.path.join(args.test_data_dir, 'Feature')
        test_label_path = os.path.join(args.test_data_dir, 'Label')
        
        # Check that test paths exist
        if not os.path.exists(test_feature_path):
            raise FileNotFoundError(f"Testing feature directory not found: {test_feature_path}")
        if not os.path.exists(test_label_path):
            raise FileNotFoundError(f"Testing label directory not found: {test_label_path}")
        
        print(f"Loading testing data from {args.test_data_dir}")
        
        # Load testing data
        test_epochs_list, test_labels_list, test_subjects_list = load_caueeg_data(
            test_feature_path, test_label_path, label_prefix="Testing")
            
        # Create combined dataset for pl_crossval
        all_epochs_list = train_epochs_list + test_epochs_list
        all_labels_list = train_labels_list + test_labels_list
        all_subjects_list = train_subjects_list + test_subjects_list
        
        # Use all training data for training and all testing data for testing
        n_train = len(train_epochs_list)
        train_indices = [list(range(n_train))]
        test_indices = [list(range(n_train, len(all_epochs_list)))]
        
        # Print number of subjects in each set
        print(f"Using {len(train_subjects_list)} subjects for training and {len(test_subjects_list)} subjects for testing")
    else:
        # Using same dataset for training and testing with true subject-based splits
        all_epochs_list = train_epochs_list
        all_labels_list = train_labels_list
        all_subjects_list = train_subjects_list
        
        # Create subject-based train/test split
        unique_subjects = list(set(all_subjects_list))
        print(f"Total unique subjects: {len(unique_subjects)}")
        
        # Create true subject-based cross-validation splits
        train_indices, test_indices = create_true_subject_cross_validation(
            all_subjects_list, 
            n_folds=args.n_folds, 
            test_fraction=args.test_fraction,
            seed=args.seed
        )
        
        # Print number of subjects in the first fold
        train_subjects = set([all_subjects_list[i] for i in train_indices[0]])
        test_subjects = set([all_subjects_list[i] for i in test_indices[0]])
        print(f"Fold 1: {len(train_subjects)} subjects for training and {len(test_subjects)} subjects for testing")
        print(f"No overlap between train and test subjects: {len(train_subjects.intersection(test_subjects)) == 0}")
    
    # Convert labels to one-hot encoding
    targets = [torch.tensor([1 if i == label else 0 for i in range(3)], dtype=torch.float32) 
              for label in all_labels_list]
    
    # Create dataset
    dataset = EpochsDataset(
        epochs=all_epochs_list,
        targets=targets,
        subjects=all_subjects_list,
        n_epochs=10,  # Use 10 epochs per subject
        padding='repeat'  # Use repeat padding if needed
    )
    
    # Log dataset information to wandb
    if args.use_wandb:
        # Log data distribution for the first fold
        train_fold_labels = [all_labels_list[i] for i in train_indices[0]]
        train_classes, train_counts = np.unique(train_fold_labels, return_counts=True)
        train_distribution = {CLASS_NAMES[c]: int(count) for c, count in zip(train_classes, train_counts)}
        
        test_fold_labels = [all_labels_list[i] for i in test_indices[0]]
        test_classes, test_counts = np.unique(test_fold_labels, return_counts=True)
        test_distribution = {CLASS_NAMES[c]: int(count) for c, count in zip(test_classes, test_counts)}
        
        wandb.log({
            "dataset_size": len(dataset),
            "unique_subjects": len(set(all_subjects_list)),
            "train_size": len(train_indices[0]),
            "test_size": len(test_indices[0]),
            "train_subjects": len(train_subjects),
            "test_subjects": len(test_subjects),
            "train_distribution": train_distribution,
            "test_distribution": test_distribution,
            "using_separate_test": args.use_separate_test
        })
    
    # Print label distribution in train/test sets with class names for the first fold
    print("Fold 1 train set label distribution:")
    train_fold_labels = [all_labels_list[i] for i in train_indices[0]]
    train_classes, train_counts = np.unique(train_fold_labels, return_counts=True)
    for c, count in zip(train_classes, train_counts):
        print(f"  {CLASS_NAMES[c]}: {count} samples")
    
    print("Fold 1 test set label distribution:")
    test_fold_labels = [all_labels_list[i] for i in test_indices[0]]
    test_classes, test_counts = np.unique(test_fold_labels, return_counts=True)
    for c, count in zip(test_classes, test_counts):
        print(f"  {CLASS_NAMES[c]}: {count} samples")
    
    # Number of channels from the first epochs object
    n_ch = len(all_epochs_list[0].ch_names)
    print(f"Using {n_ch} EEG channels")
    
    # Create model for 3-class classification
    model = get_green(
        n_freqs=args.n_freqs,
        kernel_width_s=args.kernel_width_s,
        n_ch=n_ch,
        sfreq=args.sfreq,  # Use 200.0 to match the data
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
    
    # Add wandb callback if enabled
    if args.use_wandb:
        callbacks.append(WandbCallback())
    
    # Choose the LM class based on wandb usage
    pl_module = WandbGreenClassifierLM if args.use_wandb else GreenClassifierLM
    
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
        pl_module=pl_module,
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
    
    # Log training time to wandb
    if args.use_wandb:
        wandb.log({
            "training_time_seconds": training_time,
            "training_time_minutes": training_time/60,
            "training_time_hours": training_time/3600
        })
    
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    # Perform detailed evaluation if requested
    if args.detailed_evaluation:
        print("Performing detailed evaluation...")
        
        # Load predictions
        preds_path = os.path.join(args.output_dir, 'checkpoints/fold0/preds.pkl')
        if os.path.exists(preds_path):
            all_preds = pd.read_pickle(preds_path)
            print(f"Loaded predictions from {preds_path}")
            print(f"Prediction DataFrame columns: {all_preds.columns}")
            print(f"Total predictions: {len(all_preds)}")
            
            # Evaluate predictions
            accuracy = evaluate_prediction_results(all_preds, args.output_dir, args.use_wandb)
            
            # Save accuracy
            with open(os.path.join(args.output_dir, 'accuracy.txt'), 'w') as f:
                f.write(f"Accuracy: {accuracy:.4f}\n")
        else:
            print(f"Warning: Could not find predictions file at {preds_path}")
    
    print(f"All results saved to {args.output_dir}")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()