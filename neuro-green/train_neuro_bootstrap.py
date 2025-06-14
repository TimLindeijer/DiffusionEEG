#!/usr/bin/env python
"""
Bootstrap training script for GREEN model on CAUEEG2 dataset.
Runs multiple bootstrap samples and calculates statistics (mean, std) for metrics.
Logs individual runs and aggregated statistics to wandb.
"""

import os
import time
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    balanced_accuracy_score,
    precision_score, 
    recall_score
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
from scipy import stats

import mne
import torch
from torch.utils.data import DataLoader, random_split
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
    def __init__(self, bootstrap_run=None):
        super().__init__()
        self.best_val_acc = 0
        self.best_epoch = 0
        self.bootstrap_run = bootstrap_run
        
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
                # Add bootstrap run prefix if specified
                metric_key = f"bootstrap_{self.bootstrap_run}_{key}" if self.bootstrap_run is not None else key
                epoch_metrics[metric_key] = value
            
            # Add epoch number
            epoch_key = f"bootstrap_{self.bootstrap_run}_epoch" if self.bootstrap_run is not None else "epoch"
            epoch_metrics[epoch_key] = trainer.current_epoch
            
            # Track best validation accuracy
            val_acc = metrics.get('val_acc', 0)
            if isinstance(val_acc, torch.Tensor):
                val_acc = val_acc.item()
                
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = trainer.current_epoch
                best_acc_key = f"bootstrap_{self.bootstrap_run}_best_val_acc" if self.bootstrap_run is not None else "best_val_acc"
                best_epoch_key = f"bootstrap_{self.bootstrap_run}_best_epoch" if self.bootstrap_run is not None else "best_epoch"
                epoch_metrics[best_acc_key] = self.best_val_acc
                epoch_metrics[best_epoch_key] = self.best_epoch
            
            # Log to wandb
            wandb.log(epoch_metrics)

# Custom classifier that logs to wandb with bootstrap run info
class WandbGreenClassifierLM(GreenClassifierLM):
    """Extension of GreenClassifierLM that logs to wandb with bootstrap info"""
    def __init__(self, *args, bootstrap_run=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bootstrap_run = bootstrap_run
        
    def training_step(self, batch, batch_idx):
        # Call parent method to get loss
        loss = super().training_step(batch, batch_idx)
        
        # Log to wandb with bootstrap prefix
        if wandb.run is not None and batch_idx % 10 == 0:  # Log every 10 batches
            loss_key = f"bootstrap_{self.bootstrap_run}_train_batch_loss" if self.bootstrap_run is not None else "train_batch_loss"
            wandb.log({loss_key: loss.item()})
        
        return loss

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Bootstrap train GREEN model on CAUEEG2 dataset')
    
    # Data and output directories
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training dataset directory')
    parser.add_argument('--test_data_dir', type=str, default=None,
                        help='Path to testing dataset directory (if different from training)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--use_separate_test', action='store_true',
                        help='Use separate dataset for testing')
    
    # Bootstrap parameters
    parser.add_argument('--n_bootstrap', type=int, default=10,
                        help='Number of bootstrap samples to run')
    parser.add_argument('--bootstrap_sample_ratio', type=float, default=1.0,
                        help='Ratio of original training size for each bootstrap sample (default: 1.0 for same size with replacement)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=999,
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
    
    # Shuffle configuration parameters
    parser.add_argument('--shuffle', action='store_true',
                        help='Enable shuffling in EpochsDataset')
    parser.add_argument('--shuffle_first_epoch', action='store_true',
                        help='Enable shuffle_first_epoch in EpochsDataset')
    parser.add_argument('--randomize_epochs', action='store_true',
                        help='Enable randomize_epochs in EpochsDataset')
    
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
    parser.add_argument('--wandb_project', type=str, default='green-caueeg-bootstrap',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity name (username or team name)')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='W&B run name (defaults to timestamp if not specified)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[],
                        help='Tags for the W&B run')
    
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

def load_caueeg_data(feature_path, label_path, ch_names=None, label_prefix="", normalize=False):
    """
    Load CAUEEG2 dataset and convert to mne.Epochs
    [Same implementation as original script]
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
    
    # Initialize lists for epochs, labels, and subjects
    epochs_list = []
    labels_list = []
    subjects_list = []
    
    # Set prefix for printing
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
        
        # Apply min-max normalization if requested
        if normalize:
            for epoch_idx in range(feature_data.shape[0]):
                for channel_idx in range(feature_data.shape[2]):
                    channel_data = feature_data[epoch_idx, :, channel_idx]
                    min_val = np.min(channel_data)
                    max_val = np.max(channel_data)
                    if max_val > min_val:
                        feature_data[epoch_idx, :, channel_idx] = (channel_data - min_val) / (max_val - min_val)
        
        # Feature data is (epochs, times, channels)
        # MNE expects (epochs, channels, times)
        if feature_data.shape[2] == 19:
            data = feature_data.transpose(0, 2, 1) 
        else:
            data = feature_data
        
        # Create info object
        n_channels = data.shape[1]
        used_ch_names = ch_names[:n_channels] if n_channels <= len(ch_names) else [f"ch{i}" for i in range(n_channels)]
        
        info = mne.create_info(ch_names=used_ch_names, sfreq=200.0, ch_types='eeg')
        
        # Create epochs object
        epochs = mne.EpochsArray(data, info)
        
        # Get label
        label = subject_labels[subject_id]
        
        epochs_list.append(epochs)
        labels_list.append(label)
        subjects_list.append(f"subject_{subject_id}")
    
    print(f"Loaded {len(epochs_list)} {prefix.lower()}subjects with labels")
    
    # Print class distribution with class names
    classes, counts = np.unique(labels_list, return_counts=True)
    print(f"{prefix}Class distribution:")
    for c, count in zip(classes, counts):
        print(f"  {CLASS_NAMES[c]}: {count} subjects")
    
    return epochs_list, labels_list, subjects_list

def create_bootstrap_sample(train_indices, labels_list, sample_ratio=1.0, random_state=None):
    """
    Create a bootstrap sample of training indices.
    
    Parameters
    ----------
    train_indices : list
        Original training indices
    labels_list : list
        List of all labels
    sample_ratio : float
        Ratio of original size for bootstrap sample
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    bootstrap_indices : list
        Bootstrap sample indices
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    original_size = len(train_indices)
    sample_size = int(original_size * sample_ratio)
    
    # Sample with replacement
    bootstrap_indices = np.random.choice(train_indices, size=sample_size, replace=True).tolist()
    
    # Print bootstrap sample statistics
    original_labels = [labels_list[i] for i in train_indices]
    bootstrap_labels = [labels_list[i] for i in bootstrap_indices]
    
    print(f"Bootstrap sample created:")
    print(f"  Original size: {original_size}, Bootstrap size: {sample_size}")
    
    # Compare class distributions
    orig_classes, orig_counts = np.unique(original_labels, return_counts=True)
    boot_classes, boot_counts = np.unique(bootstrap_labels, return_counts=True)
    
    print("  Class distribution comparison:")
    for c in orig_classes:
        orig_count = orig_counts[np.where(orig_classes == c)[0][0]] if c in orig_classes else 0
        boot_count = boot_counts[np.where(boot_classes == c)[0][0]] if c in boot_classes else 0
        print(f"    {CLASS_NAMES[c]}: Original={orig_count}, Bootstrap={boot_count}")
    
    return bootstrap_indices

def evaluate_prediction_results(all_preds, output_dir, bootstrap_run=None, use_wandb=False):
    """
    Evaluate prediction results and return metrics dictionary.
    """
    # Extract true and predicted labels
    y_true = all_preds['y_true'].values
    y_pred = all_preds['y_pred'].values
    
    # Calculate metrics
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate balanced/class-weighted metrics manually
    unique_classes = sorted(set(y_true) | set(y_pred))
    n_classes = len(unique_classes)
    
    # Get class weights (inverse of class frequency for balancing)
    class_counts = np.bincount(y_true.astype(int), minlength=max(unique_classes)+1)
    class_counts = class_counts[unique_classes]  # Keep only existing classes
    class_weights = 1.0 / (class_counts / np.sum(class_counts))  # Inverse frequency
    class_weights = class_weights / np.sum(class_weights)  # Normalize to sum to 1
    
    # Calculate balanced precision, recall, f1 (weighted by inverse class frequency)
    balanced_precision = 0
    balanced_recall = 0
    balanced_f1 = 0
    
    for i, cls in enumerate(unique_classes):
        if str(cls) in report:
            cls_precision = report[str(cls)]['precision']
            cls_recall = report[str(cls)]['recall']
            cls_f1 = report[str(cls)]['f1-score']
            
            # Weight by inverse class frequency
            balanced_precision += cls_precision * class_weights[i]
            balanced_recall += cls_recall * class_weights[i]
            balanced_f1 += cls_f1 * class_weights[i]
    
    # Create metrics dictionary
    metrics = {
        'accuracy': report['accuracy'],
        'balanced_accuracy': balanced_acc,
        
        # Macro averages (equal weight per class)
        'macro_f1': report['macro avg']['f1-score'],
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall'],
        
        # Weighted averages (weighted by support)
        'weighted_f1': report['weighted avg']['f1-score'],
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall': report['weighted avg']['recall'],
        
        # Balanced averages (weighted by inverse class frequency)
        'balanced_precision': balanced_precision,
        'balanced_recall': balanced_recall,
        'balanced_f1': balanced_f1,
    }
    
    # Add per-class metrics
    unique_classes = sorted(set(y_true) | set(y_pred))
    for class_idx in unique_classes:
        if str(class_idx) in report:
            class_name = CLASS_NAMES.get(class_idx, f"class_{class_idx}")
            metrics[f'{class_name}_precision'] = report[str(class_idx)]['precision']
            metrics[f'{class_name}_recall'] = report[str(class_idx)]['recall']
            metrics[f'{class_name}_f1'] = report[str(class_idx)]['f1-score']
    
    # Save results for this bootstrap run
    if bootstrap_run is not None:
        bootstrap_dir = os.path.join(output_dir, f'bootstrap_{bootstrap_run}')
        os.makedirs(bootstrap_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(bootstrap_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
    
    # Log to wandb with bootstrap prefix
    if use_wandb and wandb.run is not None:
        wandb_metrics = {}
        for key, value in metrics.items():
            wandb_key = f"bootstrap_{bootstrap_run}_{key}" if bootstrap_run is not None else key
            wandb_metrics[wandb_key] = value
        wandb.log(wandb_metrics)
    
    return metrics

def calculate_bootstrap_statistics(all_metrics, output_dir, use_wandb=False):
    """
    Calculate statistics (mean, std, confidence intervals) across bootstrap runs.
    """
    print("\n" + "="*50)
    print("BOOTSTRAP STATISTICS")
    print("="*50)
    
    # Convert list of dictionaries to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Calculate statistics
    statistics_dict = {}
    
    for metric in metrics_df.columns:
        values = metrics_df[metric].values
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # Sample standard deviation
        
        # Calculate 95% confidence interval
        confidence_level = 0.95
        alpha = 1 - confidence_level
        df_freedom = len(values) - 1
        t_critical = stats.t.ppf(1 - alpha/2, df_freedom)
        margin_error = t_critical * (std_val / np.sqrt(len(values)))
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error
        
        statistics_dict[metric] = {
            'mean': mean_val,
            'std': std_val,
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_runs': len(values)
        }
        
        # Print statistics
        print(f"\n{metric.upper()}:")
        print(f"  Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
        print(f"  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
        print(f"  Median: {np.median(values):.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Save statistics
    statistics_df = pd.DataFrame(statistics_dict).T
    statistics_df.to_csv(os.path.join(output_dir, 'bootstrap_statistics.csv'))
    
    # Save detailed results
    metrics_df.to_csv(os.path.join(output_dir, 'bootstrap_all_runs.csv'), index=False)
    
    # Save summary
    with open(os.path.join(output_dir, 'bootstrap_summary.json'), 'w') as f:
        json.dump(statistics_dict, f, indent=4)
    
    # Log to wandb
    if use_wandb and wandb.run is not None:
        wandb_stats = {}
        for metric, metric_stats in statistics_dict.items():
            wandb_stats[f"{metric}_mean"] = metric_stats['mean']
            wandb_stats[f"{metric}_std"] = metric_stats['std']
            wandb_stats[f"{metric}_ci_lower"] = metric_stats['ci_lower']
            wandb_stats[f"{metric}_ci_upper"] = metric_stats['ci_upper']
        
        # Add summary table
        summary_table = wandb.Table(dataframe=statistics_df.reset_index().rename(columns={"index": "metric"}))
        wandb_stats["bootstrap_statistics_table"] = summary_table
        
        # Add individual runs table
        runs_table = wandb.Table(dataframe=metrics_df.reset_index().rename(columns={"index": "bootstrap_run"}))
        wandb_stats["bootstrap_all_runs_table"] = runs_table
        
        wandb.log(wandb_stats)
    
    return statistics_dict

def main():
    """Main function to run bootstrap training and evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print(f"Running bootstrap training with {args.n_bootstrap} bootstrap samples")
    print(f"Bootstrap sample ratio: {args.bootstrap_sample_ratio}")
    
    # Initialize wandb if enabled
    if args.use_wandb:
        # Create a run name if not specified
        if args.wandb_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            shuffle_config = f"{'T' if args.shuffle else 'F'}{'T' if args.shuffle_first_epoch else 'F'}{'T' if args.randomize_epochs else 'F'}"
            if args.use_separate_test:
                args.wandb_name = f"BOOTSTRAP_GREEN_TRANSFER_{shuffle_config}_{args.n_bootstrap}runs_{timestamp}"
            else:
                args.wandb_name = f"BOOTSTRAP_GREEN_CAUEEG_{shuffle_config}_{args.n_bootstrap}runs_{timestamp}"
        
        # Add bootstrap tags
        bootstrap_tags = [
            f"bootstrap_{args.n_bootstrap}",
            f"sample_ratio_{args.bootstrap_sample_ratio}",
            f"shuffle_{args.shuffle}",
            f"shuffle_first_{args.shuffle_first_epoch}",
            f"randomize_{args.randomize_epochs}"
        ]
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags + bootstrap_tags + (["transfer_learning"] if args.use_separate_test else []),
            config=vars(args)
        )
        
        print(f"Initialized W&B run: {args.wandb_name}")
    
    # Load data (same as original script)
    train_feature_path = os.path.join(args.data_dir, 'Feature')
    train_label_path = os.path.join(args.data_dir, 'Label')
    
    if not os.path.exists(train_feature_path):
        raise FileNotFoundError(f"Training feature directory not found: {train_feature_path}")
    if not os.path.exists(train_label_path):
        raise FileNotFoundError(f"Training label directory not found: {train_label_path}")
    
    print(f"Loading training data from {args.data_dir}")
    
    # Load training data
    train_epochs_list, train_labels_list, train_subjects_list = load_caueeg_data(
        train_feature_path, train_label_path, label_prefix="Training", normalize=False)
    
    # Handle test dataset
    if args.use_separate_test:
        if args.test_data_dir is None:
            raise ValueError("--test_data_dir must be provided when --use_separate_test is set")
            
        test_feature_path = os.path.join(args.test_data_dir, 'Feature')
        test_label_path = os.path.join(args.test_data_dir, 'Label')
        
        if not os.path.exists(test_feature_path):
            raise FileNotFoundError(f"Testing feature directory not found: {test_feature_path}")
        if not os.path.exists(test_label_path):
            raise FileNotFoundError(f"Testing label directory not found: {test_label_path}")
        
        print(f"Loading testing data from {args.test_data_dir}")
        
        # Load testing data
        test_epochs_list, test_labels_list, test_subjects_list = load_caueeg_data(
            test_feature_path, test_label_path, label_prefix="Testing", normalize=False)
            
        # Create combined dataset
        all_epochs_list = train_epochs_list + test_epochs_list
        all_labels_list = train_labels_list + test_labels_list
        all_subjects_list = train_subjects_list + test_subjects_list
        
        # Fixed train and test indices for all bootstrap runs
        original_train_indices = list(range(len(train_epochs_list)))
        test_indices = [i + len(train_epochs_list) for i in range(len(test_epochs_list))]
        
    else:
        # Using same dataset for training and testing
        all_epochs_list = train_epochs_list
        all_labels_list = train_labels_list
        all_subjects_list = train_subjects_list
        
        # Split data into train/test (80/20) - fixed for all bootstrap runs
        n_subjects = len(all_epochs_list)
        n_train = int(0.8 * n_subjects)
        
        indices = list(range(n_subjects))
        np.random.shuffle(indices)
        
        original_train_indices = indices[:n_train]
        test_indices = indices[n_train:]
    
    print(f"Original training set size: {len(original_train_indices)}")
    print(f"Test set size: {len(test_indices)}")
    
    # Convert labels to one-hot encoding
    targets = [torch.tensor([1 if i == label else 0 for i in range(3)], dtype=torch.float32) 
              for label in all_labels_list]
    
    # Number of channels
    n_ch = len(all_epochs_list[0].ch_names)
    print(f"Using {n_ch} EEG channels")
    
    # Run bootstrap samples
    all_metrics = []
    start_time = time.time()
    
    for bootstrap_run in range(args.n_bootstrap):
        print(f"\n{'='*60}")
        print(f"BOOTSTRAP RUN {bootstrap_run + 1}/{args.n_bootstrap}")
        print(f"{'='*60}")
        
        # Set different seed for each bootstrap run
        bootstrap_seed = args.seed + bootstrap_run
        set_seed(bootstrap_seed)
        print(f"Bootstrap run {bootstrap_run + 1} using seed: {bootstrap_seed}")
        
        # Create bootstrap sample
        bootstrap_train_indices = create_bootstrap_sample(
            original_train_indices, 
            all_labels_list, 
            sample_ratio=args.bootstrap_sample_ratio,
            random_state=bootstrap_seed
        )
        
        # Create dataset with bootstrap sample
        dataset = EpochsDataset(
            epochs=all_epochs_list,
            targets=targets,
            subjects=all_subjects_list,
            n_epochs=10,
            padding='repeat',
            shuffle=args.shuffle,
            shuffle_first_epoch=args.shuffle_first_epoch,
            randomize_epochs=args.randomize_epochs
        )
        
        # Create model (fresh model for each bootstrap run)
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
            bi_out=[args.hidden_dim[0] // 4],
            dtype=torch.float32,
            out_dim=3
        )
        
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
            callbacks.append(WandbCallback(bootstrap_run=bootstrap_run))
        
        # Choose the LM class based on wandb usage
        if args.use_wandb:
            pl_module = lambda *args, **kwargs: WandbGreenClassifierLM(*args, bootstrap_run=bootstrap_run, **kwargs)
        else:
            pl_module = GreenClassifierLM
        
        # Create checkpoint directory for this bootstrap run
        bootstrap_ckpt_dir = os.path.join(args.output_dir, f'bootstrap_{bootstrap_run}', 'checkpoints')
        
        # Train model
        pl_crossval_output, _ = pl_crossval(
            model, 
            dataset=dataset,
            n_epochs=args.max_epochs,
            save_preds=True,
            ckpt_prefix=bootstrap_ckpt_dir,
            train_splits=[bootstrap_train_indices],
            test_splits=[test_indices],
            batch_size=args.batch_size,
            pl_module=pl_module,
            num_workers=args.num_workers,
            callbacks=callbacks,
            pl_params={
                'weight_decay': args.weight_decay,
                'lr': args.learning_rate
            }
        )
        
        # Evaluate this bootstrap run
        preds_path = os.path.join(bootstrap_ckpt_dir, 'fold0/preds.pkl')
        if os.path.exists(preds_path):
            all_preds = pd.read_pickle(preds_path)
            metrics = evaluate_prediction_results(
                all_preds, 
                args.output_dir, 
                bootstrap_run=bootstrap_run, 
                use_wandb=args.use_wandb
            )
            all_metrics.append(metrics)
            
            print(f"Bootstrap run {bootstrap_run + 1} completed:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        else:
            print(f"Warning: Could not find predictions file at {preds_path}")
    
    # Calculate bootstrap statistics
    if all_metrics:
        statistics_dict = calculate_bootstrap_statistics(all_metrics, args.output_dir, args.use_wandb)
        
        # Calculate total training time
        total_time = time.time() - start_time
        
        # Save timing information
        with open(os.path.join(args.output_dir, 'timing_info.txt'), 'w') as f:
            f.write(f"Total bootstrap training time: {total_time:.2f} seconds\n")
            f.write(f"Total bootstrap training time: {total_time/60:.2f} minutes\n")
            f.write(f"Total bootstrap training time: {total_time/3600:.2f} hours\n")
            f.write(f"Average time per bootstrap run: {total_time/args.n_bootstrap:.2f} seconds\n")
        
        # Log timing to wandb
        if args.use_wandb:
            wandb.log({
                "total_training_time_seconds": total_time,
                "total_training_time_minutes": total_time/60,
                "total_training_time_hours": total_time/3600,
                "avg_time_per_bootstrap_seconds": total_time/args.n_bootstrap,
                "n_bootstrap_runs": args.n_bootstrap
            })
        
        print(f"\nBootstrap training completed in {total_time/60:.2f} minutes")
        print(f"Average time per bootstrap run: {total_time/args.n_bootstrap/60:.2f} minutes")
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL BOOTSTRAP SUMMARY")
        print(f"{'='*60}")
        print(f"Number of bootstrap runs: {args.n_bootstrap}")
        print(f"Key metrics (Mean ± Std):")
        print(f"  Accuracy: {statistics_dict['accuracy']['mean']:.4f} ± {statistics_dict['accuracy']['std']:.4f}")
        print(f"  Balanced Accuracy: {statistics_dict['balanced_accuracy']['mean']:.4f} ± {statistics_dict['balanced_accuracy']['std']:.4f}")
        print(f"  Macro F1: {statistics_dict['macro_f1']['mean']:.4f} ± {statistics_dict['macro_f1']['std']:.4f}")
        
    else:
        print("No bootstrap runs completed successfully!")
    
    print(f"\nAll results saved to {args.output_dir}")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()