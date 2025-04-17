#!/usr/bin/env python
"""
Training script for GREEN model on CAUEEG2 dataset with wandb integration.
Fixed version that works with the standard pl_crossval function.
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
    
    # Print class distribution with class names
    classes, counts = np.unique(labels_list, return_counts=True)
    print("Class distribution:")
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
            args.wandb_name = f"GREEN_CAUEEG_{timestamp}"
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags,
            config=vars(args)
        )
        
        print(f"Initialized W&B run: {args.wandb_name}")
    
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
    
    # Log dataset information to wandb
    if args.use_wandb:
        classes, counts = np.unique(labels_list, return_counts=True)
        class_distribution = {CLASS_NAMES[c]: int(count) for c, count in zip(classes, counts)}
        wandb.log({
            "dataset_size": len(dataset),
            "dataset_class_distribution": class_distribution
        })
    
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
    
    # Add wandb callback if enabled
    if args.use_wandb:
        callbacks.append(WandbCallback())
    
    # Choose the LM class based on wandb usage
    pl_module = WandbGreenClassifierLM if args.use_wandb else GreenClassifierLM
    
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
    
    # Print label distribution in train/test sets with class names
    train_labels = [labels_list[i] for i in train_indices[0]]
    test_labels = [labels_list[i] for i in test_indices[0]]
    
    print("Train set label distribution:")
    train_classes, train_counts = np.unique(train_labels, return_counts=True)
    train_distribution = {}
    for c, count in zip(train_classes, train_counts):
        print(f"  {CLASS_NAMES[c]}: {count} subjects")
        train_distribution[CLASS_NAMES[c]] = int(count)
    
    print("Test set label distribution:")
    test_classes, test_counts = np.unique(test_labels, return_counts=True)
    test_distribution = {}
    for c, count in zip(test_classes, test_counts):
        print(f"  {CLASS_NAMES[c]}: {count} subjects")
        test_distribution[CLASS_NAMES[c]] = int(count)
    
    # Log split information to wandb
    if args.use_wandb:
        wandb.log({
            "train_size": len(train_indices[0]),
            "test_size": len(test_indices[0]),
            "train_distribution": train_distribution,
            "test_distribution": test_distribution
        })
    
    # Start timing
    start_time = time.time()
    
    # Train model with the fixed approach (no logger parameter)
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