#!/usr/bin/env python
"""
EEG Classification using GREEN (Gabor Riemann EEGNet) architecture with fixes
for variable-sized data batching and stratified cross-validation.

This script loads preprocessed EEG data from the CAUEEG2 dataset and trains
a GREEN model to classify subjects into different cognitive states:
- Healthy Control (0)
- Mild Cognitive Impairment (1)
- Dementia (2)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import pandas as pd
import argparse
import warnings

# Enable Tensor Core optimizations for H100 GPU
torch.set_float32_matmul_precision('high')  # Use Tensor Cores for faster matrix multiplications

# Import GREEN model components
from green.wavelet_layers import RealCovariance
from green.research_code.pl_utils import get_green, GreenClassifierLM
from green.research_code.crossval_utils import pl_crossval

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def setup_logging():
    try:
        import tensorboard
        print("TensorBoard is available for logging")
    except ImportError:
        warnings.warn(
            "TensorBoard not found. If you want to use TensorBoard for logging, "
            "install it with: pip install tensorboard"
        )
        print("Using CSVLogger as the default logger")


def custom_balanced_accuracy(y_true, y_pred, labels=None):
    """
    Custom implementation of balanced accuracy that handles missing classes.
    Uses confusion matrix to calculate per-class recall and then averages.
    """
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    # Convert to numpy arrays if they're tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # If after filtering there are no samples, return 0
    if len(y_true) == 0:
        return 0.0
    
    # Use confusion matrix with labels to compute balanced accuracy manually
    if labels is not None:
        # Get confusion matrix with specified labels
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Calculate per-class recall (true positive rate)
        per_class_recall = np.zeros(len(labels))
        for i in range(len(labels)):
            if np.sum(y_true == labels[i]) > 0:  # Only if class exists in y_true
                per_class_recall[i] = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        
        # Count classes that actually appear in y_true
        active_classes = np.sum([np.sum(y_true == label) > 0 for label in labels])
        
        # Average the per-class recalls for classes that exist in y_true
        if active_classes > 0:
            return np.sum(per_class_recall) / active_classes
        return 0.0
    else:
        # If no labels provided, use the unique values in y_true and y_pred
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        return custom_balanced_accuracy(y_true, y_pred, labels=unique_labels)


class CAUEEG2Dataset(Dataset):
    """
    Dataset class for CAUEEG2 preprocessed EEG data.
    """
    def __init__(self, feature_path, label_path, transform=lambda x: torch.tensor(x), 
                 max_epochs=10, pad_to_length=None):
        """
        Args:
            feature_path: Path to the features directory
            label_path: Path to the labels directory
            transform: Transform to apply to the data (default: convert to tensor)
            max_epochs: Maximum number of epochs to use per subject
            pad_to_length: If set, pad/truncate all time series to this length
        """
        self.feature_path = feature_path
        self.max_epochs = max_epochs
        self.pad_to_length = pad_to_length
        
        # Load labels
        labels = np.load(os.path.join(label_path, 'label.npy'))
        self.subject_ids = labels[:, 1]
        self.labels = labels[:, 0]
        
        # Get feature filenames
        self.feature_files = {}
        for subject_id in self.subject_ids:
            subject_files = []
            # Check for all possible feature files for this subject
            base_pattern = f'feature_{subject_id:02d}'
            for file in os.listdir(feature_path):
                if file.startswith(base_pattern) and file.endswith('.npy'):
                    subject_files.append(file)
            
            if subject_files:
                self.feature_files[subject_id] = subject_files
            else:
                print(f"Warning: No feature files found for subject {subject_id}")
        
        # Filter out subjects with no feature files
        valid_indices = [i for i, sid in enumerate(self.subject_ids) if sid in self.feature_files]
        self.subject_ids = self.subject_ids[valid_indices]
        self.labels = self.labels[valid_indices]
        
        self.transform = transform
        
        # Encode labels as one-hot
        self.n_classes = len(np.unique(self.labels))
        self.one_hot_labels = np.zeros((len(self.labels), self.n_classes))
        for i, label in enumerate(self.labels):
            self.one_hot_labels[i, label] = 1
            
        print(f"Loaded {len(self.subject_ids)} subjects with {self.n_classes} classes")
        print(f"Class distribution: {np.bincount(self.labels)}")
        
        # Determine common dimensions
        self._determine_common_dimensions()

    def _determine_common_dimensions(self):
        """Determine common dimensions (channels, time points) across dataset"""
        # Check first few samples to determine common dimensions
        sample_features = []
        for i in range(min(10, len(self.subject_ids))):
            subject_id = self.subject_ids[i]
            feature_file = self.feature_files[subject_id][0]
            feature_path = os.path.join(self.feature_path, feature_file)
            try:
                feature_data = np.load(feature_path)
                sample_features.append(feature_data)
            except Exception as e:
                print(f"Error loading {feature_path}: {e}")
                continue
        
        if not sample_features:
            raise ValueError("Could not load any sample features to determine dimensions")
        
        # Determine common n_channels
        n_channels_list = [f.shape[2] for f in sample_features]
        self.n_channels = max(set(n_channels_list), key=n_channels_list.count)
        
        # Set default padding length if not specified
        if self.pad_to_length is None:
            time_lengths = [f.shape[1] for f in sample_features]
            self.pad_to_length = max(time_lengths)  # Use longest time series
            
        print(f"Common dimensions: {self.n_channels} channels, {self.pad_to_length} time points")

    def __len__(self):
        return len(self.subject_ids)

    def _prepare_epochs(self, feature_data):
        """Prepare and standardize epochs from feature data"""
        # Handle time dimension padding/truncation
        if self.pad_to_length:
            n_epochs, curr_length, n_channels = feature_data.shape
            if curr_length > self.pad_to_length:
                # Truncate time dimension
                feature_data = feature_data[:, :self.pad_to_length, :]
            elif curr_length < self.pad_to_length:
                # Pad time dimension with zeros
                padding = np.zeros((n_epochs, self.pad_to_length - curr_length, n_channels))
                feature_data = np.concatenate([feature_data, padding], axis=1)
        
        # Handle channel dimension if needed
        if feature_data.shape[2] != self.n_channels:
            print(f"Warning: Subject has {feature_data.shape[2]} channels instead of {self.n_channels}")
            if feature_data.shape[2] > self.n_channels:
                # Take only the first n_channels
                feature_data = feature_data[:, :, :self.n_channels]
            else:
                # Pad with zeros
                n_epochs, time_len, curr_channels = feature_data.shape
                padding = np.zeros((n_epochs, time_len, self.n_channels - curr_channels))
                feature_data = np.concatenate([feature_data, padding], axis=2)
        
        return feature_data

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        feature_file = self.feature_files[subject_id][0]  # Use first feature file if multiple
        
        # Load feature data
        feature_path = os.path.join(self.feature_path, feature_file)
        feature_data = np.load(feature_path)
        
        # Prepare epochs
        feature_data = self._prepare_epochs(feature_data)
        
        # Ensure we always have exactly max_epochs
        n_current_epochs = feature_data.shape[0]
        if n_current_epochs < self.max_epochs:
            # If we have fewer epochs than required, duplicate some existing ones
            indices_to_repeat = np.random.choice(
                n_current_epochs, 
                size=self.max_epochs - n_current_epochs, 
                replace=True
            )
            extra_epochs = feature_data[indices_to_repeat]
            feature_data = np.concatenate([feature_data, extra_epochs], axis=0)
        elif n_current_epochs > self.max_epochs:
            # If we have more epochs than required, select a random subset
            indices_to_keep = np.random.choice(
                n_current_epochs, 
                size=self.max_epochs, 
                replace=False
            )
            feature_data = feature_data[indices_to_keep]
        
        # Get one-hot encoded label
        label = self.one_hot_labels[idx]
        
        # Apply transform
        feature_tensor = self.transform(feature_data)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return feature_tensor, label_tensor


def custom_collate_fn(batch):
    """
    Custom collate function that ensures all samples in a batch have the same dimensions.
    This helps avoid the "trying to resize storage that is not resizable" error.
    """
    # Extract features and labels
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Ensure all features have the same shape
    shapes = [f.shape for f in features]
    if len(set(shapes)) > 1:
        print(f"Warning: Inconsistent shapes in batch: {shapes}")
        # Find the minimum dimensions
        min_epochs = min(s[0] for s in shapes)
        min_channels = min(s[1] for s in shapes)
        min_times = min(s[2] for s in shapes)
        
        # Resize all tensors to the minimum dimensions
        features = [f[:min_epochs, :min_channels, :min_times] for f in features]
    
    try:
        # Stack tensors
        features = torch.stack(features)
        labels = torch.stack(labels)
    except Exception as e:
        print(f"Error stacking tensors: {e}")
        print(f"Feature shapes: {[f.shape for f in features]}")
        print(f"Label shapes: {[l.shape for l in labels]}")
        # Fall back solution - take the first valid shape and make all match it
        reference_shape = features[0].shape
        features = [f[:reference_shape[0], :reference_shape[1], :reference_shape[2]] 
                   if f.shape != reference_shape else f
                   for f in features]
        features = torch.stack(features)
        labels = torch.stack(labels)
    
    return features, labels


def preprocess_features(data):
    """
    Preprocess EEG features - convert to PyTorch tensor and normalize
    """
    # Convert to PyTorch tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # Normalize data to μV (assuming data is in V)
    data_tensor = data_tensor * 1e6
    
    # Transpose from (epochs, times, channels) to (epochs, channels, times)
    return data_tensor.transpose(1, 2)


def get_dataloader(dataset, indices, batch_size, shuffle=False, num_workers=0):
    """
    Create a dataloader with custom collate function
    """
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=False
    )


# Custom implementation of GreenClassifierLM for fixing the class warnings
class FixedGreenClassifierLM(GreenClassifierLM):
    def validation_step(self, batch, batch_idx):
        if self.use_age:
            x, age, y_true = batch
            y_pred = self.model(x, age)
        else:
            x, y_true = batch
            y_pred = self.model(x)

        val_loss = self.criterion(y_pred, y_true)
        y_pred = y_pred.cpu()
        y_true = y_true.to(self.data_type)
        y_true = y_true.cpu()

        # Get all possible class labels (0, 1, 2 for your case)
        all_classes = list(range(y_pred.shape[1]))
        
        val_acc = custom_balanced_accuracy(
            y_pred=torch.argmax(y_pred, dim=1).numpy(),
            y_true=torch.argmax(y_true, dim=1).numpy(),
            labels=all_classes)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        if self.use_age:
            x, age, y_true = batch
            y_pred = self.model(x, age)
        else:
            x, y_true = batch
            y_pred = self.model(x)

        test_loss = self.criterion(y_pred, y_true)

        y_pred = y_pred.cpu()
        y_true = y_true.to(self.data_type)
        y_true = y_true.cpu()

        # Get all possible class labels (0, 1, 2 for your case)
        all_classes = list(range(y_pred.shape[1]))
        
        test_score = custom_balanced_accuracy(
            y_pred=torch.argmax(y_pred, dim=1).numpy(),
            y_true=torch.argmax(y_true, dim=1).numpy(),
            labels=all_classes)
            
        self.log("test_loss", test_loss)
        self.log("test_score", test_score)

    def predict_step(self, batch, batch_idx):
        if self.use_age:
            x, age, y_true = batch
            y_pred = self.model(x, age).cpu()
        else:
            x, y_true = batch
            y_pred = self.model(x).cpu()
        y_true = y_true.to(self.data_type)
        y_true = y_true.cpu()
        
        # Get all possible class labels (0, 1, 2 for your case)
        all_classes = list(range(y_pred.shape[1]))
        
        pred_acc = custom_balanced_accuracy(
            y_pred=torch.argmax(y_pred, dim=1).numpy(),
            y_true=torch.argmax(y_true, dim=1).numpy(),
            labels=all_classes)
            
        print("pred_acc = ", pred_acc)

        self.predict_outputs.append(pd.DataFrame(dict(
            y_pred=torch.argmax(y_pred, dim=1).numpy(),
            y_true=torch.argmax(y_true, dim=1).numpy(),
            y_pred_proba=tuple(y_pred.numpy()),
            y_true_proba=tuple(y_true.numpy())
        )))
        return pd.DataFrame(dict(
            y_pred=torch.argmax(y_pred, dim=1).numpy(),
            y_true=torch.argmax(y_true, dim=1).numpy(),
            y_pred_proba=tuple(y_pred.numpy()),
            y_true_proba=tuple(y_true.numpy())
        ))


# Add a custom wrapper for GreenClassifierLM that provides a forward method
class GreenClassifierWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Ensure model is on the same device as input
        device = x.device
        if next(self.model.parameters()).device != device:
            self.model = self.model.to(device)
        return self.model.model(x)


def train_model(args):
    """
    Train GREEN model on CAUEEG2 dataset using cross-validation
    """
    # Check TensorBoard availability
    setup_logging()
    
    # Create dataset with standardized dimensions
    dataset = CAUEEG2Dataset(
        feature_path=args.feature_path,
        label_path=args.label_path,
        transform=preprocess_features,
        max_epochs=args.max_epochs,
        pad_to_length=args.pad_to_length
    )
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Get dimensions from first item
    sample_data, _ = dataset[0]
    n_epochs, n_channels, n_times = sample_data.shape
    print(f"Data shape: {sample_data.shape}")
    
    # Create model
    green_model = get_green(
        n_freqs=args.n_freqs,          # Number of frequency bands
        kernel_width_s=args.kernel_width,  # Width of wavelet kernels in seconds
        n_ch=n_channels,               # Number of EEG channels
        sfreq=args.sampling_freq,      # Sampling frequency
        orth_weights=True,             # Use orthogonal weights
        dropout=args.dropout,          # Dropout rate
        hidden_dim=args.hidden_dims,   # Hidden layer dimensions
        pool_layer=RealCovariance(),   # Use covariance pooling
        bi_out=args.bimap_dims,        # BiMap output dimensions
        out_dim=dataset.n_classes      # Number of output classes
    )
    # Move model to device immediately
    green_model = green_model.to(device)
    print(green_model)
    
    # Setup cross-validation - use stratified CV to maintain class distribution
    n_splits = args.cv_folds
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Prepare train/test splits with stratification
    train_splits = []
    test_splits = []
    print(f"Setting up {n_splits}-fold stratified cross-validation")
    for train_idx, test_idx in cv.split(np.arange(len(dataset)), dataset.labels):
        # Check class distribution in this fold
        train_labels = dataset.labels[train_idx]
        test_labels = dataset.labels[test_idx]
        print(f"Train set class dist: {np.bincount(train_labels)}")
        print(f"Test set class dist: {np.bincount(test_labels)}")
        
        train_splits.append(train_idx)
        test_splits.append(test_idx)
    
    # Define all possible class labels
    all_classes = np.arange(dataset.n_classes)
    class_names = ['Healthy', 'MCI', 'Dementia']
    
    # Manual cross-validation since we need custom dataloaders
    results = []
    fold_predictions = []
    
    for fold_idx in range(n_splits):
        print(f"\nTraining fold {fold_idx + 1}/{n_splits}")
        
        # Get train/test indices for this fold
        train_indices = train_splits[fold_idx]
        test_indices = test_splits[fold_idx]
        
        # Create dataloaders with custom collate function
        train_loader = get_dataloader(
            dataset, train_indices, batch_size=args.batch_size, 
            shuffle=True, num_workers=args.num_workers
        )
        
        test_loader = get_dataloader(
            dataset, test_indices, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers
        )
        
        # Reset model weights for this fold
        model = get_green(
            n_freqs=args.n_freqs,
            kernel_width_s=args.kernel_width,
            n_ch=n_channels,
            sfreq=args.sampling_freq,
            orth_weights=True,
            dropout=args.dropout,
            hidden_dim=args.hidden_dims,
            pool_layer=RealCovariance(),
            bi_out=args.bimap_dims,
            out_dim=dataset.n_classes
        )
        # Move model to device
        model = model.to(device)
        
        # Create Lightning model with our fixed class
        lightning_model = FixedGreenClassifierLM(
            model=model,
            weight_decay=args.weight_decay
        )
        
        # Create trainer
        from lightning import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{args.checkpoint_dir}/fold{fold_idx}",
            save_top_k=1,
            monitor="val_acc",
            mode="max",
            filename="model-{epoch:02d}-{val_acc:.4f}"
        )
        
        early_stopping = EarlyStopping(
            monitor="val_acc",
            patience=10,
            mode="max"
        )
        
        trainer = Trainer(
            max_epochs=args.epochs,
            accelerator="gpu" if torch.cuda.is_available() and not args.no_cuda else "cpu",
            devices=1,
            default_root_dir=f"{args.checkpoint_dir}/fold{fold_idx}",
            callbacks=[checkpoint_callback, early_stopping],
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        
        # Train model
        trainer.fit(lightning_model, train_loader, test_loader)
        
        # Test model
        test_results = trainer.test(lightning_model, test_loader)
        results.append(test_results[0])
        
        # Save predictions
        all_predictions = []
        all_targets = []
        all_probs = []
        
        # Create a proper wrapper that has a forward method
        wrapper_model = GreenClassifierWrapper(lightning_model)
        wrapper_model.eval()  # Set to evaluation mode
        
        # Explicitly move wrapper to the device
        wrapper_model = wrapper_model.to(device)
        
        for batch in test_loader:
            features, targets = batch
            features = features.to(device)
            targets = targets.to(device)  # Also move targets to device
            
            with torch.no_grad():
                # Use the wrapper model's forward method (which now ensures device consistency)
                outputs = wrapper_model(features)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                targets_idx = torch.argmax(targets, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets_idx.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        # Create fold predictions dataframe
        fold_df = pd.DataFrame({
            'true_label': all_targets,
            'predicted_label': all_predictions,
            'test_index': test_indices,
            'fold': fold_idx
        })
        
        # Add probability columns
        for i in range(dataset.n_classes):
            fold_df[f'prob_class_{i}'] = [probs[i] for probs in all_probs]
        
        fold_predictions.append(fold_df)
        
        # Save fold predictions to CSV
        fold_df.to_csv(f"{args.checkpoint_dir}/fold{fold_idx}/predictions.csv", index=False)
        
        # Generate fold-specific metrics with explicit labels
        # This avoids warnings about classes not in y_true
        print(f"\nFold {fold_idx + 1} Classification Report:")
        fold_report = classification_report(
            all_targets, 
            all_predictions,
            labels=all_classes,  # Use all classes, even if not in this fold
            target_names=class_names,
            zero_division=0
        )
        print(fold_report)
        
        with open(f"{args.checkpoint_dir}/fold{fold_idx}/classification_report.txt", "w") as f:
            f.write(fold_report)
        
        # Generate confusion matrix with all classes
        fold_cm = confusion_matrix(
            all_targets, 
            all_predictions, 
            labels=all_classes  # Use all classes
        )
        
        # Plot fold confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(fold_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Fold {fold_idx + 1} Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = fold_cm.max() / 2.
        for i in range(fold_cm.shape[0]):
            for j in range(fold_cm.shape[1]):
                plt.text(j, i, format(fold_cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if fold_cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save plot
        plt.savefig(f"{args.checkpoint_dir}/fold{fold_idx}/confusion_matrix.png")
        plt.close()
    
    # Aggregate results
    avg_results = {key: np.mean([result[key] for result in results]) for key in results[0].keys()}
    print("\nAverage results across all folds:")
    for key, value in avg_results.items():
        print(f"{key}: {value:.4f}")
    
    # Save average results
    with open(f"{args.checkpoint_dir}/avg_results.txt", "w") as f:
        for key, value in avg_results.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # Combine all fold predictions
    if fold_predictions:
        combined_preds = pd.concat(fold_predictions)
        combined_preds.to_csv(f"{args.checkpoint_dir}/all_predictions.csv", index=False)
        
        # Generate classification report for all predictions
        print("\nOverall Classification Report:")
        report = classification_report(
            combined_preds['true_label'], 
            combined_preds['predicted_label'],
            labels=all_classes,  # Use all possible classes
            target_names=class_names,
            zero_division=0
        )
        print(report)
        
        with open(f"{args.checkpoint_dir}/classification_report.txt", "w") as f:
            f.write(report)
        
        # Generate confusion matrix for all predictions
        cm = confusion_matrix(
            combined_preds['true_label'], 
            combined_preds['predicted_label'],
            labels=all_classes  # Use all possible classes
        )
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Overall Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save plot
        plt.savefig(os.path.join(args.checkpoint_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Generate per-class metrics
        class_metrics = pd.DataFrame(index=class_names)
        
        # Calculate per-class accuracy
        for i, class_name in enumerate(class_names):
            class_data = combined_preds[combined_preds['true_label'] == i]
            if len(class_data) > 0:
                accuracy = (class_data['true_label'] == class_data['predicted_label']).mean()
                class_metrics.loc[class_name, 'Accuracy'] = accuracy
                class_metrics.loc[class_name, 'Samples'] = len(class_data)
            else:
                class_metrics.loc[class_name, 'Accuracy'] = np.nan
                class_metrics.loc[class_name, 'Samples'] = 0
        
        # Save per-class metrics
        class_metrics.to_csv(f"{args.checkpoint_dir}/per_class_metrics.csv")
        
        print("\nPer-class metrics:")
        print(class_metrics)
        
        print(f"Results saved to {args.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train GREEN model on CAUEEG2 dataset')
    
    # Data paths
    parser.add_argument('--feature-path', type=str, default='/path/to/CAUEEG2/Feature',
                        help='Path to feature directory')
    parser.add_argument('--label-path', type=str, default='/path/to/CAUEEG2/Label',
                        help='Path to label directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/caueeg2',
                        help='Directory to save checkpoints')
    
    # Dataset parameters
    parser.add_argument('--max-epochs', type=int, default=10,
                        help='Maximum number of epochs per subject')
    parser.add_argument('--pad-to-length', type=int, default=None,
                        help='Pad/truncate time series to this length')
    
    # Model parameters
    parser.add_argument('--n-freqs', type=int, default=8,
                        help='Number of frequency bands')
    parser.add_argument('--kernel-width', type=float, default=0.5,
                        help='Width of wavelet kernels in seconds')
    parser.add_argument('--sampling-freq', type=float, default=250,
                        help='Sampling frequency of the data')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 32],
                        help='Dimensions of hidden layers')
    parser.add_argument('--bimap-dims', type=int, nargs='+', default=[16],
                        help='Dimensions of BiMap layers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of worker processes for data loading')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Train model
    train_model(args)


if __name__ == '__main__':
    main()