#!/usr/bin/env python
"""
Script to test if synthetic EEG data is too easy to classify compared to genuine data.
This script:
1. Loads both genuine and synthetic EEG data
2. Applies simple classification models to both datasets
3. Compares performance metrics between the two
4. Visualizes feature separation using dimensionality reduction
5. Computes statistics to measure variability in the data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
import mne
import argparse

# Class name mapping
CLASS_NAMES = {
    0: "HC (+SMC)",
    1: "MCI",
    2: "Dementia"
}

def parse_args():
    parser = argparse.ArgumentParser(description='Test classification difficulty of synthetic EEG data')
    
    parser.add_argument('--genuine_dir', type=str, required=True,
                        help='Path to genuine CAUEEG2 dataset directory')
    parser.add_argument('--synthetic_dir', type=str, required=True,
                        help='Path to synthetic EEG dataset directory')
    parser.add_argument('--output_dir', type=str, default='eeg_analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)

def load_caueeg_data(feature_path, label_path, ch_names=None):
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
    
    # Display label structure
    print(f"Label array shape: {labels_array.shape}")
    print(f"Label array data type: {labels_array.dtype}")
    print(f"First few rows of label array:")
    print(labels_array[:5])
    
    # Get mapping between subject_id and label
    subject_labels = {int(entry[1]): int(entry[0]) for entry in labels_array}
    
    # Get feature files
    feature_files = [f for f in os.listdir(feature_path) if f.startswith('feature_') and f.endswith('.npy')]
    feature_files.sort()
    
    print(f"Found {len(feature_files)} feature files")
    
    epochs_list = []
    labels_list = []
    subjects_list = []
    
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
        else:
            data = feature_data
        
        # Handle slicing for synthetic data (36 points on each side)
        if data.shape[2] == 928:  # Already sliced data
            pass
        elif data.shape[2] == 1000:  # Unsliced data
            data = data[:, :, 36:-36]  # Apply slicing
        
        # Create info object - make sure to only use available channels
        n_channels = data.shape[1]
        used_ch_names = ch_names[:n_channels] if n_channels <= len(ch_names) else [f"ch{i}" for i in range(n_channels)]
        
        info = mne.create_info(ch_names=used_ch_names, sfreq=200.0, ch_types='eeg')
        
        # Create epochs object
        epochs = mne.EpochsArray(data, info)
        
        # Get label
        label = subject_labels[subject_id]
        
        epochs_list.append(epochs)
        labels_list.append(label)
        subjects_list.append(subject_id)
    
    print(f"Loaded {len(epochs_list)} subjects with labels")
    
    # Print class distribution
    classes, counts = np.unique(labels_list, return_counts=True)
    print(f"Class distribution:")
    for c, count in zip(classes, counts):
        print(f"  {CLASS_NAMES[c]}: {count} subjects")
    
    return epochs_list, labels_list, subjects_list

def create_features_for_classification(epochs_list, labels_list, subjects_list):
    """
    Extract features from epochs for classification with handling for varying shapes
    """
    # Initialize lists to store features
    features_list = []
    final_labels = []
    final_subjects = []
    
    # Extract features from each subject's epochs
    for i, epochs in enumerate(epochs_list):
        try:
            # Get raw data
            data = epochs.get_data()
            
            # Basic features - use mean across epochs to ensure consistent shape
            # Mean power across time per channel (average across all epochs)
            mean_power = np.mean(np.mean(data**2, axis=2), axis=0)
            
            # Variance of signal per channel (average across all epochs)
            var_signal = np.mean(np.var(data, axis=2), axis=0)
            
            # Mean absolute value per channel (average across all epochs)
            mean_abs = np.mean(np.mean(np.abs(data), axis=2), axis=0)
            
            # Combine features into a fixed-length vector
            subject_features = np.concatenate([mean_power, var_signal, mean_abs])
            
            # Add to list
            features_list.append(subject_features)
            final_labels.append(labels_list[i])
            final_subjects.append(subjects_list[i])
            
        except Exception as e:
            print(f"Error processing subject {i}: {e}")
            continue
    
    # Convert to numpy array
    X = np.vstack(features_list)  # Stack vertically to handle variable length features
    y = np.array(final_labels)
    subjects = np.array(final_subjects)
    
    # Print diagnostic info
    print(f"Created features with shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Create DataFrame
    features_df = pd.DataFrame(X)
    features_df['label'] = y
    features_df['subject_id'] = subjects
    
    return features_df

def create_subject_based_splits(features_df, test_size=0.2, seed=42):
    """
    Create train/test splits based on subjects
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features, labels, and subject IDs
    test_size : float, optional
        Fraction of subjects to use for testing, by default 0.2
    seed : int, optional
        Random seed, by default 42
    
    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
        Train and test splits
    """
    # Get unique subjects
    unique_subjects = features_df['subject_id'].unique()
    
    # Split subjects
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=test_size, random_state=seed, stratify=None
    )
    
    # Create masks
    train_mask = features_df['subject_id'].isin(train_subjects)
    test_mask = features_df['subject_id'].isin(test_subjects)
    
    # Create splits
    X_train = features_df.loc[train_mask].drop(['label', 'subject_id'], axis=1).values
    X_test = features_df.loc[test_mask].drop(['label', 'subject_id'], axis=1).values
    y_train = features_df.loc[train_mask, 'label'].values
    y_test = features_df.loc[test_mask, 'label'].values
    
    print(f"Train set: {len(X_train)} samples, {len(set(features_df.loc[train_mask, 'subject_id']))} subjects")
    print(f"Test set: {len(X_test)} samples, {len(set(features_df.loc[test_mask, 'subject_id']))} subjects")
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_name):
    """
    Train and evaluate multiple models
    
    Parameters
    ----------
    X_train, X_test, y_train, y_test : numpy arrays
        Train and test splits
    dataset_name : str
        Name of the dataset (genuine or synthetic)
    
    Returns
    -------
    results : dict
        Dictionary with model names and performance metrics
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Results dictionary
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name} on {dataset_name} dataset...")
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'predictions': y_pred,
            'model': model
        }
        
        # Print results
        print(f"{name} on {dataset_name} dataset:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=[CLASS_NAMES[i] for i in sorted(set(y_test))]))
        print()
    
    return results

def visualize_feature_space(features_df, output_dir, dataset_name):
    """
    Visualize features using dimensionality reduction
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features and labels
    output_dir : str
        Directory to save plots
    dataset_name : str
        Name of the dataset (genuine or synthetic)
    """
    # Extract features and labels
    X = features_df.drop(['label', 'subject_id'], axis=1).values
    y = features_df['label'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot PCA
    for label in sorted(set(y)):
        mask = y == label
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=CLASS_NAMES[label], alpha=0.7)
    
    axes[0].set_title(f'PCA - {dataset_name} Dataset')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot t-SNE
    for label in sorted(set(y)):
        mask = y == label
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=CLASS_NAMES[label], alpha=0.7)
    
    axes[1].set_title(f't-SNE - {dataset_name} Dataset')
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_feature_visualization.png'), dpi=300)
    plt.close()
    
    # Calculate and return explained variance from PCA
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    plt.grid(alpha=0.3)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Explained Variance - {dataset_name} Dataset')
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_explained_variance.png'), dpi=300)
    plt.close()
    
    return pca_full.explained_variance_ratio_

def compute_data_statistics(epochs_list, dataset_name):
    """
    Compute statistics on the EEG data
    
    Parameters
    ----------
    epochs_list : list
        List of mne.Epochs objects
    dataset_name : str
        Name of the dataset (genuine or synthetic)
    
    Returns
    -------
    stats : dict
        Dictionary with various statistics
    """
    # Initialize statistics dictionary
    stats = {}
    
    # Combine all epochs data
    all_data = np.vstack([epochs.get_data() for epochs in epochs_list])
    
    # Basic statistics
    stats['mean'] = np.mean(all_data)
    stats['std'] = np.std(all_data)
    stats['min'] = np.min(all_data)
    stats['max'] = np.max(all_data)
    stats['range'] = stats['max'] - stats['min']
    
    # Inter-channel correlation
    n_channels = all_data.shape[1]
    channel_means = np.mean(all_data, axis=2)  # Average across time
    correlation_matrix = np.corrcoef(channel_means.T)
    stats['mean_channel_correlation'] = np.mean(np.abs(correlation_matrix - np.eye(n_channels)))
    
    # Variance distribution
    variances = np.var(all_data, axis=2)  # Variance across time for each epoch/channel
    stats['variance_mean'] = np.mean(variances)
    stats['variance_std'] = np.std(variances)
    
    # Print statistics
    print(f"\nStatistics for {dataset_name} dataset:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return stats

def compare_performance(genuine_results, synthetic_results, output_dir):
    """
    Compare model performance between genuine and synthetic datasets
    
    Parameters
    ----------
    genuine_results : dict
        Results dictionary for genuine dataset
    synthetic_results : dict
        Results dictionary for synthetic dataset
    output_dir : str
        Directory to save plots
    """
    # Extract accuracies
    models = list(genuine_results.keys())
    genuine_acc = [genuine_results[model]['accuracy'] for model in models]
    synthetic_acc = [synthetic_results[model]['accuracy'] for model in models]
    genuine_balanced = [genuine_results[model]['balanced_accuracy'] for model in models]
    synthetic_balanced = [synthetic_results[model]['balanced_accuracy'] for model in models]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': models + models,
        'Dataset': ['Genuine'] * len(models) + ['Synthetic'] * len(models),
        'Accuracy': genuine_acc + synthetic_acc,
        'Balanced Accuracy': genuine_balanced + synthetic_balanced
    })
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='Accuracy', hue='Dataset', data=df)
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15)
    
    # Add text labels
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.01, f'{height:.3f}', 
                ha="center", va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
    plt.close()
    
    # Plot balanced accuracy comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='Balanced Accuracy', hue='Dataset', data=df)
    plt.title('Model Balanced Accuracy Comparison')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15)
    
    # Add text labels
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.01, f'{height:.3f}', 
                ha="center", va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'balanced_accuracy_comparison.png'), dpi=300)
    plt.close()
    
    # Create summary table
    summary = pd.DataFrame({
        'Model': models,
        'Genuine Accuracy': genuine_acc,
        'Synthetic Accuracy': synthetic_acc,
        'Accuracy Difference': [s - g for s, g in zip(synthetic_acc, genuine_acc)],
        'Genuine Balanced Acc': genuine_balanced,
        'Synthetic Balanced Acc': synthetic_balanced,
        'Balanced Acc Difference': [s - g for s, g in zip(synthetic_balanced, genuine_balanced)]
    })
    
    # Save summary table
    summary.to_csv(os.path.join(output_dir, 'performance_comparison.csv'), index=False)
    
    # Print summary
    print("\nPerformance Summary:")
    print(summary)
    
    return summary

def compare_statistics(genuine_stats, synthetic_stats, output_dir):
    """
    Compare statistics between genuine and synthetic datasets
    
    Parameters
    ----------
    genuine_stats : dict
        Statistics dictionary for genuine dataset
    synthetic_stats : dict
        Statistics dictionary for synthetic dataset
    output_dir : str
        Directory to save plots
    """
    # Create comparison table
    comparison = pd.DataFrame({
        'Statistic': list(genuine_stats.keys()),
        'Genuine': list(genuine_stats.values()),
        'Synthetic': list(synthetic_stats.values()),
        'Ratio (Synthetic/Genuine)': [synthetic_stats[k]/genuine_stats[k] if genuine_stats[k] != 0 else float('inf') 
                                       for k in genuine_stats.keys()]
    })
    
    # Save comparison table
    comparison.to_csv(os.path.join(output_dir, 'statistics_comparison.csv'), index=False)
    
    # Print comparison
    print("\nStatistics Comparison:")
    print(comparison)
    
    return comparison

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define paths
    genuine_feature_path = os.path.join(args.genuine_dir, 'Feature')
    genuine_label_path = os.path.join(args.genuine_dir, 'Label')
    synthetic_feature_path = os.path.join(args.synthetic_dir, 'Feature')
    synthetic_label_path = os.path.join(args.synthetic_dir, 'Label')
    
    # Load genuine data
    print("\n===== Loading Genuine Data =====")
    genuine_epochs, genuine_labels, genuine_subjects = load_caueeg_data(
        genuine_feature_path, genuine_label_path
    )
    
    # Load synthetic data
    print("\n===== Loading Synthetic Data =====")
    synthetic_epochs, synthetic_labels, synthetic_subjects = load_caueeg_data(
        synthetic_feature_path, synthetic_label_path
    )
    
    # Create features for classification
    print("\n===== Creating Features =====")
    genuine_features = create_features_for_classification(genuine_epochs, genuine_labels, genuine_subjects)
    synthetic_features = create_features_for_classification(synthetic_epochs, synthetic_labels, synthetic_subjects)
    
    # Visualize feature spaces
    print("\n===== Visualizing Feature Spaces =====")
    genuine_explained_var = visualize_feature_space(genuine_features, args.output_dir, 'Genuine')
    synthetic_explained_var = visualize_feature_space(synthetic_features, args.output_dir, 'Synthetic')
    
    # Compute data statistics
    print("\n===== Computing Data Statistics =====")
    genuine_stats = compute_data_statistics(genuine_epochs, 'Genuine')
    synthetic_stats = compute_data_statistics(synthetic_epochs, 'Synthetic')
    
    # Compare statistics
    statistics_comparison = compare_statistics(genuine_stats, synthetic_stats, args.output_dir)
    
    # Create subject-based splits
    print("\n===== Creating Subject-Based Splits =====")
    print("Genuine Data:")
    genuine_X_train, genuine_X_test, genuine_y_train, genuine_y_test = create_subject_based_splits(
        genuine_features, test_size=0.2, seed=args.seed
    )
    
    print("\nSynthetic Data:")
    synthetic_X_train, synthetic_X_test, synthetic_y_train, synthetic_y_test = create_subject_based_splits(
        synthetic_features, test_size=0.2, seed=args.seed
    )
    
    # Train and evaluate models
    print("\n===== Training and Evaluating Models =====")
    genuine_results = train_and_evaluate_models(
        genuine_X_train, genuine_X_test, genuine_y_train, genuine_y_test, 'Genuine'
    )
    
    synthetic_results = train_and_evaluate_models(
        synthetic_X_train, synthetic_X_test, synthetic_y_train, synthetic_y_test, 'Synthetic'
    )
    
    # Compare performance
    print("\n===== Comparing Performance =====")
    performance_comparison = compare_performance(genuine_results, synthetic_results, args.output_dir)
    
    # Save PCA explained variance comparison
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(genuine_explained_var), marker='o', label='Genuine')
    plt.plot(np.cumsum(synthetic_explained_var), marker='s', label='Synthetic')
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    plt.grid(alpha=0.3)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance Comparison')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'explained_variance_comparison.png'), dpi=300)
    plt.close()
    
    print(f"\nAll results saved to {args.output_dir}")
    
    # Print interpretation
    print("\n===== Interpretation =====")
    max_synthetic_acc = max([synthetic_results[model]['accuracy'] for model in synthetic_results.keys()])
    max_genuine_acc = max([genuine_results[model]['accuracy'] for model in genuine_results.keys()])
    
    if max_synthetic_acc > 0.95 and max_synthetic_acc - max_genuine_acc > 0.2:
        print("CONCLUSION: The synthetic data appears to be TOO EASY to classify compared to genuine data.")
        print("This suggests that your synthetic EEG data might lack the natural variability and complexity")
        print("of real EEG signals, making the classification task artificially simple.")
        
        # Suggestions
        print("\nSuggestions:")
        print("1. Add more noise or variability to your generative model")
        print("2. Ensure your synthetic data preserves the intra-class variability of real data")
        print("3. Check for data leakage or artifacts in your synthetic data generation process")
        print("4. Try creating synthetic data independently for each class rather than transforming existing data")
    elif max_synthetic_acc > 0.9 and max_genuine_acc > 0.9:
        print("CONCLUSION: Both synthetic and genuine data yield very high classification accuracy.")
        print("This might indicate that the task itself is relatively easy, or that both datasets")
        print("contain strong separable features for classification.")
    elif abs(max_synthetic_acc - max_genuine_acc) < 0.1:
        print("CONCLUSION: The synthetic data has similar classification difficulty to genuine data.")
        print("This is a good sign that your synthetic data is capturing the essential patterns")
        print("and variability of real EEG signals.")
    else:
        print(f"CONCLUSION: The synthetic data shows different classification patterns than genuine data.")
        print(f"Max accuracy on synthetic data: {max_synthetic_acc:.4f}")
        print(f"Max accuracy on genuine data: {max_genuine_acc:.4f}")
        print("Examine the visualizations and statistics to understand these differences.")

if __name__ == "__main__":
    main()