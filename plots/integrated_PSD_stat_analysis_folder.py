import numpy as np
import mne
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.ndimage import label
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(path, sfreq=200):
    """Load and preprocess EEG data from .npy file"""
    data = np.load(path)
    print(f"Original data shape: {data.shape}")
    
    # Handle data shape - ensure (n_epochs, n_channels, n_times)
    if data.shape[2] == 19:  # Assuming 19 channels
        data = data.transpose(0, 2, 1) 
        print(f"Transposed data shape: {data.shape}")
    
    n_epochs, n_channels, n_times = data.shape
    
    # Check for problematic values
    print(f"NaN values: {np.isnan(data).any()}")
    print(f"Inf values: {np.isinf(data).any()}")
    print(f"Data range: {np.min(data)} to {np.max(data)}")
    
    # Fix data if needed
    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data, nan=1e-6, posinf=1.0, neginf=-1.0)
        print("Replaced NaN/Inf values")
    
    # Add tiny jitter if data is all zeros
    if np.allclose(data, 0, atol=1e-6):
        data = data + np.random.normal(0, 1e-5, data.shape)
        print("Added small jitter to avoid all-zero data")
    
    return data, n_epochs, n_channels, n_times

def calculate_psd_for_statistical_analysis(data, sfreq, fmin=1, fmax=30, nperseg=None):
    """
    Calculate PSD data in format needed for statistical analysis.
    
    Parameters:
    -----------
    data : array, shape (n_epochs, n_channels, n_times)
        EEG data
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Frequency range for analysis
    nperseg : int, optional
        Length of each segment for Welch's method
        
    Returns:
    --------
    psd_data : array, shape (n_epochs, n_channels, n_freqs)
        PSD data for each epoch, channel, and frequency
    freqs : array
        Frequency values
    """
    n_epochs, n_channels, n_times = data.shape
    
    # Set nperseg if not provided
    if nperseg is None:
        nperseg = min(256, n_times)
    
    # Calculate frequencies using first epoch/channel
    freqs, _ = signal.welch(data[0, 0], fs=sfreq, nperseg=nperseg)
    
    # Filter to desired frequency range
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    
    # Initialize PSD array
    psd_data = np.zeros((n_epochs, n_channels, len(freqs)))
    
    print("Calculating PSDs for all epochs and channels...")
    
    # Calculate PSD for each epoch and channel
    for epoch_idx in tqdm(range(n_epochs), desc="Processing epochs"):
        for ch_idx in range(n_channels):
            _, psd = signal.welch(data[epoch_idx, ch_idx], fs=sfreq, nperseg=nperseg)
            psd_data[epoch_idx, ch_idx] = psd[mask]
    
    return psd_data, freqs

class PSDStatisticalComparison:
    def __init__(self, n_permutations=1000, alpha=0.05, cluster_alpha=0.01):
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.cluster_alpha = cluster_alpha
        
    def permutation_test_single(self, real_data, synthetic_data):
        """Perform permutation test for a single frequency-channel combination."""
        observed_diff = np.mean(real_data) - np.mean(synthetic_data)
        
        combined_data = np.concatenate([real_data, synthetic_data])
        n_real = len(real_data)
        
        perm_diffs = []
        for _ in range(self.n_permutations):
            shuffled = np.random.permutation(combined_data)
            perm_real = shuffled[:n_real]
            perm_synthetic = shuffled[n_real:]
            perm_diff = np.mean(perm_real) - np.mean(perm_synthetic)
            perm_diffs.append(perm_diff)
        
        perm_diffs = np.array(perm_diffs)
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        return p_value, observed_diff
    
    def mass_univariate_test(self, real_psds, synthetic_psds, freqs, channels):
        """Perform mass-univariate permutation tests."""
        n_channels, n_freqs = len(channels), len(freqs)
        p_values = np.zeros((n_channels, n_freqs))
        effect_sizes = np.zeros((n_channels, n_freqs))
        
        print("Running mass-univariate permutation tests...")
        
        total_tests = n_channels * n_freqs
        pbar = tqdm(total=total_tests, desc="Testing freq-channel combinations")
        
        for ch_idx in range(n_channels):
            for freq_idx in range(n_freqs):
                real_data = real_psds[:, ch_idx, freq_idx]
                synthetic_data = synthetic_psds[:, ch_idx, freq_idx]
                
                p_val, effect = self.permutation_test_single(real_data, synthetic_data)
                
                p_values[ch_idx, freq_idx] = p_val
                effect_sizes[ch_idx, freq_idx] = effect
                
                pbar.update(1)
        
        pbar.close()
        return p_values, effect_sizes
    
    def cluster_correction(self, p_values, freqs, channels):
        """Apply cluster-based correction for multiple comparisons."""
        uncorrected_sig = p_values < self.cluster_alpha
        labeled_array, num_clusters = label(uncorrected_sig)
        
        significant_clusters = np.zeros_like(p_values, dtype=bool)
        cluster_info = {}
        
        if num_clusters > 0:
            print(f"Found {num_clusters} potential clusters, evaluating significance...")
            
            for cluster_id in range(1, num_clusters + 1):
                cluster_mask = labeled_array == cluster_id
                cluster_size = np.sum(cluster_mask)
                cluster_p_values = p_values[cluster_mask]
                min_p = np.min(cluster_p_values)
                mean_p = np.mean(cluster_p_values)
                
                if cluster_size >= 3 and min_p < self.alpha:
                    significant_clusters[cluster_mask] = True
                    
                    ch_indices, freq_indices = np.where(cluster_mask)
                    freq_range = (freqs[freq_indices.min()], freqs[freq_indices.max()])
                    ch_range = (channels[ch_indices.min()], channels[ch_indices.max()])
                    
                    cluster_info[cluster_id] = {
                        'size': cluster_size,
                        'min_p': min_p,
                        'mean_p': mean_p,
                        'freq_range': freq_range,
                        'channel_range': ch_range,
                        'channels': [channels[i] for i in np.unique(ch_indices)],
                        'freq_indices': np.unique(freq_indices)
                    }
        
        return significant_clusters, cluster_info
    
    def plot_statistical_heatmap(self, p_values, effect_sizes, significant_clusters, 
                               freqs, channels, cluster_info=None, figsize=(15, 10)):
        """Create comprehensive statistical heatmap visualization."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Statistical Comparison: Real vs Synthetic EEG PSDs', fontsize=16, fontweight='bold')
        
        # 1. P-values heatmap
        ax1 = axes[0, 0]
        im1 = ax1.imshow(-np.log10(p_values), aspect='auto', cmap='hot', interpolation='nearest')
        ax1.set_title('Significance Map\n(-log10 p-values)')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Channels')
        
        # Set frequency ticks
        freq_ticks = np.linspace(0, len(freqs)-1, 6).astype(int)
        ax1.set_xticks(freq_ticks)
        ax1.set_xticklabels([f'{freqs[i]:.1f}' for i in freq_ticks])
        
        # Set channel ticks
        ax1.set_yticks(range(len(channels)))
        ax1.set_yticklabels(channels)
        
        plt.colorbar(im1, ax=ax1, label='-log10(p)')
        
        # Add significance threshold line
        sig_threshold = -np.log10(self.alpha)
        ax1.contour(-np.log10(p_values), levels=[sig_threshold], colors='white', linewidths=2)
        
        # 2. Effect sizes heatmap
        ax2 = axes[0, 1]
        im2 = ax2.imshow(effect_sizes, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        ax2.set_title('Effect Sizes\n(Real - Synthetic)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Channels')
        
        ax2.set_xticks(freq_ticks)
        ax2.set_xticklabels([f'{freqs[i]:.1f}' for i in freq_ticks])
        ax2.set_yticks(range(len(channels)))
        ax2.set_yticklabels(channels)
        
        plt.colorbar(im2, ax=ax2, label='Power Difference')
        
        # 3. Significant clusters
        ax3 = axes[1, 0]
        im3 = ax3.imshow(significant_clusters.astype(int), aspect='auto', cmap='Reds', interpolation='nearest')
        ax3.set_title('Significant Clusters\n(Cluster-corrected)')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Channels')
        
        ax3.set_xticks(freq_ticks)
        ax3.set_xticklabels([f'{freqs[i]:.1f}' for i in freq_ticks])
        ax3.set_yticks(range(len(channels)))
        ax3.set_yticklabels(channels)
        
        plt.colorbar(im3, ax=ax3, label='Significant')
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        n_significant = np.sum(p_values < self.alpha)
        n_total = p_values.size
        percent_sig = (n_significant / n_total) * 100
        n_clusters = len(cluster_info) if cluster_info else 0
        
        summary_text = f"""Statistical Summary:

Total tests: {n_total:,}
Significant (uncorrected): {n_significant:,} ({percent_sig:.1f}%)
Significance level: α = {self.alpha}
Permutations: {self.n_permutations:,}

Cluster Analysis:
Significant clusters: {n_clusters}
Cluster threshold: α = {self.cluster_alpha}

Effect Size Range:
Min: {np.min(effect_sizes):.4f}
Max: {np.max(effect_sizes):.4f}
Mean: {np.mean(effect_sizes):.4f}
"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def print_cluster_results(self, cluster_info, freqs):
        """Print detailed cluster analysis results."""
        if not cluster_info:
            print("No significant clusters found.")
            return
            
        print(f"\n{'='*60}")
        print(f"SIGNIFICANT CLUSTERS ANALYSIS")
        print(f"{'='*60}")
        
        for cluster_id, info in cluster_info.items():
            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {info['size']} frequency-channel combinations")
            print(f"  Minimum p-value: {info['min_p']:.6f}")
            print(f"  Mean p-value: {info['mean_p']:.6f}")
            print(f"  Frequency range: {info['freq_range'][0]:.1f} - {info['freq_range'][1]:.1f} Hz")
            print(f"  Channels involved: {', '.join(info['channels'])}")
            
            # Determine frequency band
            freq_range = info['freq_range']
            if freq_range[0] >= 0.5 and freq_range[1] <= 4:
                band = "Delta"
            elif freq_range[0] >= 4 and freq_range[1] <= 8:
                band = "Theta"  
            elif freq_range[0] >= 8 and freq_range[1] <= 12:
                band = "Alpha"
            elif freq_range[0] >= 12 and freq_range[1] <= 30:
                band = "Beta"
            elif freq_range[0] >= 30:
                band = "Gamma"
            else:
                band = "Mixed"
            
            print(f"  Frequency band: {band}")
            print(f"  {'-'*40}")

def load_labels_with_subject_ids(dataset_folder):
    """
    Load labels and extract both condition labels and subject IDs.
    
    Parameters:
    -----------
    dataset_folder : str
        Path to dataset folder (e.g., 'dataset/CAUEEG2')
        
    Returns:
    --------
    labels_array : array, shape (n_samples, 2) 
        Array where first column is condition label, second column is subject ID
    condition_labels : array
        Condition labels only
    subject_ids : array
        Subject IDs only
    """
    label_path = os.path.join(dataset_folder, 'Label', 'label.npy')
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    labels_array = np.load(label_path)
    print(f"Loaded labels shape: {labels_array.shape}")
    
    # Extract condition labels and subject IDs
    if labels_array.ndim == 2 and labels_array.shape[1] == 2:
        condition_labels = labels_array[:, 0]
        subject_ids = labels_array[:, 1]
    elif labels_array.ndim == 1:
        # If only one column, assume these are condition labels and generate subject IDs
        condition_labels = labels_array
        subject_ids = np.arange(1, len(labels_array) + 1)
        print("Warning: Only condition labels found, generating sequential subject IDs")
    else:
        raise ValueError(f"Unexpected label array shape: {labels_array.shape}")
    
    print(f"Unique condition labels: {np.unique(condition_labels)}")
    print(f"Subject ID range: {np.min(subject_ids)} to {np.max(subject_ids)}")
    
    return labels_array, condition_labels, subject_ids

def find_matching_subjects_by_condition(real_dataset_folder, synthetic_dataset_folder, 
                                       conditions=['HC', 'MCI', 'Dementia']):
    """
    Find matching subjects between real and synthetic datasets for each condition.
    
    Parameters:
    -----------
    real_dataset_folder : str
        Path to real EEG dataset folder
    synthetic_dataset_folder : str  
        Path to synthetic EEG dataset folder
    conditions : list
        List of conditions to find matches for
        
    Returns:
    --------
    matched_samples : dict
        Dictionary with condition names as keys and dictionaries as values.
        Each sub-dictionary contains 'real_path', 'synthetic_path', 'subject_id'
    """
    
    # Map label values to condition names (adjust these mappings as needed)
    label_mapping = {
        0: 'HC',        # Healthy Control
        1: 'MCI',       # Mild Cognitive Impairment  
        2: 'Dementia'   # Dementia
    }
    
    print("Loading real dataset labels...")
    real_labels_array, real_condition_labels, real_subject_ids = load_labels_with_subject_ids(real_dataset_folder)
    
    print("Loading synthetic dataset labels...")
    synth_labels_array, synth_condition_labels, synth_subject_ids = load_labels_with_subject_ids(synthetic_dataset_folder)
    
    matched_samples = {}
    
    for condition in conditions:
        print(f"\nFinding matches for condition: {condition}")
        
        # Find the label value for this condition
        label_value = None
        for val, cond in label_mapping.items():
            if cond == condition:
                label_value = val
                break
        
        if label_value is None:
            print(f"Warning: Condition '{condition}' not found in label mapping")
            continue
        
        # Find subjects with this condition in real dataset
        real_condition_mask = real_condition_labels == label_value
        real_subjects_with_condition = real_subject_ids[real_condition_mask]
        real_indices_with_condition = np.where(real_condition_mask)[0]
        
        # Find subjects with this condition in synthetic dataset  
        synth_condition_mask = synth_condition_labels == label_value
        synth_subjects_with_condition = synth_subject_ids[synth_condition_mask]
        synth_indices_with_condition = np.where(synth_condition_mask)[0]
        
        print(f"Real dataset: {len(real_subjects_with_condition)} subjects with {condition}")
        print(f"Synthetic dataset: {len(synth_subjects_with_condition)} subjects with {condition}")
        
        # Find common subject IDs
        common_subject_ids = np.intersect1d(real_subjects_with_condition, synth_subjects_with_condition)
        
        if len(common_subject_ids) == 0:
            print(f"Warning: No matching subject IDs found for condition '{condition}'")
            continue
        
        print(f"Found {len(common_subject_ids)} matching subjects: {common_subject_ids}")
        
        # Select the first matching subject (you can modify this to select differently)
        selected_subject_id = common_subject_ids[0]
        
        # Find the indices for this subject in both datasets
        real_idx = np.where((real_condition_labels == label_value) & 
                           (real_subject_ids == selected_subject_id))[0][0]
        synth_idx = np.where((synth_condition_labels == label_value) & 
                            (synth_subject_ids == selected_subject_id))[0][0]
        
        # Generate file paths
        real_feature_file = f"feature_{real_idx+1:02d}.npy"  # 1-indexed filenames
        real_feature_path = os.path.join(real_dataset_folder, 'Feature', real_feature_file)
        
        synth_feature_file = f"feature_{synth_idx+1:02d}.npy"  # 1-indexed filenames
        synth_feature_path = os.path.join(synthetic_dataset_folder, 'Feature', synth_feature_file)
        
        # Verify files exist
        if not os.path.exists(real_feature_path):
            print(f"Warning: Real feature file not found: {real_feature_path}")
            continue
            
        if not os.path.exists(synth_feature_path):
            print(f"Warning: Synthetic feature file not found: {synth_feature_path}")
            continue
        
        matched_samples[condition] = {
            'real_path': real_feature_path,
            'synthetic_path': synth_feature_path,
            'subject_id': selected_subject_id,
            'real_index': real_idx,
            'synthetic_index': synth_idx
        }
        
        print(f"Selected subject {selected_subject_id} for {condition}:")
        print(f"  Real: {real_feature_file} (index {real_idx}, label {real_labels_array[real_idx]})")
        print(f"  Synthetic: {synth_feature_file} (index {synth_idx}, label {synth_labels_array[synth_idx]})")
    
    return matched_samples

def run_complete_psd_analysis(real_dataset_folder, synthetic_dataset_folder, output_dir='statistical_analysis', 
                            conditions=['HC', 'MCI', 'Dementia'], sfreq=200, fmin=1, fmax=30, n_permutations=1000):
    """
    Run complete PSD statistical analysis comparing real and synthetic EEG data for multiple conditions.
    Now matches subjects by subject ID instead of index position.
    
    Parameters:
    -----------
    real_dataset_folder : str
        Path to real EEG dataset folder (e.g., 'dataset/CAUEEG2')
    synthetic_dataset_folder : str  
        Path to synthetic EEG dataset folder
    conditions : list
        List of conditions to analyze (default: ['HC', 'MCI', 'Dementia'])
    output_dir : str
        Directory to save results
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Frequency range for analysis
    n_permutations : int
        Number of permutations for statistical tests
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("FINDING MATCHING SUBJECTS BY CONDITION AND SUBJECT ID")
    print("="*60)
    
    # Find matching subjects between datasets
    matched_samples = find_matching_subjects_by_condition(
        real_dataset_folder, synthetic_dataset_folder, conditions
    )
    
    if not matched_samples:
        raise ValueError("No matching subjects found between real and synthetic datasets")
    
    print(f"\nAnalyzing {len(matched_samples)} conditions with matching subjects")
    
    all_results = {}
    
    # Process each condition
    for condition, sample_info in matched_samples.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING CONDITION: {condition}")
        print(f"SUBJECT ID: {sample_info['subject_id']}")
        print(f"{'='*60}")
        
        real_data_path = sample_info['real_path']
        synthetic_data_path = sample_info['synthetic_path']
        
        print(f"Real data: {real_data_path}")
        print(f"Synthetic data: {synthetic_data_path}")
        print(f"Matching subject ID: {sample_info['subject_id']}")
        
        # Load and preprocess data
        print("Loading real EEG data...")
        real_data, n_epochs_real, n_channels, n_times = load_and_preprocess_data(real_data_path, sfreq)
        
        print("Loading synthetic EEG data...")
        synthetic_data, n_epochs_synth, _, _ = load_and_preprocess_data(synthetic_data_path, sfreq)
        
        # Check data compatibility
        if real_data.shape[1:] != synthetic_data.shape[1:]:
            raise ValueError(f"Data shape mismatch for {condition}: Real {real_data.shape} vs Synthetic {synthetic_data.shape}")
        
        # Use standard 10-20 EEG channel names
        standard_channels = ['FP1', 'F3', 'C3', 'P3', 'O1', 'FP2', 'F4', 'C4', 'P4', 'O2', 
                            'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'FZ', 'CZ', 'PZ']
        
        if n_channels == len(standard_channels):
            channels = standard_channels
        else:
            print(f"Warning: Expected {len(standard_channels)} channels but found {n_channels}")
            print("Using generic channel names instead")
            channels = [f"EEG {i+1}" for i in range(n_channels)]
        
        print("Calculating PSDs for real data...")
        real_psds, freqs = calculate_psd_for_statistical_analysis(real_data, sfreq, fmin, fmax)
        
        print("Calculating PSDs for synthetic data...")
        synthetic_psds, _ = calculate_psd_for_statistical_analysis(synthetic_data, sfreq, fmin, fmax)
        
        print(f"PSD data shapes: Real {real_psds.shape}, Synthetic {synthetic_psds.shape}")
        print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz ({len(freqs)} frequencies)")
        
        # Run statistical comparison
        print("Starting statistical comparison...")
        comparator = PSDStatisticalComparison(n_permutations=n_permutations, alpha=0.05, cluster_alpha=0.01)
        
        # Mass-univariate tests
        p_values, effect_sizes = comparator.mass_univariate_test(real_psds, synthetic_psds, freqs, channels)
        
        # Cluster correction
        significant_clusters, cluster_info = comparator.cluster_correction(p_values, freqs, channels)
        
        # Create condition-specific output directory
        condition_output_dir = os.path.join(output_dir, condition)
        os.makedirs(condition_output_dir, exist_ok=True)
        
        # Create visualization
        fig = comparator.plot_statistical_heatmap(
            p_values, effect_sizes, significant_clusters, freqs, channels, cluster_info
        )
        fig.suptitle(f'Statistical Comparison: Real vs Synthetic EEG PSDs - {condition}\nSubject ID: {sample_info["subject_id"]}', 
                    fontsize=16, fontweight='bold')
        
        # Save figure
        fig_path = os.path.join(condition_output_dir, f'statistical_comparison_heatmap_{condition}_subject_{sample_info["subject_id"]}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved statistical heatmap to {fig_path}")
        
        # Print results
        print(f"\n{condition} CLUSTER RESULTS (Subject {sample_info['subject_id']}):")
        comparator.print_cluster_results(cluster_info, freqs)
        
        # Create traditional PSD comparison plot
        print("Creating traditional PSD comparison plot...")
        fig_psd, ax = plt.subplots(figsize=(12, 8))
        
        # Average PSDs across epochs
        real_psd_avg = np.mean(real_psds, axis=0)
        synthetic_psd_avg = np.mean(synthetic_psds, axis=0)
        
        # Plot average across all channels
        real_grand_avg = np.mean(real_psd_avg, axis=0)
        synthetic_grand_avg = np.mean(synthetic_psd_avg, axis=0)
        
        ax.semilogy(freqs, real_grand_avg, 'b-', linewidth=2, label='Real EEG', alpha=0.8)
        ax.semilogy(freqs, synthetic_grand_avg, 'r-', linewidth=2, label='Synthetic EEG', alpha=0.8)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (µV²/Hz)')
        ax.set_title(f'Power Spectral Density Comparison - {condition}\nSubject ID: {sample_info["subject_id"]} (Grand average across all channels)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        psd_comparison_path = os.path.join(condition_output_dir, f'psd_comparison_plot_{condition}_subject_{sample_info["subject_id"]}.png')
        fig_psd.savefig(psd_comparison_path, dpi=300, bbox_inches='tight')
        print(f"Saved PSD comparison plot to {psd_comparison_path}")
        
        # Save results
        results = {
            'p_values': p_values,
            'effect_sizes': effect_sizes,
            'significant_clusters': significant_clusters,
            'cluster_info': cluster_info,
            'freqs': freqs,
            'channels': channels,
            'real_psds': real_psds,
            'synthetic_psds': synthetic_psds,
            'condition': condition,
            'subject_id': sample_info['subject_id'],
            'sample_info': sample_info
        }
        
        results_path = os.path.join(condition_output_dir, f'statistical_results_{condition}_subject_{sample_info["subject_id"]}.npz')
        np.savez(results_path, **{k: v for k, v in results.items() if not isinstance(v, dict)})
        print(f"Saved statistical results to {results_path}")
        
        all_results[condition] = results
        
        plt.close('all')  # Close figures to save memory
    
    # Create summary comparison across all conditions
    print(f"\n{'='*60}")
    print("CREATING SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    create_summary_comparison(all_results, output_dir)
    
    print(f"\nAnalysis complete! Check the '{output_dir}' directory for outputs.")
    print(f"Individual condition results are in subdirectories: {list(all_results.keys())}")
    
    return all_results

def create_summary_comparison(all_results, output_dir):
    """Create a summary comparison plot across all conditions."""
    
    if not all_results:
        print("No results to summarize")
        return
    
    # Create summary plot
    fig, axes = plt.subplots(len(all_results), 2, figsize=(15, 5*len(all_results)))
    if len(all_results) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Summary: PSD Comparisons Across Conditions (Matched Subjects)', fontsize=16, fontweight='bold')
    
    for idx, (condition, results) in enumerate(all_results.items()):
        freqs = results['freqs']
        real_psds = results['real_psds']
        synthetic_psds = results['synthetic_psds']
        significant_clusters = results['significant_clusters']
        subject_id = results['subject_id']
        
        # PSD comparison
        ax1 = axes[idx, 0]
        real_psd_avg = np.mean(real_psds, axis=(0, 1))  # Average across epochs and channels
        synthetic_psd_avg = np.mean(synthetic_psds, axis=(0, 1))
        
        ax1.semilogy(freqs, real_psd_avg, 'b-', linewidth=2, label='Real EEG', alpha=0.8)
        ax1.semilogy(freqs, synthetic_psd_avg, 'r-', linewidth=2, label='Synthetic EEG', alpha=0.8)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('PSD (µV²/Hz)')
        ax1.set_title(f'{condition} - Subject {subject_id}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Significance heatmap
        ax2 = axes[idx, 1]
        im = ax2.imshow(significant_clusters.astype(int), aspect='auto', cmap='Reds', interpolation='nearest')
        ax2.set_title(f'{condition} - Significant Clusters')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Channels')
        
        # Set frequency ticks
        freq_ticks = np.linspace(0, len(freqs)-1, 6).astype(int)
        ax2.set_xticks(freq_ticks)
        ax2.set_xticklabels([f'{freqs[i]:.1f}' for i in freq_ticks])
        
        # Set channel ticks (show fewer for readability)
        channels = results['channels']
        if len(channels) > 10:
            ch_ticks = np.linspace(0, len(channels)-1, 6).astype(int)
            ax2.set_yticks(ch_ticks)
            ax2.set_yticklabels([channels[i] for i in ch_ticks])
        else:
            ax2.set_yticks(range(len(channels)))
            ax2.set_yticklabels(channels)
        
        plt.colorbar(im, ax=ax2, label='Significant')
    
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, 'summary_comparison_matched_subjects.png')
    fig.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary comparison to {summary_path}")
    
    plt.close()

# Example usage
if __name__ == "__main__":
    # Example paths - replace with your actual file paths
    real_data_path = 'dataset/CAUEEG2'
    synthetic_data_path = 'dataset/SYNTH-CAUEEG2-NORMALIZED'
    
    # Run complete analysis
    results = run_complete_psd_analysis(
        real_dataset_folder=real_data_path,
        synthetic_dataset_folder=synthetic_data_path,
        conditions=['HC', 'MCI', 'Dementia'],
        output_dir='images/statistical_analysis_matched_subjects_SYNTH-CAUEEG2-NORMALIZED'
    )
    
    print("\nAnalysis complete! Check the output directory for results.")