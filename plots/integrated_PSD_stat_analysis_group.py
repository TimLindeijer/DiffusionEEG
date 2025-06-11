import numpy as np
import mne
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.ndimage import label
from scipy.stats import ttest_ind
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

def create_beautiful_psd_plot(freqs, group_mean, group_std, title, color='blue', figsize=(12, 6), normalized=True, norm_method='robust'):
    """
    Create a beautiful PSD plot in the style of the provided image.
    
    Parameters:
    -----------
    freqs : array
        Frequency values
    group_mean : array
        Mean PSD values
    group_std : array
        Standard deviation of PSD values
    title : str
        Plot title
    color : str
        Color for the plot
    figsize : tuple
        Figure size
    normalized : bool
        Whether the data is normalized to 0-1 scale
    norm_method : str
        Normalization method used
        
    Returns:
    --------
    fig : matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create shaded area for variability
    ax.fill_between(freqs, group_mean - group_std, group_mean + group_std, 
                   color=color, alpha=0.3, label='± 1 SD')
    
    # Bold center line for mean
    ax.plot(freqs, group_mean, color=color, linewidth=3, label='Mean', alpha=0.9)
    
    # Formatting to match the provided image
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    if normalized:
        ax.set_ylabel(f'Normalized Power (0-1, {norm_method})', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
    else:
        ax.set_ylabel('Power (dB µV²/Hz)', fontsize=12, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Clean appearance like the provided image
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Set frequency range
    ax.set_xlim(freqs[0], freqs[-1])
    
    plt.tight_layout()
    return fig

def normalize_psd_to_01(psd_data, method='robust'):
    """
    Normalize PSD data to 0-1 scale using different methods.
    
    Parameters:
    -----------
    psd_data : array, shape (n_epochs, n_channels, n_freqs)
        PSD data
    method : str
        Normalization method: 'global', 'robust', or 'percentile'
        
    Returns:
    --------
    normalized_psd : array, same shape as input
        PSD data normalized to 0-1 range
    """
    if method == 'global':
        # Original global min-max normalization
        global_min = np.min(psd_data)
        global_max = np.max(psd_data)
        normalized_psd = (psd_data - global_min) / (global_max - global_min)
        print(f"Global normalization: {global_min:.6f} to {global_max:.6f} -> 0.0 to 1.0")
        
    elif method == 'robust':
        # Use 5th and 95th percentiles to avoid outlier compression
        p5 = np.percentile(psd_data, 5)
        p95 = np.percentile(psd_data, 95)
        normalized_psd = (psd_data - p5) / (p95 - p5)
        # Clip to 0-1 range to handle values outside percentile range
        normalized_psd = np.clip(normalized_psd, 0, 1)
        print(f"Robust normalization: P5={p5:.6f} to P95={p95:.6f} -> 0.0 to 1.0")
        
    elif method == 'percentile':
        # Use 10th and 90th percentiles for even more robust normalization
        p10 = np.percentile(psd_data, 10)
        p90 = np.percentile(psd_data, 90)
        normalized_psd = (psd_data - p10) / (p90 - p10)
        # Clip to 0-1 range
        normalized_psd = np.clip(normalized_psd, 0, 1)
        print(f"Percentile normalization: P10={p10:.6f} to P90={p90:.6f} -> 0.0 to 1.0")
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_psd

def calculate_psd_for_statistical_analysis(data, sfreq, fmin=1, fmax=30, nperseg=None, normalize=True, norm_method='robust'):
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
    normalize : bool
        Whether to normalize PSDs to 0-1 scale
    norm_method : str
        Normalization method: 'global', 'robust', or 'percentile'
        
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
    
    # Normalize PSDs to 0-1 scale if requested
    if normalize:
        psd_data = normalize_psd_to_01(psd_data, method=norm_method)
    
    return psd_data, freqs

class GroupPSDStatisticalComparison:
    def __init__(self, n_permutations=1000, alpha=0.05, cluster_alpha=0.01):
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.cluster_alpha = cluster_alpha
        
    def permutation_test_groups(self, group1_data, group2_data):
        """
        Perform permutation test comparing two groups.
        
        Parameters:
        -----------
        group1_data : array, shape (n_subjects_group1, n_channels, n_freqs)
        group2_data : array, shape (n_subjects_group2, n_channels, n_freqs)
        
        Returns:
        --------
        p_values : array, shape (n_channels, n_freqs)
        effect_sizes : array, shape (n_channels, n_freqs)
        """
        n_channels, n_freqs = group1_data.shape[1], group1_data.shape[2]
        p_values = np.zeros((n_channels, n_freqs))
        effect_sizes = np.zeros((n_channels, n_freqs))
        
        print("Running group-level permutation tests...")
        
        total_tests = n_channels * n_freqs
        pbar = tqdm(total=total_tests, desc="Testing freq-channel combinations")
        
        for ch_idx in range(n_channels):
            for freq_idx in range(n_freqs):
                # Extract data for this channel-frequency combination
                data1 = group1_data[:, ch_idx, freq_idx]  # All subjects in group 1
                data2 = group2_data[:, ch_idx, freq_idx]  # All subjects in group 2
                
                # Calculate observed difference
                observed_diff = np.mean(data1) - np.mean(data2)
                
                # Combine data for permutation
                combined_data = np.concatenate([data1, data2])
                n1 = len(data1)
                
                # Permutation test
                perm_diffs = []
                for _ in range(self.n_permutations):
                    shuffled = np.random.permutation(combined_data)
                    perm_group1 = shuffled[:n1]
                    perm_group2 = shuffled[n1:]
                    perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
                    perm_diffs.append(perm_diff)
                
                perm_diffs = np.array(perm_diffs)
                p_val = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
                
                p_values[ch_idx, freq_idx] = p_val
                effect_sizes[ch_idx, freq_idx] = observed_diff
                
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
                               freqs, channels, cluster_info=None, condition="", figsize=(15, 10)):
        """Create comprehensive statistical heatmap visualization with improved effect size scaling."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Group Statistical Comparison: Real vs Synthetic EEG PSDs - {condition}', 
                    fontsize=16, fontweight='bold')
        
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
        
        # 2. Effect sizes heatmap - CENTERED AROUND 0
        ax2 = axes[0, 1]
        
        # Center the colormap around 0
        max_abs_effect = np.max(np.abs(effect_sizes))
        vmin, vmax = -max_abs_effect, max_abs_effect
        
        im2 = ax2.imshow(effect_sizes, aspect='auto', cmap='RdBu_r', 
                        interpolation='nearest', vmin=vmin, vmax=vmax)
        ax2.set_title(f'Effect Sizes (Centered at 0)\n(Real - Synthetic)')
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
Abs Max: {max_abs_effect:.4f}

Note: PSDs normalized using robust method
Effect sizes centered at 0
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

def load_all_subjects_by_condition(real_dataset_folder, synthetic_dataset_folder, 
                                 conditions=['HC', 'MCI', 'Dementia']):
    """
    Load ALL subjects for each condition from both real and synthetic datasets.
    
    Parameters:
    -----------
    real_dataset_folder : str
        Path to real EEG dataset folder
    synthetic_dataset_folder : str  
        Path to synthetic EEG dataset folder
    conditions : list
        List of conditions to load
        
    Returns:
    --------
    grouped_samples : dict
        Dictionary with condition names as keys and dictionaries as values.
        Each sub-dictionary contains 'real_paths', 'synthetic_paths', 'real_indices', 'synthetic_indices'
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
    
    grouped_samples = {}
    
    for condition in conditions:
        print(f"\nLoading all subjects for condition: {condition}")
        
        # Find the label value for this condition
        label_value = None
        for val, cond in label_mapping.items():
            if cond == condition:
                label_value = val
                break
        
        if label_value is None:
            print(f"Warning: Condition '{condition}' not found in label mapping")
            continue
        
        # Find ALL subjects with this condition in real dataset
        real_condition_mask = real_condition_labels == label_value
        real_indices_with_condition = np.where(real_condition_mask)[0]
        
        # Find ALL subjects with this condition in synthetic dataset  
        synth_condition_mask = synth_condition_labels == label_value
        synth_indices_with_condition = np.where(synth_condition_mask)[0]
        
        print(f"Real dataset: {len(real_indices_with_condition)} subjects with {condition}")
        print(f"Synthetic dataset: {len(synth_indices_with_condition)} subjects with {condition}")
        
        # Generate file paths for ALL subjects
        real_feature_paths = []
        synth_feature_paths = []
        
        for idx in real_indices_with_condition:
            real_feature_file = f"feature_{idx+1:02d}.npy"  # 1-indexed filenames
            real_feature_path = os.path.join(real_dataset_folder, 'Feature', real_feature_file)
            if os.path.exists(real_feature_path):
                real_feature_paths.append(real_feature_path)
            else:
                print(f"Warning: Real feature file not found: {real_feature_path}")
        
        for idx in synth_indices_with_condition:
            synth_feature_file = f"feature_{idx+1:02d}.npy"  # 1-indexed filenames
            synth_feature_path = os.path.join(synthetic_dataset_folder, 'Feature', synth_feature_file)
            if os.path.exists(synth_feature_path):
                synth_feature_paths.append(synth_feature_path)
            else:
                print(f"Warning: Synthetic feature file not found: {synth_feature_path}")
        
        if len(real_feature_paths) == 0 or len(synth_feature_paths) == 0:
            print(f"Warning: No valid feature files found for condition '{condition}'")
            continue
        
        grouped_samples[condition] = {
            'real_paths': real_feature_paths,
            'synthetic_paths': synth_feature_paths,
            'real_indices': real_indices_with_condition,
            'synthetic_indices': synth_indices_with_condition
        }
        
        print(f"Loaded {len(real_feature_paths)} real and {len(synth_feature_paths)} synthetic subjects for {condition}")
    
    return grouped_samples

def run_group_psd_analysis(real_dataset_folder, synthetic_dataset_folder, output_dir='group_statistical_analysis', 
                          conditions=['HC', 'MCI', 'Dementia'], sfreq=200, fmin=1, fmax=30, n_permutations=1000,
                          normalization_method='robust'):
    """
    Run complete group-level PSD statistical analysis comparing real and synthetic EEG data.
    
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
    normalization_method : str
        Normalization method: 'global', 'robust', or 'percentile'
        'robust' uses 5th-95th percentiles, 'percentile' uses 10th-90th percentiles
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("LOADING ALL SUBJECTS BY CONDITION FOR GROUP ANALYSIS")
    print("="*60)
    
    # Load all subjects for each condition
    grouped_samples = load_all_subjects_by_condition(
        real_dataset_folder, synthetic_dataset_folder, conditions
    )
    
    if not grouped_samples:
        raise ValueError("No valid subjects found for any condition")
    
    print(f"\nAnalyzing {len(grouped_samples)} conditions with group comparisons")
    
    all_results = {}
    
    # Use standard 10-20 EEG channel names
    standard_channels = ['FP1', 'F3', 'C3', 'P3', 'O1', 'FP2', 'F4', 'C4', 'P4', 'O2', 
                        'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'FZ', 'CZ', 'PZ']
    
    # Process each condition
    for condition, sample_info in grouped_samples.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING CONDITION: {condition}")
        print(f"{'='*60}")
        
        real_paths = sample_info['real_paths']
        synthetic_paths = sample_info['synthetic_paths']
        
        print(f"Real subjects: {len(real_paths)}")
        print(f"Synthetic subjects: {len(synthetic_paths)}")
        
        # Load all real subjects
        print("Loading all real EEG data...")
        real_all_psds = []
        for i, path in enumerate(real_paths):
            print(f"  Loading real subject {i+1}/{len(real_paths)}: {os.path.basename(path)}")
            data, _, n_channels, _ = load_and_preprocess_data(path, sfreq)
            psds, freqs = calculate_psd_for_statistical_analysis(data, sfreq, fmin, fmax, 
                                                               normalize=True, norm_method=normalization_method)
            real_all_psds.append(psds)
        
        # Load all synthetic subjects
        print("Loading all synthetic EEG data...")
        synthetic_all_psds = []
        for i, path in enumerate(synthetic_paths):
            print(f"  Loading synthetic subject {i+1}/{len(synthetic_paths)}: {os.path.basename(path)}")
            data, _, _, _ = load_and_preprocess_data(path, sfreq)
            psds, _ = calculate_psd_for_statistical_analysis(data, sfreq, fmin, fmax, 
                                                           normalize=True, norm_method=normalization_method)
            synthetic_all_psds.append(psds)
        
        # Convert to numpy arrays and reshape for group analysis
        # Shape: (n_subjects, n_epochs, n_channels, n_freqs) -> (n_subjects*n_epochs, n_channels, n_freqs)
        real_group_data = np.concatenate(real_all_psds, axis=0)
        synthetic_group_data = np.concatenate(synthetic_all_psds, axis=0)
        
        print(f"Group data shapes: Real {real_group_data.shape}, Synthetic {synthetic_group_data.shape}")
        print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz ({len(freqs)} frequencies)")
        
        # Set channel names
        if n_channels == len(standard_channels):
            channels = standard_channels
        else:
            print(f"Warning: Expected {len(standard_channels)} channels but found {n_channels}")
            print("Using generic channel names instead")
            channels = [f"EEG {i+1}" for i in range(n_channels)]
        
        # Run group-level statistical comparison
        print("Starting group-level statistical comparison...")
        comparator = GroupPSDStatisticalComparison(n_permutations=n_permutations, alpha=0.05, cluster_alpha=0.01)
        
        # Group-level permutation tests
        p_values, effect_sizes = comparator.permutation_test_groups(real_group_data, synthetic_group_data)
        
        # Cluster correction
        significant_clusters, cluster_info = comparator.cluster_correction(p_values, freqs, channels)
        
        # Create condition-specific output directory
        condition_output_dir = os.path.join(output_dir, condition)
        os.makedirs(condition_output_dir, exist_ok=True)
        
        # Create visualization
        fig = comparator.plot_statistical_heatmap(
            p_values, effect_sizes, significant_clusters, freqs, channels, cluster_info, condition
        )
        
        # Save figure
        fig_path = os.path.join(condition_output_dir, f'group_statistical_comparison_{condition}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved group statistical heatmap to {fig_path}")
        
        # Print results
        print(f"\n{condition} GROUP CLUSTER RESULTS:")
        comparator.print_cluster_results(cluster_info, freqs)
        
        # Create normalized PSD comparison plot in the style of the provided image
        print("Creating normalized PSD comparison plot...")
        fig_psd, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate PSDs averaged across channels for each subject, then get group statistics
        real_subject_psds = []  # PSD for each subject (averaged across channels)
        for subject_psd in real_all_psds:
            subject_avg = np.mean(subject_psd, axis=(0, 1))  # Average across epochs and channels
            real_subject_psds.append(subject_avg)
        real_subject_psds = np.array(real_subject_psds)
        
        synthetic_subject_psds = []
        for subject_psd in synthetic_all_psds:
            subject_avg = np.mean(subject_psd, axis=(0, 1))  # Average across epochs and channels
            synthetic_subject_psds.append(subject_avg)
        synthetic_subject_psds = np.array(synthetic_subject_psds)
        
        # Calculate group mean and standard deviation
        real_group_mean = np.mean(real_subject_psds, axis=0)
        real_group_std = np.std(real_subject_psds, axis=0)
        
        synthetic_group_mean = np.mean(synthetic_subject_psds, axis=0)
        synthetic_group_std = np.std(synthetic_subject_psds, axis=0)
        
        # Plot with bold center line and shaded area (like the provided image)
        # Real EEG
        ax.fill_between(freqs, real_group_mean - real_group_std, real_group_mean + real_group_std, 
                       color='blue', alpha=0.3, label=f'Real EEG spread (n={len(real_paths)})')
        ax.plot(freqs, real_group_mean, 'b-', linewidth=3, label=f'Real EEG mean', alpha=0.9)
        
        # Synthetic EEG  
        ax.fill_between(freqs, synthetic_group_mean - synthetic_group_std, synthetic_group_mean + synthetic_group_std, 
                       color='red', alpha=0.3, label=f'Synthetic EEG spread (n={len(synthetic_paths)})')
        ax.plot(freqs, synthetic_group_mean, 'r-', linewidth=3, label=f'Synthetic EEG mean', alpha=0.9)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Normalized PSD (0-1 scale)', fontsize=12)
        ax.set_title(f'Normalized Power Spectral Density - {condition}\n(Group comparison, grand average across all channels)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)  # Since we normalized to 0-1
        
        # Make the plot look more like the provided image
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        psd_comparison_path = os.path.join(condition_output_dir, f'normalized_psd_comparison_{condition}.png')
        fig_psd.savefig(psd_comparison_path, dpi=300, bbox_inches='tight')
        print(f"Saved normalized PSD comparison plot to {psd_comparison_path}")
        
        # Create individual beautiful plots for each group (like the provided image)
        print("Creating individual beautiful PSD plots...")
        
        # Real EEG plot
        fig_real = create_beautiful_psd_plot(
            freqs, real_group_mean, real_group_std, 
            f'Real EEG - {condition} (n={len(real_paths)})', 
            color='blue', normalized=True, norm_method=normalization_method
        )
        real_plot_path = os.path.join(condition_output_dir, f'real_eeg_psd_{condition}.png')
        fig_real.savefig(real_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved real EEG plot to {real_plot_path}")
        
        # Synthetic EEG plot
        fig_synthetic = create_beautiful_psd_plot(
            freqs, synthetic_group_mean, synthetic_group_std, 
            f'Synthetic EEG - {condition} (n={len(synthetic_paths)})', 
            color='red', normalized=True, norm_method=normalization_method
        )
        synthetic_plot_path = os.path.join(condition_output_dir, f'synthetic_eeg_psd_{condition}.png')
        fig_synthetic.savefig(synthetic_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved synthetic EEG plot to {synthetic_plot_path}")
        
        plt.close(fig_real)
        plt.close(fig_synthetic)
        
        # Save results
        results = {
            'p_values': p_values,
            'effect_sizes': effect_sizes,
            'significant_clusters': significant_clusters,
            'cluster_info': cluster_info,
            'freqs': freqs,
            'channels': channels,
            'condition': condition,
            'n_real_subjects': len(real_paths),
            'n_synthetic_subjects': len(synthetic_paths),
            'real_group_mean': real_group_mean,
            'real_group_std': real_group_std,
            'synthetic_group_mean': synthetic_group_mean,
            'synthetic_group_std': synthetic_group_std
        }
        
        results_path = os.path.join(condition_output_dir, f'group_statistical_results_{condition}.npz')
        np.savez(results_path, **{k: v for k, v in results.items() if not isinstance(v, dict)})
        print(f"Saved group statistical results to {results_path}")
        
        all_results[condition] = results
        
        plt.close('all')  # Close figures to save memory
    
    # Create summary comparison across all conditions
    print(f"\n{'='*60}")
    print("CREATING GROUP SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    create_group_summary_comparison(all_results, output_dir)
    
    print(f"\nGroup analysis complete! Check the '{output_dir}' directory for outputs.")
    print(f"Individual condition results are in subdirectories: {list(all_results.keys())}")
    
    return all_results

def create_group_summary_comparison(all_results, output_dir):
    """Create a summary comparison plot across all conditions for group analysis."""
    
    if not all_results:
        print("No results to summarize")
        return
    
    # Create summary plot
    fig, axes = plt.subplots(len(all_results), 2, figsize=(15, 5*len(all_results)))
    if len(all_results) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Group Summary: Normalized PSD Comparisons Across Conditions', fontsize=16, fontweight='bold')
    
    for idx, (condition, results) in enumerate(all_results.items()):
        freqs = results['freqs']
        real_group_mean = results['real_group_mean']
        real_group_std = results['real_group_std']
        synthetic_group_mean = results['synthetic_group_mean']
        synthetic_group_std = results['synthetic_group_std']
        significant_clusters = results['significant_clusters']
        n_real = results['n_real_subjects']
        n_synthetic = results['n_synthetic_subjects']
        
        # Normalized PSD comparison with bold center line and shaded area
        ax1 = axes[idx, 0]
        
        # Real EEG
        ax1.fill_between(freqs, real_group_mean - real_group_std, real_group_mean + real_group_std, 
                        color='blue', alpha=0.3)
        ax1.plot(freqs, real_group_mean, 'b-', linewidth=3, label=f'Real EEG (n={n_real})', alpha=0.9)
        
        # Synthetic EEG
        ax1.fill_between(freqs, synthetic_group_mean - synthetic_group_std, synthetic_group_mean + synthetic_group_std, 
                        color='red', alpha=0.3)
        ax1.plot(freqs, synthetic_group_mean, 'r-', linewidth=3, label=f'Synthetic EEG (n={n_synthetic})', alpha=0.9)
        
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Normalized PSD (0-1)')
        ax1.set_title(f'{condition} - Group Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Clean up the plot appearance
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
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
    
    summary_path = os.path.join(output_dir, 'group_summary_comparison.png')
    fig.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Saved group summary comparison to {summary_path}")
    
    plt.close()

# Example usage
if __name__ == "__main__":
    # Example paths - replace with your actual file paths
    real_data_path = 'dataset/CAUEEG2'
    synthetic_data_path = 'dataset/PURE_LDM_PSD_Normalized'
    
    # Run complete group-level analysis with robust normalization (recommended)
    # This uses 5th-95th percentiles for normalization, giving better range utilization
    results = run_group_psd_analysis(
        real_dataset_folder=real_data_path,
        synthetic_dataset_folder=synthetic_data_path,
        conditions=['HC', 'MCI', 'Dementia'],
        output_dir='images/group_statistical_analysis_robust_normalized_PURE_LDM_PSD_Normalized',
        normalization_method='robust'  # Options: 'global', 'robust', 'percentile'
    )
    
    # Alternative: If you want even more range utilization, try 'percentile' method
    # results = run_group_psd_analysis(
    #     real_dataset_folder=real_data_path,
    #     synthetic_dataset_folder=synthetic_data_path,
    #     conditions=['HC', 'MCI', 'Dementia'],
    #     output_dir='images/group_statistical_analysis_percentile_normalized',
    #     normalization_method='percentile'  # Uses 10th-90th percentiles
    # )
    
    print("\nGroup analysis complete! Check the output directory for results.")