import numpy as np
import mne
from mne.stats import permutation_cluster_test
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

def normalize_psd_0_1(psd_data):
    """
    Normalize PSD data to 0-1 scale for visualization.
    
    Parameters:
    -----------
    psd_data : array
        PSD data (can be 1D, 2D, or 3D)
        
    Returns:
    --------
    normalized_psd : array
        PSD data normalized to 0-1 scale
    """
    psd_min = np.min(psd_data)
    psd_max = np.max(psd_data)
    
    if psd_max - psd_min > 0:
        normalized_psd = (psd_data - psd_min) / (psd_max - psd_min)
    else:
        normalized_psd = psd_data
    
    return normalized_psd

def load_all_subjects_by_condition(dataset_folder, condition_label):
    """
    Load all subjects for a specific condition.
    
    Parameters:
    -----------
    dataset_folder : str
        Path to dataset folder
    condition_label : int
        Label value for the condition (0=HC, 1=MCI, 2=Dementia)
        
    Returns:
    --------
    data_list : list
        List of data arrays for all subjects with this condition
    subject_ids : array
        Subject IDs for the loaded data
    indices : array
        Original indices in the dataset
    """
    # Load labels
    label_path = os.path.join(dataset_folder, 'Label', 'label.npy')
    labels_array = np.load(label_path)
    
    if labels_array.ndim == 2 and labels_array.shape[1] == 2:
        condition_labels = labels_array[:, 0]
        subject_ids_all = labels_array[:, 1]
    else:
        condition_labels = labels_array
        subject_ids_all = np.arange(1, len(labels_array) + 1)
    
    # Find all subjects with this condition
    condition_mask = condition_labels == condition_label
    indices = np.where(condition_mask)[0]
    subject_ids = subject_ids_all[condition_mask]
    
    data_list = []
    valid_indices = []
    valid_subject_ids = []
    
    print(f"Loading {len(indices)} subjects for condition {condition_label}")
    
    for idx, subj_id in zip(indices, subject_ids):
        feature_file = f"feature_{idx+1:02d}.npy"
        feature_path = os.path.join(dataset_folder, 'Feature', feature_file)
        
        if os.path.exists(feature_path):
            try:
                data, _, _, _ = load_and_preprocess_data(feature_path)
                data_list.append(data)
                valid_indices.append(idx)
                valid_subject_ids.append(subj_id)
            except Exception as e:
                print(f"Error loading {feature_file}: {e}")
        else:
            print(f"File not found: {feature_path}")
    
    print(f"Successfully loaded {len(data_list)} subjects")
    
    return data_list, np.array(valid_subject_ids), np.array(valid_indices)

def run_mne_cluster_permutation_test(real_psds, synthetic_psds, freqs, n_permutations=1000, alpha=0.05):
    """
    Run MNE's permutation cluster test for paired samples.
    
    Parameters:
    -----------
    real_psds : list of arrays
        List of PSD arrays for real subjects, each shape (n_epochs, n_channels, n_freqs)
    synthetic_psds : list of arrays
        List of PSD arrays for synthetic subjects (must be paired with real)
    freqs : array
        Frequency values
    n_permutations : int
        Number of permutations
    alpha : float
        Significance level
        
    Returns:
    --------
    cluster_results : dict
        Results from cluster permutation test
    """
    # Average PSDs across epochs for each subject
    real_avg_list = [np.mean(psd, axis=0) for psd in real_psds]  # Each becomes (n_channels, n_freqs)
    synth_avg_list = [np.mean(psd, axis=0) for psd in synthetic_psds]
    
    # Stack subjects to create (n_subjects, n_channels, n_freqs)
    real_stacked = np.stack(real_avg_list, axis=0)
    synth_stacked = np.stack(synth_avg_list, axis=0)
    
    # Reshape for MNE format: (n_subjects, n_channels * n_freqs)
    n_subjects, n_channels, n_freqs = real_stacked.shape
    real_reshaped = real_stacked.reshape(n_subjects, -1)
    synth_reshaped = synth_stacked.reshape(n_subjects, -1)
    
    # Create observation array for paired test
    X = np.array([real_reshaped, synth_reshaped])
    X = np.transpose(X, [1, 2, 0])  # Shape: (n_subjects, n_features, n_conditions)
    
    print(f"Running paired permutation cluster test with {n_subjects} subjects...")
    print(f"Data shape for test: {X.shape}")
    
    # Run permutation cluster test
    # For paired samples, we test the difference against 0
    threshold = None  # Will use default threshold based on t-distribution
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        X, n_permutations=n_permutations, threshold=threshold, tail=0,
        n_jobs=1, buffer_size=None, out_type='mask'
    )
    
    # Reshape T_obs back to (n_channels, n_freqs)
    T_obs_2d = T_obs.reshape(n_channels, n_freqs)
    
    # Process clusters
    significant_clusters_mask = np.zeros((n_channels, n_freqs), dtype=bool)
    cluster_info = {}
    
    for cluster_idx, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
        if p_val < alpha:
            # Reshape cluster mask
            cluster_2d = cluster.reshape(n_channels, n_freqs)
            significant_clusters_mask |= cluster_2d
            
            # Find extent of cluster
            ch_indices, freq_indices = np.where(cluster_2d)
            
            if len(ch_indices) > 0:
                cluster_info[cluster_idx] = {
                    'p_value': p_val,
                    'size': len(ch_indices),
                    'freq_range': (freqs[freq_indices.min()], freqs[freq_indices.max()]),
                    'channels': np.unique(ch_indices).tolist(),
                    'freq_indices': np.unique(freq_indices).tolist()
                }
    
    return {
        'T_obs': T_obs_2d,
        'significant_clusters': significant_clusters_mask,
        'cluster_info': cluster_info,
        'cluster_p_values': cluster_p_values,
        'n_subjects': n_subjects
    }

def plot_group_statistical_results(cluster_results, freqs, channels, condition, output_dir):
    """
    Create comprehensive visualization of group statistical results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Group Statistical Comparison: Real vs Synthetic EEG - {condition}\n'
                 f'({cluster_results["n_subjects"]} paired subjects)', 
                 fontsize=16, fontweight='bold')
    
    # 1. T-statistic heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(cluster_results['T_obs'], aspect='auto', cmap='RdBu_r', 
                     interpolation='nearest')
    ax1.set_title('T-statistic Map\n(Paired t-test)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Channels')
    
    # Set frequency ticks
    freq_ticks = np.linspace(0, len(freqs)-1, 6).astype(int)
    ax1.set_xticks(freq_ticks)
    ax1.set_xticklabels([f'{freqs[i]:.1f}' for i in freq_ticks])
    
    # Set channel ticks
    ax1.set_yticks(range(len(channels)))
    ax1.set_yticklabels(channels)
    
    plt.colorbar(im1, ax=ax1, label='T-statistic')
    
    # 2. Effect sizes with centered colormap
    ax2 = axes[0, 1]
    # Calculate effect sizes as mean difference normalized by pooled std
    effect_sizes = cluster_results['T_obs'] / np.sqrt(cluster_results['n_subjects'])
    
    # Center colormap at 0
    vmax = np.max(np.abs(effect_sizes))
    im2 = ax2.imshow(effect_sizes, aspect='auto', cmap='RdBu_r', 
                     interpolation='nearest', vmin=-vmax, vmax=vmax)
    ax2.set_title('Effect Sizes\n(Centered at 0)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Channels')
    
    ax2.set_xticks(freq_ticks)
    ax2.set_xticklabels([f'{freqs[i]:.1f}' for i in freq_ticks])
    ax2.set_yticks(range(len(channels)))
    ax2.set_yticklabels(channels)
    
    cbar = plt.colorbar(im2, ax=ax2, label='Effect Size')
    
    # 3. Significant clusters
    ax3 = axes[1, 0]
    im3 = ax3.imshow(cluster_results['significant_clusters'].astype(int), 
                     aspect='auto', cmap='Reds', interpolation='nearest')
    ax3.set_title('Significant Clusters\n(Cluster-corrected, p < 0.05)')
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
    
    n_significant_clusters = len([p for p in cluster_results['cluster_p_values'] if p < 0.05])
    n_total_clusters = len(cluster_results['cluster_p_values'])
    
    summary_text = f"""Statistical Summary:

Paired subjects: {cluster_results['n_subjects']}
Test type: Paired permutation cluster test
Significance level: α = 0.05

Cluster Analysis:
Total clusters found: {n_total_clusters}
Significant clusters: {n_significant_clusters}

Effect Size Range:
Min: {np.min(effect_sizes):.4f}
Max: {np.max(effect_sizes):.4f}
Mean: {np.mean(effect_sizes):.4f}

Note: Effect sizes centered at 0
(0 = no difference between conditions)
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, f'group_statistical_comparison_{condition}.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved group statistical comparison to {fig_path}")
    
    return fig

def plot_normalized_psd_comparison(real_psds, synthetic_psds, freqs, condition, output_dir):
    """
    Create PSD comparison plot with 0-1 normalization.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Power Spectral Density Comparison - {condition}\n'
                 f'({len(real_psds)} subjects per group)', 
                 fontsize=16, fontweight='bold')
    
    # Calculate grand averages
    real_grand_avg_list = []
    synth_grand_avg_list = []
    
    for real_psd, synth_psd in zip(real_psds, synthetic_psds):
        # Average across epochs and channels for each subject
        real_grand_avg_list.append(np.mean(real_psd, axis=(0, 1)))
        synth_grand_avg_list.append(np.mean(synth_psd, axis=(0, 1)))
    
    # Stack and calculate mean and std across subjects
    real_grand_avg = np.mean(real_grand_avg_list, axis=0)
    real_grand_std = np.std(real_grand_avg_list, axis=0)
    synth_grand_avg = np.mean(synth_grand_avg_list, axis=0)
    synth_grand_std = np.std(synth_grand_avg_list, axis=0)
    
    # Plot 1: Original scale (log)
    ax1.semilogy(freqs, real_grand_avg, 'b-', linewidth=2, label='Real EEG', alpha=0.8)
    ax1.fill_between(freqs, real_grand_avg - real_grand_std, real_grand_avg + real_grand_std,
                     alpha=0.3, color='blue')
    ax1.semilogy(freqs, synth_grand_avg, 'r-', linewidth=2, label='Synthetic EEG', alpha=0.8)
    ax1.fill_between(freqs, synth_grand_avg - synth_grand_std, synth_grand_avg + synth_grand_std,
                     alpha=0.3, color='red')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD (µV²/Hz)')
    ax1.set_title('Original Scale (Log)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Normalized 0-1 scale
    # Normalize each separately to see their patterns
    real_norm = normalize_psd_0_1(real_grand_avg)
    synth_norm = normalize_psd_0_1(synth_grand_avg)
    
    ax2.plot(freqs, real_norm, 'b-', linewidth=2, label='Real EEG (normalized)', alpha=0.8)
    ax2.plot(freqs, synth_norm, 'r-', linewidth=2, label='Synthetic EEG (normalized)', alpha=0.8)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Normalized PSD (0-1)')
    ax2.set_title('Normalized Scale (0-1)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    # Add frequency band annotations
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 
             'Beta': (12, 30), 'Gamma': (30, 100)}
    
    for band_name, (low, high) in bands.items():
        if low < freqs[-1]:
            band_mask = (freqs >= low) & (freqs <= min(high, freqs[-1]))
            if np.any(band_mask):
                mid_freq = (low + min(high, freqs[-1])) / 2
                if mid_freq <= freqs[-1]:
                    ax2.axvspan(low, min(high, freqs[-1]), alpha=0.1, color='gray')
                    ax2.text(mid_freq, 1.02, band_name, ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, f'psd_comparison_normalized_{condition}.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved normalized PSD comparison to {fig_path}")
    
    return fig

def run_group_psd_analysis(real_dataset_folder, synthetic_dataset_folder, output_dir='statistical_analysis_group', 
                          conditions=['HC', 'MCI', 'Dementia'], sfreq=200, fmin=1, fmax=30, n_permutations=1000):
    """
    Run group-level PSD statistical analysis comparing real and synthetic EEG data.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Label mapping
    label_mapping = {
        'HC': 0,        # Healthy Control
        'MCI': 1,       # Mild Cognitive Impairment  
        'Dementia': 2   # Dementia
    }
    
    # Standard 10-20 EEG channel names
    standard_channels = ['FP1', 'F3', 'C3', 'P3', 'O1', 'FP2', 'F4', 'C4', 'P4', 'O2', 
                        'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'FZ', 'CZ', 'PZ']
    
    all_results = {}
    
    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"PROCESSING CONDITION: {condition}")
        print(f"{'='*60}")
        
        condition_label = label_mapping.get(condition)
        if condition_label is None:
            print(f"Warning: Unknown condition {condition}")
            continue
        
        # Load all subjects for this condition
        print(f"\nLoading real {condition} subjects...")
        real_data_list, real_subject_ids, real_indices = load_all_subjects_by_condition(
            real_dataset_folder, condition_label
        )
        
        print(f"\nLoading synthetic {condition} subjects...")
        synth_data_list, synth_subject_ids, synth_indices = load_all_subjects_by_condition(
            synthetic_dataset_folder, condition_label
        )
        
        # Find matching subjects
        common_ids = np.intersect1d(real_subject_ids, synth_subject_ids)
        print(f"\nFound {len(common_ids)} matching subject IDs: {common_ids}")
        
        if len(common_ids) < 2:
            print(f"Warning: Need at least 2 matching subjects for paired test. Skipping {condition}.")
            continue
        
        # Match subjects and calculate PSDs
        real_psds = []
        synth_psds = []
        
        print("\nCalculating PSDs for matched subjects...")
        for subj_id in common_ids:
            # Find indices
            real_idx = np.where(real_subject_ids == subj_id)[0][0]
            synth_idx = np.where(synth_subject_ids == subj_id)[0][0]
            
            # Calculate PSDs
            real_psd, freqs = calculate_psd_for_statistical_analysis(
                real_data_list[real_idx], sfreq, fmin, fmax
            )
            synth_psd, _ = calculate_psd_for_statistical_analysis(
                synth_data_list[synth_idx], sfreq, fmin, fmax
            )
            
            real_psds.append(real_psd)
            synth_psds.append(synth_psd)
        
        print(f"Processed {len(real_psds)} matched subject pairs")
        
        # Check channel count
        n_channels = real_psds[0].shape[1]
        if n_channels == len(standard_channels):
            channels = standard_channels
        else:
            channels = [f"EEG {i+1}" for i in range(n_channels)]
        
        # Run MNE cluster permutation test
        print("\nRunning paired permutation cluster test...")
        cluster_results = run_mne_cluster_permutation_test(
            real_psds, synth_psds, freqs, n_permutations
        )
        
        # Create condition-specific output directory
        condition_output_dir = os.path.join(output_dir, condition)
        os.makedirs(condition_output_dir, exist_ok=True)
        
        # Create visualizations
        print("\nCreating visualizations...")
        
        # 1. Group statistical results
        plot_group_statistical_results(
            cluster_results, freqs, channels, condition, condition_output_dir
        )
        
        # 2. Normalized PSD comparison
        plot_normalized_psd_comparison(
            real_psds, synth_psds, freqs, condition, condition_output_dir
        )
        
        # Print cluster results
        print(f"\n{condition} CLUSTER RESULTS:")
        if cluster_results['cluster_info']:
            for cluster_id, info in cluster_results['cluster_info'].items():
                print(f"\nCluster {cluster_id}:")
                print(f"  p-value: {info['p_value']:.6f}")
                print(f"  Size: {info['size']} channel-frequency combinations")
                print(f"  Frequency range: {info['freq_range'][0]:.1f} - {info['freq_range'][1]:.1f} Hz")
                print(f"  Channels: {len(info['channels'])} involved")
        else:
            print("No significant clusters found.")
        
        # Save results
        results = {
            'cluster_results': cluster_results,
            'freqs': freqs,
            'channels': channels,
            'condition': condition,
            'n_subjects': len(common_ids),
            'subject_ids': common_ids
        }
        
        all_results[condition] = results
        
        # Save numerical results
        results_path = os.path.join(condition_output_dir, f'group_results_{condition}.npz')
        np.savez(results_path, 
                 T_obs=cluster_results['T_obs'],
                 significant_clusters=cluster_results['significant_clusters'],
                 freqs=freqs,
                 n_subjects=cluster_results['n_subjects'],
                 subject_ids=common_ids)
        print(f"Saved results to {results_path}")
        
        plt.close('all')  # Close figures to save memory
    
    # Create summary
    print(f"\n{'='*60}")
    print("CREATING SUMMARY")
    print(f"{'='*60}")
    
    create_group_summary(all_results, output_dir)
    
    print(f"\nAnalysis complete! Check the '{output_dir}' directory for outputs.")
    
    return all_results

def create_group_summary(all_results, output_dir):
    """Create a summary visualization across all conditions."""
    
    if not all_results:
        print("No results to summarize")
        return
    
    fig, axes = plt.subplots(len(all_results), 2, figsize=(15, 5*len(all_results)))
    if len(all_results) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Summary: Group-Level Statistical Comparisons', fontsize=16, fontweight='bold')
    
    for idx, (condition, results) in enumerate(all_results.items()):
        cluster_results = results['cluster_results']
        freqs = results['freqs']
        channels = results['channels']
        
        # T-statistic map
        ax1 = axes[idx, 0]
        im1 = ax1.imshow(cluster_results['T_obs'], aspect='auto', cmap='RdBu_r', 
                        interpolation='nearest')
        ax1.set_title(f'{condition} - T-statistics ({results["n_subjects"]} subjects)')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Channels')
        
        # Set frequency ticks
        freq_ticks = np.linspace(0, len(freqs)-1, 6).astype(int)
        ax1.set_xticks(freq_ticks)
        ax1.set_xticklabels([f'{freqs[i]:.1f}' for i in freq_ticks])
        
        # Set channel ticks (show fewer for readability)
        if len(channels) > 10:
            ch_ticks = np.linspace(0, len(channels)-1, 6).astype(int)
            ax1.set_yticks(ch_ticks)
            ax1.set_yticklabels([channels[i] for i in ch_ticks])
        else:
            ax1.set_yticks(range(len(channels)))
            ax1.set_yticklabels(channels)
        
        plt.colorbar(im1, ax=ax1, label='T-statistic')
        
        # Significant clusters
        ax2 = axes[idx, 1]
        im2 = ax2.imshow(cluster_results['significant_clusters'].astype(int), 
                        aspect='auto', cmap='Reds', interpolation='nearest')
        ax2.set_title(f'{condition} - Significant Clusters')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Channels')
        
        ax2.set_xticks(freq_ticks)
        ax2.set_xticklabels([f'{freqs[i]:.1f}' for i in freq_ticks])
        
        if len(channels) > 10:
            ax2.set_yticks(ch_ticks)
            ax2.set_yticklabels([channels[i] for i in ch_ticks])
        else:
            ax2.set_yticks(range(len(channels)))
            ax2.set_yticklabels(channels)
        
        plt.colorbar(im2, ax=ax2, label='Significant')
    
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, 'group_summary_comparison.png')
    fig.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Saved group summary to {summary_path}")
    
    plt.close()

# Example usage
if __name__ == "__main__":
    # Example paths - replace with your actual file paths
    real_data_path = 'dataset/CAUEEG2'
    synthetic_data_path = 'dataset/DM_NO_SPEC'
    
    # Run group-level analysis
    results = run_group_psd_analysis(
        real_dataset_folder=real_data_path,
        synthetic_dataset_folder=synthetic_data_path,
        conditions=['HC', 'MCI', 'Dementia'],
        output_dir='images/group_statistical_analysis_DM_NO_SPEC',
        n_permutations=1000
    )
    
    print("\nAnalysis complete! Check the output directory for results.")