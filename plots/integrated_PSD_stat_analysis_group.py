import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.ndimage import label
from tqdm import tqdm
import warnings
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
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

def load_all_patients_by_condition(dataset_folder, conditions=['HC', 'MCI', 'Dementia']):
    """
    Load ALL patients for each condition from the dataset.
    
    Parameters:
    -----------
    dataset_folder : str
        Path to dataset folder (e.g., 'dataset/CAUEEG2')
    conditions : list
        List of conditions to find (default: ['HC', 'MCI', 'Dementia'])
        
    Returns:
    --------
    condition_data : dict
        Dictionary with condition names as keys and concatenated data as values
    """
    
    # Load labels
    label_path = os.path.join(dataset_folder, 'Label', 'label.npy')
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    labels = np.load(label_path)
    print(f"Loaded labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # Map label values to condition names
    label_mapping = {
        0: 'HC',        # Healthy Control
        1: 'MCI',       # Mild Cognitive Impairment  
        2: 'Dementia'   # Dementia
    }
    
    condition_data = {}
    
    for condition in conditions:
        # Find the label value for this condition
        label_value = None
        for val, cond in label_mapping.items():
            if cond == condition:
                label_value = val
                break
        
        if label_value is None:
            print(f"Warning: Condition '{condition}' not found in label mapping")
            continue
            
        # Find ALL indices where this condition occurs
        condition_indices = np.where(labels == label_value)[0]
        
        if len(condition_indices) == 0:
            print(f"Warning: No samples found for condition '{condition}' (label {label_value})")
            continue
        
        print(f"Found {len(condition_indices)} patients for {condition}")
        
        # Load and concatenate all patients for this condition
        all_patient_data = []
        successful_loads = 0
        
        for patient_idx in condition_indices:
            feature_file = f"feature_{patient_idx+1:02d}.npy"  # 1-indexed filenames
            feature_path = os.path.join(dataset_folder, 'Feature', feature_file)
            
            if os.path.exists(feature_path):
                try:
                    patient_data, _, _, _ = load_and_preprocess_data(feature_path)
                    all_patient_data.append(patient_data)
                    successful_loads += 1
                except Exception as e:
                    print(f"Error loading {feature_file}: {e}")
            else:
                print(f"Warning: Feature file not found: {feature_path}")
        
        if all_patient_data:
            # Concatenate all patients along the epoch dimension
            condition_data[condition] = np.concatenate(all_patient_data, axis=0)
            print(f"Successfully loaded {successful_loads} patients for {condition}")
            print(f"Combined data shape: {condition_data[condition].shape}")
        else:
            print(f"No data loaded for condition {condition}")
    
    return condition_data

def calculate_psd_for_statistical_analysis(data, sfreq, fmin=1, fmax=30, nperseg=None):
    """Calculate PSD data in format needed for statistical analysis."""
    n_epochs, n_channels, n_times = data.shape
    
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

def global_psd_normalization(all_condition_psds):
    """
    Normalize PSDs globally across all conditions (0 to 1 scaling).
    
    Parameters:
    -----------
    all_condition_psds : dict
        Dictionary with condition names as keys and PSD arrays as values
        
    Returns:
    --------
    normalized_psds : dict
        Dictionary with globally normalized PSD arrays
    """
    print("Applying global PSD normalization across all conditions...")
    
    # Collect all PSD values from all conditions
    all_psd_values = []
    for condition, psds in all_condition_psds.items():
        all_psd_values.append(psds.flatten())
    
    # Concatenate all values
    global_psd_values = np.concatenate(all_psd_values)
    
    # Find global min and max
    global_min = np.min(global_psd_values)
    global_max = np.max(global_psd_values)
    
    print(f"Global PSD range: {global_min:.6f} to {global_max:.6f}")
    
    # Normalize each condition's PSDs using global range
    normalized_psds = {}
    for condition, psds in all_condition_psds.items():
        normalized_psds[condition] = (psds - global_min) / (global_max - global_min)
        print(f"{condition} normalized range: {np.min(normalized_psds[condition]):.6f} to {np.max(normalized_psds[condition]):.6f}")
    
    return normalized_psds

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

def create_supervisor_requested_figure(all_results, freqs, channels, output_dir):
    """
    Create the 3x2 grid figure as requested by supervisor:
    - Top row: PSD comparisons for each condition with 5-95% confidence intervals
    - Bottom row: Effect size heatmaps with significance contours
    
    FIXED: Ensures conditions are always in HC, MCI, Dementia order
    ADDED: Diagnostic prints to check for identical synthetic data
    ADDED: 5-95% confidence intervals on PSD plots
    """
    
    # FIXED ORDER: Always HC, MCI, Dementia
    desired_order = ['HC', 'MCI', 'Dementia']
    available_conditions = list(all_results.keys())
    
    # Get conditions in the desired order
    conditions = [cond for cond in desired_order if cond in available_conditions]
    
    # Add any other conditions that might exist but aren't in the standard list
    for cond in available_conditions:
        if cond not in conditions:
            conditions.append(cond)
    
    print(f"Available conditions: {available_conditions}")
    print(f"Plotting conditions in fixed order: {conditions}")
    
    # DIAGNOSTIC: Check if synthetic PSDs are actually different
    print("\n" + "="*60)
    print("DIAGNOSTIC: Checking synthetic PSD differences")
    print("="*60)
    
    synthetic_stats = {}
    for condition in conditions:
        synthetic_psds = all_results[condition]['synthetic_psds']
        synthetic_grand_avg = np.mean(synthetic_psds, axis=(0, 1))
        synthetic_stats[condition] = {
            'mean': np.mean(synthetic_grand_avg),
            'std': np.std(synthetic_grand_avg),
            'max': np.max(synthetic_grand_avg),
            'min': np.min(synthetic_grand_avg)
        }
        print(f"{condition} synthetic - Mean: {synthetic_stats[condition]['mean']:.6f}, "
              f"Std: {synthetic_stats[condition]['std']:.6f}, "
              f"Range: {synthetic_stats[condition]['min']:.6f} to {synthetic_stats[condition]['max']:.6f}")
    
    # Check if synthetic data is identical between conditions
    if len(conditions) >= 2:
        synth_1 = np.mean(all_results[conditions[0]]['synthetic_psds'], axis=(0, 1))
        synth_2 = np.mean(all_results[conditions[1]]['synthetic_psds'], axis=(0, 1))
        correlation = np.corrcoef(synth_1, synth_2)[0, 1]
        max_diff = np.max(np.abs(synth_1 - synth_2))
        print(f"\nCorrelation between {conditions[0]} and {conditions[1]} synthetic PSDs: {correlation:.6f}")
        print(f"Max absolute difference: {max_diff:.6f}")
        
        if correlation > 0.99 and max_diff < 1e-6:
            print("⚠️  WARNING: Synthetic PSDs appear nearly identical between conditions!")
            print("   This suggests the synthetic data generation didn't preserve condition differences.")
        elif correlation > 0.95:
            print("⚠️  WARNING: Synthetic PSDs are very similar between conditions (r > 0.95)")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Define frequency ticks
    freq_ticks = np.linspace(0, len(freqs)-1, 6).astype(int)
    freq_labels = [f'{freqs[i]:.1f}' for i in freq_ticks]
    
    # Top row: PSD comparisons with confidence intervals
    for idx, condition in enumerate(conditions):
        ax = axes[0, idx]
        results = all_results[condition]
        
        real_psds = results['real_psds'] 
        synthetic_psds = results['synthetic_psds']
        
        # Average across channels first, then calculate statistics across epochs
        # Shape: (n_epochs, n_channels, n_freqs) -> (n_epochs, n_freqs)
        real_channel_avg = np.mean(real_psds, axis=1)  # Average across channels
        synthetic_channel_avg = np.mean(synthetic_psds, axis=1)  # Average across channels
        
        # Calculate mean and percentiles across epochs for each frequency
        real_mean = np.mean(real_channel_avg, axis=0)
        real_p5 = np.percentile(real_channel_avg, 5, axis=0)
        real_p95 = np.percentile(real_channel_avg, 95, axis=0)
        
        synthetic_mean = np.mean(synthetic_channel_avg, axis=0)
        synthetic_p5 = np.percentile(synthetic_channel_avg, 5, axis=0)
        synthetic_p95 = np.percentile(synthetic_channel_avg, 95, axis=0)
        
        # Plot confidence intervals first (so they appear behind the lines)
        ax.fill_between(freqs, real_p5, real_p95, 
                       color='blue', alpha=0.2, label='Real 5-95%')
        ax.fill_between(freqs, synthetic_p5, synthetic_p95, 
                       color='red', alpha=0.2, label='Synthetic 5-95%')
        
        # Plot mean lines on top
        ax.semilogy(freqs, real_mean, 'b-', linewidth=2.5, label='Real EEG (mean)', alpha=0.9)
        ax.semilogy(freqs, synthetic_mean, 'r-', linewidth=2.5, label='Synthetic EEG (mean)', alpha=0.9)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('PSD (µV²/Hz)', fontsize=12)
        ax.set_title(f'{condition} - PSD Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add frequency band shading
        alpha_band = (freqs >= 8) & (freqs <= 12)
        if np.any(alpha_band):
            ax.axvspan(8, 12, alpha=0.1, color='green', label='Alpha' if idx == 0 else "")
    
    # Bottom row: Effect size heatmaps with significance contours
    for idx, condition in enumerate(conditions):
        ax = axes[1, idx]
        results = all_results[condition]
        
        effect_sizes = results['effect_sizes']
        significant_clusters = results['significant_clusters']
        
        # Create centered colormap for effect sizes (centered at 0)
        vmax = np.max(np.abs(effect_sizes))
        vmin = -vmax
        
        # Plot effect sizes
        im = ax.imshow(effect_sizes, aspect='auto', cmap='RdBu_r', 
                      interpolation='nearest', vmin=vmin, vmax=vmax)
        
        # Add significance contours with more opaque lines
        if np.any(significant_clusters):
            # Create contours around significant regions
            ax.contour(significant_clusters.astype(float), levels=[0.5], 
                      colors='black', linewidths=2.5, linestyles='-', alpha=1.0)
            
            # Optional: Add filled contours for better visibility
            ax.contourf(significant_clusters.astype(float), levels=[0.5, 1.5], 
                       colors=['none'], hatches=['///'], alpha=0.15)
        
        # Set ticks and labels
        ax.set_xticks(freq_ticks)
        ax.set_xticklabels(freq_labels)
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels(channels, fontsize=10)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Channels', fontsize=12)
        ax.set_title(f'{condition} - Effect Sizes with Significance', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Effect Size (Real - Synthetic)', fontsize=11)
        
        # Add text annotation for significant clusters
        n_sig = np.sum(significant_clusters)
        ax.text(0.02, 0.98, f'Significant: {n_sig}/{significant_clusters.size}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, 'supervisor_requested_psd_analysis.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved supervisor-requested figure to {fig_path}")
    
    plt.close()
    
    return fig_path

def run_complete_psd_analysis(real_dataset_folder, synthetic_dataset_folder, output_dir='statistical_analysis', 
                            conditions=['HC', 'MCI', 'Dementia'], sfreq=200, fmin=1, fmax=30, 
                            n_permutations=1000, use_global_normalization=True):
    """
    Run complete PSD statistical analysis using ALL patients for each condition.
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("LOADING ALL PATIENTS BY CONDITION")
    print("="*80)
    
    # Load ALL patients for each condition from both datasets
    print("\nReal dataset:")
    real_condition_data = load_all_patients_by_condition(real_dataset_folder, conditions)
    
    print("\nSynthetic dataset:")
    synthetic_condition_data = load_all_patients_by_condition(synthetic_dataset_folder, conditions)
    
    # Check that we have data for each condition in both datasets
    common_conditions = set(real_condition_data.keys()) & set(synthetic_condition_data.keys())
    if not common_conditions:
        raise ValueError("No common conditions found between real and synthetic datasets")
    
    # FIXED ORDER: Always process conditions in HC, MCI, Dementia order
    desired_order = ['HC', 'MCI', 'Dementia']
    ordered_conditions = [cond for cond in desired_order if cond in common_conditions]
    
    # Add any other conditions that might exist but aren't in the standard list
    for cond in common_conditions:
        if cond not in ordered_conditions:
            ordered_conditions.append(cond)
    
    print(f"Available conditions: {list(common_conditions)}")
    print(f"Processing conditions in fixed order: {ordered_conditions}")
    
    # Use standard 10-20 EEG channel names
    standard_channels = ['FP1', 'F3', 'C3', 'P3', 'O1', 'FP2', 'F4', 'C4', 'P4', 'O2', 
                        'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'FZ', 'CZ', 'PZ']
    
    # Get number of channels from first dataset
    first_condition = ordered_conditions[0]
    n_channels = real_condition_data[first_condition].shape[1]
    
    if n_channels == len(standard_channels):
        channels = standard_channels
    else:
        print(f"Warning: Expected {len(standard_channels)} channels but found {n_channels}")
        channels = [f"EEG {i+1}" for i in range(n_channels)]
    
    print("="*80)
    print("CALCULATING PSDs FOR ALL CONDITIONS")
    print("="*80)
    
    # Calculate PSDs for all conditions in fixed order
    real_psds_all = {}
    synthetic_psds_all = {}
    
    for condition in ordered_conditions:
        print(f"\nProcessing {condition}...")
        
        # Calculate PSDs
        print("Calculating PSDs for real data...")
        real_psds_all[condition], freqs = calculate_psd_for_statistical_analysis(
            real_condition_data[condition], sfreq, fmin, fmax)
        
        print("Calculating PSDs for synthetic data...")
        synthetic_psds_all[condition], _ = calculate_psd_for_statistical_analysis(
            synthetic_condition_data[condition], sfreq, fmin, fmax)
        
        print(f"Real PSDs shape: {real_psds_all[condition].shape}")
        print(f"Synthetic PSDs shape: {synthetic_psds_all[condition].shape}")
    
    # Apply global normalization if requested
    if use_global_normalization:
        print("\n" + "="*80)
        print("APPLYING GLOBAL PSD NORMALIZATION")
        print("="*80)
        
        # Combine real and synthetic for global normalization in fixed order
        all_psds_combined = {}
        for condition in ordered_conditions:
            all_psds_combined[f"{condition}_real"] = real_psds_all[condition]
            all_psds_combined[f"{condition}_synthetic"] = synthetic_psds_all[condition]
        
        # Apply global normalization
        normalized_psds = global_psd_normalization(all_psds_combined)
        
        # Separate back into real and synthetic, maintaining order
        for condition in ordered_conditions:
            real_psds_all[condition] = normalized_psds[f"{condition}_real"]
            synthetic_psds_all[condition] = normalized_psds[f"{condition}_synthetic"]
    
    print("\n" + "="*80)
    print("RUNNING STATISTICAL COMPARISONS")
    print("="*80)
    
    # Store results in ordered dictionary to maintain condition order
    all_results = {}
    
    # Run statistical analysis for each condition in fixed order
    for condition in ordered_conditions:
        print(f"\n{'='*60}")
        print(f"STATISTICAL ANALYSIS: {condition}")
        print(f"{'='*60}")
        
        real_psds = real_psds_all[condition]
        synthetic_psds = synthetic_psds_all[condition]
        
        print(f"Real epochs: {real_psds.shape[0]}, Synthetic epochs: {synthetic_psds.shape[0]}")
        
        # Run statistical comparison
        comparator = PSDStatisticalComparison(n_permutations=n_permutations, alpha=0.05, cluster_alpha=0.01)
        
        # Mass-univariate tests
        p_values, effect_sizes = comparator.mass_univariate_test(real_psds, synthetic_psds, freqs, channels)
        
        # Cluster correction
        significant_clusters, cluster_info = comparator.cluster_correction(p_values, freqs, channels)
        
        # Store results
        all_results[condition] = {
            'p_values': p_values,
            'effect_sizes': effect_sizes,
            'significant_clusters': significant_clusters,
            'cluster_info': cluster_info,
            'real_psds': real_psds,
            'synthetic_psds': synthetic_psds,
            'n_real_epochs': real_psds.shape[0],
            'n_synthetic_epochs': synthetic_psds.shape[0]
        }
        
        # Print cluster results
        print(f"\n{condition} CLUSTER RESULTS:")
        if cluster_info:
            for cluster_id, info in cluster_info.items():
                print(f"  Cluster {cluster_id}: {info['size']} freq-ch combinations, "
                      f"freq range: {info['freq_range'][0]:.1f}-{info['freq_range'][1]:.1f} Hz, "
                      f"channels: {', '.join(info['channels'][:3])}{'...' if len(info['channels']) > 3 else ''}")
        else:
            print("  No significant clusters found.")
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create the supervisor-requested figure with fixed condition order
    supervisor_fig_path = create_supervisor_requested_figure(all_results, freqs, channels, output_dir)
    
    # Save comprehensive results
    results_summary = {
        'conditions': ordered_conditions,  # Use ordered conditions
        'freqs': freqs,
        'channels': channels,
        'use_global_normalization': use_global_normalization,
        'n_permutations': n_permutations
    }
    
    # Add condition-specific statistics
    for condition, results in all_results.items():
        results_summary[f'{condition}_n_real'] = results['n_real_epochs']
        results_summary[f'{condition}_n_synthetic'] = results['n_synthetic_epochs']
        results_summary[f'{condition}_n_significant'] = np.sum(results['significant_clusters'])
        results_summary[f'{condition}_n_clusters'] = len(results['cluster_info'])
    
    # Save summary
    summary_path = os.path.join(output_dir, 'analysis_summary.npz')
    np.savez(summary_path, **results_summary)
    print(f"Saved analysis summary to {summary_path}")
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Main figure: {supervisor_fig_path}")
    print(f"Conditions processed in order: {ordered_conditions}")
    
    return all_results, freqs, channels

# Example usage
if __name__ == "__main__":
    # Specify dataset folders
    real_dataset_folder = '/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG2'
    synthetic_dataset_folder = '/home/stud/timlin/bhome/DiffusionEEG/dataset/SYNTH-CAUEEG2-NORMALIZED'
    
    # Run complete analysis
    results, freqs, channels = run_complete_psd_analysis(
        real_dataset_folder=real_dataset_folder,
        synthetic_dataset_folder=synthetic_dataset_folder,
        conditions=['HC', 'MCI', 'Dementia'],
        output_dir='images/5_95/statistical_analysis_full_datasets_SYNTH-CAUEEG2-NORMALIZED',
        sfreq=200,
        fmin=1,
        fmax=30,
        n_permutations=500,  # Increase for final analysis
        use_global_normalization=True  # Test both True and False
    )
    
    print("\nAnalysis complete! Check the output directory for results.")