import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import glob

# ---------------------------
# Configuration and Constants
# ---------------------------
# Standard set of 19 EEG channels (10-20 system)
STANDARD_CHANNELS = ['FP1', 'F3', 'C3', 'P3', 'O1', 'FP2', 'F4', 'C4', 'P4', 'O2', 
                    'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'FZ', 'CZ', 'PZ']

# Class names for labels
CLASS_NAMES = {0: 'HC (+SMC)', 1: 'MCI', 2: 'Dementia'}

def load_npy_to_epochs(feature_path, label_path=None, ch_names=None, normalize=False, sfreq=200.0, debug=True):
    """
    Load .npy files and convert to mne.Epochs objects
    
    Parameters
    ----------
    feature_path : str
        Path to feature directory containing .npy files
    label_path : str, optional
        Path to label directory (if you have labels)
    ch_names : list, optional
        List of channel names
    normalize : bool, optional
        Whether to apply min-max normalization to the data
    sfreq : float, optional
        Sampling frequency in Hz
        
    Returns
    -------
    epochs_list : list
        List of mne.Epochs objects
    labels_list : list
        List of labels (if label_path provided)
    subjects_list : list
        List of subject identifiers
    """
    
    # Define default channel names if not provided
    if ch_names is None:
        ch_names = STANDARD_CHANNELS
    
    # Load labels if provided
    subject_labels = {}
    if label_path and os.path.exists(os.path.join(label_path, 'label.npy')):
        labels_array = np.load(os.path.join(label_path, 'label.npy'))
        subject_labels = {int(entry[1]): int(entry[0]) for entry in labels_array}
        print(f"Loaded labels for {len(subject_labels)} subjects")
    
    # Get feature files
    feature_files = [f for f in os.listdir(feature_path) if f.startswith('feature_') and f.endswith('.npy')]
    feature_files.sort()
    
    # Initialize lists
    epochs_list = []
    labels_list = []
    subjects_list = []
    
    print(f"Found {len(feature_files)} feature files")
    
    # Process each feature file
    for feature_file in feature_files:
        print(f"\nProcessing: {feature_file}")
        
        # Extract subject ID
        try:
            subject_id = int(feature_file.split('_')[1].split('.')[0])
        except:
            subject_id = len(subjects_list) + 1  # fallback ID
        
                # Load feature data
        feature_data = np.load(os.path.join(feature_path, feature_file))
        print(f"  Original shape: {feature_data.shape}")
        
        # Debug: Check data statistics
        if debug:
            print(f"  Data statistics before processing:")
            print(f"    Min: {np.min(feature_data):.6f}")
            print(f"    Max: {np.max(feature_data):.6f}")
            print(f"    Mean: {np.mean(feature_data):.6f}")
            print(f"    Std: {np.std(feature_data):.6f}")
            print(f"    Contains NaN: {np.isnan(feature_data).any()}")
            print(f"    Contains Inf: {np.isinf(feature_data).any()}")
            
            # Check if data is all zeros or very close to zero
            if np.abs(np.max(feature_data) - np.min(feature_data)) < 1e-10:
                print(f"    WARNING: Data appears to be constant or near-zero!")
                
            # Show some sample values
            print(f"    Sample values from first epoch, first channel:")
            if len(feature_data.shape) == 3:
                sample_data = feature_data[0, :10, 0] if feature_data.shape[2] <= feature_data.shape[1] else feature_data[0, 0, :10]
                print(f"      {sample_data}")
        
        # Apply min-max normalization if requested
        if normalize:
            # Normalize each channel separately
            # Assuming feature_data shape is (epochs, times, channels)
            for epoch_idx in range(feature_data.shape[0]):
                for channel_idx in range(feature_data.shape[2]):
                    channel_data = feature_data[epoch_idx, :, channel_idx]
                    min_val = np.min(channel_data)
                    max_val = np.max(channel_data)
                    if max_val > min_val:  # Avoid division by zero
                        feature_data[epoch_idx, :, channel_idx] = (channel_data - min_val) / (max_val - min_val)
            print(f"  Applied min-max normalization")
            
            # Debug: Check data after normalization
            if debug:
                print(f"  Data statistics after normalization:")
                print(f"    Min: {np.min(feature_data):.6f}")
                print(f"    Max: {np.max(feature_data):.6f}")
                print(f"    Mean: {np.mean(feature_data):.6f}")
                print(f"    Std: {np.std(feature_data):.6f}")
        
        # Convert data shape if needed
        # Check if data is (epochs, times, channels) and convert to (epochs, channels, times)
        if len(feature_data.shape) == 3:
            if feature_data.shape[2] == 19 or feature_data.shape[2] <= feature_data.shape[1]:
                # Likely (epochs, times, channels) -> transpose to (epochs, channels, times)
                data = feature_data.transpose(0, 2, 1)
                print(f"  Transposed from (epochs, times, channels) to (epochs, channels, times)")
            else:
                # Already (epochs, channels, times)
                data = feature_data
                print(f"  Data already in (epochs, channels, times) format")
        else:
            data = feature_data
            print(f"  Using data as-is with shape: {data.shape}")
        
        print(f"  Final data shape: {data.shape}")
        
        # Debug: Check final data statistics
        if debug:
            print(f"  Final data statistics:")
            print(f"    Min: {np.min(data):.6f}")
            print(f"    Max: {np.max(data):.6f}")
            print(f"    Mean: {np.mean(data):.6f}")
            print(f"    Std: {np.std(data):.6f}")
            print(f"    Dynamic range: {np.max(data) - np.min(data):.6f}")
            
            # Check individual channel statistics
            if len(data.shape) == 3:
                print(f"  Channel-wise statistics (first epoch):")
                for ch_idx in range(min(5, data.shape[1])):  # Show first 5 channels
                    ch_data = data[0, ch_idx, :]
                    print(f"    Ch {ch_idx}: min={np.min(ch_data):.6f}, max={np.max(ch_data):.6f}, "
                          f"mean={np.mean(ch_data):.6f}, std={np.std(ch_data):.6f}")
        
        # Create info object
        n_channels = data.shape[1]
        used_ch_names = ch_names[:n_channels] if n_channels <= len(ch_names) else [f"ch{i}" for i in range(n_channels)]
        
        print(f"  Using {n_channels} channels: {used_ch_names}")
        
        # Create info
        info = mne.create_info(ch_names=used_ch_names, sfreq=sfreq, ch_types='eeg')
        
        # Create epochs object
        epochs = mne.EpochsArray(data, info)
        
        # Get label if available
        label = subject_labels.get(subject_id, -1)
        
        epochs_list.append(epochs)
        labels_list.append(label)
        subjects_list.append(f"subject_{subject_id}")
        
        print(f"  Created epochs object with {len(epochs)} epochs")
        if label != -1:
            print(f"  Label: {label} ({CLASS_NAMES.get(label, 'Unknown')})")
        
        # Debug: Final MNE epochs statistics
        if debug:
            epochs_data = epochs.get_data()
            print(f"  MNE epochs data statistics:")
            print(f"    Shape: {epochs_data.shape}")
            print(f"    Min: {np.min(epochs_data):.6f}")
            print(f"    Max: {np.max(epochs_data):.6f}")
            print(f"    Mean: {np.mean(epochs_data):.6f}")
            print(f"    Std: {np.std(epochs_data):.6f}")
            
            # Check if we have any signal variation
            if np.std(epochs_data) < 1e-10:
                print(f"    WARNING: Very low signal variation - waveforms may appear flat!")
                print(f"    Consider checking your data preprocessing or normalization.")
            else:
                print(f"    Signal variation looks good for plotting.")
    
    print(f"\nLoaded {len(epochs_list)} subjects total")
    
    # Print class distribution if labels available
    if any(l != -1 for l in labels_list):
        valid_labels = [l for l in labels_list if l != -1]
        classes, counts = np.unique(valid_labels, return_counts=True)
        print(f"Class distribution:")
        for c, count in zip(classes, counts):
            print(f"  {CLASS_NAMES.get(c, f'Class {c}')}: {count} subjects")
    
    return epochs_list, labels_list, subjects_list

def plot_epochs_mne_save(epochs, subject_name="", label=None, output_dir="plots", epoch_range=(0, 5)):
    """
    Plot epochs using MNE and save as PNG
    
    Parameters:
    -----------
    epochs : mne.Epochs
        The epochs to plot
    subject_name : str
        Subject identifier for title
    label : int, optional
        Subject label for title
    output_dir : str
        Directory to save plots
    epoch_range : tuple
        (start, end) epochs to plot
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create title
    title = f"EEG Epochs - {subject_name}"
    if label is not None and label != -1:
        title += f" ({CLASS_NAMES.get(label, f'Label {label}')})"
    
    # Set matplotlib backend to non-interactive
    import matplotlib
    matplotlib.use('Agg')
    
    # Create plot
    start_epoch, end_epoch = epoch_range
    end_epoch = min(end_epoch, len(epochs))
    
    fig = epochs[start_epoch:end_epoch].plot(
        n_epochs=end_epoch-start_epoch,
        n_channels=len(epochs.ch_names),
        title=title,
        show=False  # Don't show, we'll save instead
    )
    
    # Save the plot
    filename = f"eeg_epochs_{subject_name}"
    if label is not None and label != -1:
        filename += f"_label{label}"
    filename += f"_epochs{start_epoch}-{end_epoch}.png"
    
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved MNE plot: {filepath}")
    return filepath

def plot_epochs_matplotlib(epochs, epoch_idx=0, subject_name="", label=None, output_dir="plots"):
    """
    Plot a single epoch using matplotlib and save as PNG
    
    Parameters:
    -----------
    epochs : mne.Epochs
        The epochs to plot
    epoch_idx : int
        Index of the epoch to plot
    subject_name : str
        Subject identifier for title
    label : int, optional
        Subject label for title
    output_dir : str
        Directory to save plots
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set matplotlib backend to non-interactive
    import matplotlib
    matplotlib.use('Agg')
    
    data = epochs.get_data()[epoch_idx]  # Shape: (channels, times)
    times = epochs.times
    
    # Calculate figure size based on number of channels
    n_channels = data.shape[0]
    fig_height = max(12, n_channels * 0.8)  # Ensure adequate height per channel
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, fig_height))
    
    # Handle case where there's only one channel
    if n_channels == 1:
        axes = [axes]
    
    # Create title
    title = f"EEG Epoch {epoch_idx} - {subject_name}"
    if label is not None and label != -1:
        title += f" ({CLASS_NAMES.get(label, f'Label {label}')})"
    
    # Plot each channel
    for i, ch_name in enumerate(epochs.ch_names):
        axes[i].plot(times, data[i], linewidth=0.8)
        axes[i].set_ylabel(ch_name, fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', labelsize=8)
        
        if i == 0:
            axes[i].set_title(title, fontsize=12, pad=10)
        if i == n_channels - 1:
            axes[i].set_xlabel('Time (s)', fontsize=10)
    
    # Adjust layout manually instead of tight_layout
    plt.subplots_adjust(
        left=0.08,    # Left margin
        bottom=0.05,  # Bottom margin  
        right=0.95,   # Right margin
        top=0.95,     # Top margin
        hspace=0.4    # Height spacing between subplots
    )
    
    # Save the plot
    filename = f"eeg_single_epoch_{subject_name}_epoch{epoch_idx}"
    if label is not None and label != -1:
        filename += f"_label{label}"
    filename += ".png"
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved matplotlib plot: {filepath}")
    return filepath

def plot_single_subject(feature_path, subject_idx=0, label_path=None, normalize=False, output_dir="plots", debug=True):
    """
    Quick function to plot a single subject's data and save as PNG
    
    Parameters:
    -----------
    feature_path : str
        Path to feature directory
    subject_idx : int
        Index of subject to plot (0 = first subject)
    label_path : str, optional
        Path to label directory
    normalize : bool
        Whether to normalize the data
    output_dir : str
        Directory to save plots
    debug : bool
        Whether to show debugging information
    """
    
    # Load all data
    epochs_list, labels_list, subjects_list = load_npy_to_epochs(
        feature_path, label_path, normalize=normalize, debug=debug
    )
    
    if len(epochs_list) == 0:
        print("No data loaded!")
        return
    
    # Select subject
    if subject_idx >= len(epochs_list):
        subject_idx = 0
        print(f"Subject index too high, using subject 0")
    
    epochs = epochs_list[subject_idx]
    label = labels_list[subject_idx]
    subject_name = subjects_list[subject_idx]
    
    print(f"\nPlotting data for {subject_name}")
    print(f"Epochs shape: {epochs.get_data().shape}")
    print(f"Channels: {epochs.ch_names}")
    print(f"Sampling rate: {epochs.info['sfreq']} Hz")
    
    # Get data for additional debugging
    data = epochs.get_data()
    print(f"Data range: {np.min(data):.6f} to {np.max(data):.6f}")
    print(f"Data std: {np.std(data):.6f}")
    
    # Check if data seems problematic
    if np.std(data) < 1e-10:
        print("⚠️  WARNING: Data appears to have very low variation!")
        print("   This could be why waveforms aren't visible.")
        print("   Possible causes:")
        print("   - Data is already heavily normalized/standardized")
        print("   - Data contains mostly zeros or constants")
        print("   - Preprocessing removed signal variation")
        
        # Try to rescale for visualization
        print("\n🔧 Attempting to rescale data for better visualization...")
        # Check if we can amplify the signal
        if np.max(np.abs(data)) > 0:
            scale_factor = 1.0 / np.max(np.abs(data))
            print(f"   Applying scale factor: {scale_factor:.2e}")
            
            # Create a copy with scaled data
            scaled_data = data * scale_factor
            info = epochs.info.copy()
            epochs_scaled = mne.EpochsArray(scaled_data, info)
            
            print(f"   Scaled data range: {np.min(scaled_data):.6f} to {np.max(scaled_data):.6f}")
            print(f"   Scaled data std: {np.std(scaled_data):.6f}")
            
            # Use the scaled epochs for plotting
            epochs = epochs_scaled
            subject_name += "_scaled"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot using MNE and save
    print(f"\nSaving MNE plot to {output_dir}...")
    plot_epochs_mne_save(epochs, subject_name, label, output_dir, epoch_range=(0, 5))
    
    # Plot using matplotlib and save
    print(f"\nSaving matplotlib plot to {output_dir}...")
    plot_epochs_matplotlib(epochs, epoch_idx=0, subject_name=subject_name, label=label, output_dir=output_dir)
    
    # Additional diagnostic plot - show raw data values
    if debug:
        print(f"\nSaving diagnostic plot...")
        save_diagnostic_plot(epochs, subject_name, label, output_dir)
    
    print(f"\nAll plots saved in: {output_dir}")

def save_diagnostic_plot(epochs, subject_name="", label=None, output_dir="plots"):
    """
    Create a diagnostic plot showing data statistics and sample values
    """
    import matplotlib
    matplotlib.use('Agg')
    
    data = epochs.get_data()  # Shape: (epochs, channels, times)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Diagnostic Plot - {subject_name}", fontsize=14)
    
    # 1. Data distribution histogram
    axes[0, 0].hist(data.flatten(), bins=50, alpha=0.7)
    axes[0, 0].set_title('Data Value Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Channel-wise statistics
    channel_means = np.mean(data, axis=(0, 2))  # Mean across epochs and time
    channel_stds = np.std(data, axis=(0, 2))    # Std across epochs and time
    
    x_channels = range(len(channel_means))
    axes[0, 1].bar(x_channels, channel_means, alpha=0.7, label='Mean')
    axes[0, 1].set_title('Channel-wise Mean Values')
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Mean Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Channel-wise standard deviations
    axes[1, 0].bar(x_channels, channel_stds, alpha=0.7, color='orange', label='Std')
    axes[1, 0].set_title('Channel-wise Standard Deviations')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Sample waveform (first epoch, first channel)
    times = epochs.times
    sample_data = data[0, 0, :]  # First epoch, first channel
    axes[1, 1].plot(times, sample_data)
    axes[1, 1].set_title(f'Sample Waveform (Epoch 0, {epochs.ch_names[0]})')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Data Statistics:
    Shape: {data.shape}
    Min: {np.min(data):.6f}
    Max: {np.max(data):.6f}
    Mean: {np.mean(data):.6f}
    Std: {np.std(data):.6f}
    Range: {np.max(data) - np.min(data):.6f}
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for statistics text
    
    # Save the diagnostic plot
    filename = f"diagnostic_{subject_name}"
    if label is not None and label != -1:
        filename += f"_label{label}"
    filename += ".png"
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved diagnostic plot: {filepath}")
    return filepath

def plot_multiple_subjects(feature_path, label_path=None, normalize=False, output_dir="plots", max_subjects=5):
    """
    Plot multiple subjects and save all as PNG files
    
    Parameters:
    -----------
    feature_path : str
        Path to feature directory
    label_path : str, optional
        Path to label directory
    normalize : bool
        Whether to normalize the data
    output_dir : str
        Directory to save plots
    max_subjects : int
        Maximum number of subjects to plot
    """
    
    # Load all data
    epochs_list, labels_list, subjects_list = load_npy_to_epochs(
        feature_path, label_path, normalize=normalize
    )
    
    if len(epochs_list) == 0:
        print("No data loaded!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot up to max_subjects
    n_subjects = min(len(epochs_list), max_subjects)
    
    print(f"\nPlotting {n_subjects} subjects and saving to {output_dir}...")
    
    for i in range(n_subjects):
        epochs = epochs_list[i]
        label = labels_list[i]
        subject_name = subjects_list[i]
        
        print(f"\nProcessing {subject_name} ({i+1}/{n_subjects})...")
        
        # Save MNE plot (first 3 epochs)
        plot_epochs_mne_save(epochs, subject_name, label, output_dir, epoch_range=(0, 3))
        
        # Save matplotlib plot (first epoch)
        plot_epochs_matplotlib(epochs, epoch_idx=0, subject_name=subject_name, label=label, output_dir=output_dir)
    
    print(f"\nAll plots saved in: {output_dir}")

# ---------------------------
# Main Script
# ---------------------------

def main():
    """
    Main function - modify these paths to match your data
    """
    
    # MODIFY THESE PATHS TO MATCH YOUR DATA
    feature_path = '/home/stud/timlin/bhome/DiffusionEEG/dataset/LDM_PSD_Normalized_FIX_NO_SPEC/Feature'
    label_path = '/home/stud/timlin/bhome/DiffusionEEG/dataset/LDM_PSD_Normalized_FIX_NO_SPEC/Label'  # Set to None if no labels
    output_dir = 'images/eeg_plots/2LDM_PSD_Normalized_FIX_NO_SPEC'  # Directory where PNG files will be saved
    
    # Check if paths exist
    if not os.path.exists(feature_path):
        print(f"Feature path does not exist: {feature_path}")
        print("Please modify the feature_path variable to point to your .npy files")
        return
    
    print(f"Loading data from: {feature_path}")
    if label_path and os.path.exists(label_path):
        print(f"Loading labels from: {label_path}")
    else:
        print("No labels will be loaded")
        label_path = None
    
    print(f"Plots will be saved to: {output_dir}")
    
    # Plot single subject (change subject_idx to plot different subjects)
    plot_single_subject(
        feature_path=feature_path,
        subject_idx=0,  # Change this to plot different subjects
        label_path=label_path,
        normalize=False,  # Set to True if you want normalization
        output_dir=output_dir,
        debug=True  # Enable debugging to see data statistics
    )
    
    # Uncomment the line below to plot multiple subjects instead
    # plot_multiple_subjects(feature_path, label_path, normalize=False, output_dir=output_dir, max_subjects=3)

if __name__ == "__main__":
    main()