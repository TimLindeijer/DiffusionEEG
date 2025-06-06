#!/usr/bin/env python3
"""
Standalone script to reprocess existing .npy reconstruction files 
and generate extended PSD plots (0-35 Hz instead of 0-12 Hz).

Usage:
    python analyze_existing_files.py --input_dir /path/to/npy/files --sampling_freq 100 --max_freq 35
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from pathlib import Path
import re

def analyze_reconstruction_pair(original_file, recons_file, output_dir, 
                               sampling_freq=100, max_freq=35, nperseg=256):
    """
    Analyze a pair of original and reconstructed EEG files.
    """
    print(f"Analyzing: {original_file} vs {recons_file}")
    
    # Load the data
    try:
        original_data = np.load(original_file)
        recons_data = np.load(recons_file)
    except Exception as e:
        print(f"Error loading files: {e}")
        return None
    
    print(f"  Original shape: {original_data.shape}")
    print(f"  Reconstructed shape: {recons_data.shape}")
    
    # Take first sample if batch dimension exists
    if original_data.ndim == 3:  # [batch, channels, time]
        eeg_orig = original_data[0]  # [channels, time]
        eeg_recons = recons_data[0]  # [channels, time]
    else:  # [channels, time]
        eeg_orig = original_data
        eeg_recons = recons_data
    
    # Handle potential dimension issues
    if eeg_orig.ndim == 1:
        eeg_orig = eeg_orig.reshape(1, -1)
    if eeg_recons.ndim == 1:
        eeg_recons = eeg_recons.reshape(1, -1)
        
    print(f"  Processing shape: {eeg_orig.shape}")
    
    # Calculate PSDs
    try:
        freqs_orig, psds_orig = signal.welch(eeg_orig, fs=sampling_freq, 
                                           nperseg=nperseg, axis=-1)
        freqs_recons, psds_recons = signal.welch(eeg_recons, fs=sampling_freq, 
                                                nperseg=nperseg, axis=-1)
    except Exception as e:
        print(f"Error calculating PSDs: {e}")
        return None
    
    # Limit frequency range
    freq_mask = freqs_orig <= max_freq
    freqs_plot = freqs_orig[freq_mask]
    psds_orig_plot = psds_orig[:, freq_mask]
    psds_recons_plot = psds_recons[:, freq_mask]
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: All channels overlaid
    plt.subplot(3, 1, 1)
    n_channels = min(psds_orig_plot.shape[0], 19)  # Limit to 19 channels for clarity
    
    for ch in range(n_channels):
        plt.semilogy(freqs_plot, psds_orig_plot[ch], 'r-', alpha=0.7, linewidth=1)
        plt.semilogy(freqs_plot, psds_recons_plot[ch], 'b-', alpha=0.7, linewidth=1)
    
    # Add legend
    plt.semilogy([], [], 'r-', label='Original', linewidth=2)
    plt.semilogy([], [], 'b-', label='Reconstructed', linewidth=2)
    
    plt.ylabel('Power (μV²/Hz)', fontsize=12)
    plt.title(f'PSD Comparison: Original vs Reconstructed (0-{max_freq} Hz)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_freq)
    
    # Subplot 2: Average across channels
    plt.subplot(3, 1, 2)
    mean_orig = np.mean(psds_orig_plot, axis=0)
    mean_recons = np.mean(psds_recons_plot, axis=0)
    std_orig = np.std(psds_orig_plot, axis=0)
    std_recons = np.std(psds_recons_plot, axis=0)
    
    plt.semilogy(freqs_plot, mean_orig, 'r-', linewidth=2, label='Original (mean)')
    plt.fill_between(freqs_plot, mean_orig - std_orig, mean_orig + std_orig, 
                     color='red', alpha=0.2)
    plt.semilogy(freqs_plot, mean_recons, 'b-', linewidth=2, label='Reconstructed (mean)')
    plt.fill_between(freqs_plot, mean_recons - std_recons, mean_recons + std_recons, 
                     color='blue', alpha=0.2)
    
    plt.ylabel('Power (μV²/Hz)', fontsize=12)
    plt.title('Average PSD with Standard Deviation', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_freq)
    
    # Subplot 3: Difference analysis
    plt.subplot(3, 1, 3)
    # Calculate relative error
    relative_error = np.abs(psds_recons_plot - psds_orig_plot) / (psds_orig_plot + 1e-10)
    mean_error = np.mean(relative_error, axis=0)
    std_error = np.std(relative_error, axis=0)
    
    plt.plot(freqs_plot, mean_error, 'g-', linewidth=2, label='Mean Relative Error')
    plt.fill_between(freqs_plot, mean_error - std_error, mean_error + std_error, 
                     color='green', alpha=0.2)
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Relative Error', fontsize=12)
    plt.title('Reconstruction Error by Frequency', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_freq)
    
    plt.tight_layout()
    
    # Save the plot
    file_id = Path(original_file).stem.replace('original_', '')
    save_path = output_dir / f"extended_psd_analysis_{file_id}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical data
    np.save(output_dir / f"frequencies_{file_id}.npy", freqs_plot)
    np.save(output_dir / f"original_psds_{file_id}.npy", psds_orig_plot)
    np.save(output_dir / f"reconstructed_psds_{file_id}.npy", psds_recons_plot)
    
    print(f"  Analysis saved to {save_path}")
    print(f"  Frequency range: {freqs_plot[0]:.2f} - {freqs_plot[-1]:.2f} Hz")
    
    return freqs_plot, psds_orig_plot, psds_recons_plot

def find_file_pairs(input_dir):
    """
    Find matching pairs of original and reconstructed files.
    """
    input_path = Path(input_dir)
    
    # Find all original files
    original_files = list(input_path.glob("original_*.npy"))
    
    # Find corresponding reconstruction files
    pairs = []
    
    for orig_file in original_files:
        # Extract identifier from original filename
        orig_name = orig_file.stem
        # Remove 'original_' prefix to get the identifier
        identifier = orig_name.replace('original_', '')
        
        # Look for corresponding reconstruction file
        # Try different naming patterns
        possible_recons_names = [
            f"reconstr_{identifier}.npy",
            f"recons_{identifier}.npy", 
            f"reconstructed_{identifier}.npy",
            f"reconstruction_{identifier}.npy"
        ]
        
        recons_file = None
        for recons_name in possible_recons_names:
            potential_file = input_path / recons_name
            if potential_file.exists():
                recons_file = potential_file
                break
        
        # If exact match not found, try pattern matching
        if recons_file is None:
            recons_files = list(input_path.glob("reconstr*.npy")) + list(input_path.glob("recons*.npy"))
            for candidate in recons_files:
                if identifier in candidate.stem:
                    recons_file = candidate
                    break
        
        if recons_file is not None:
            pairs.append((orig_file, recons_file))
            print(f"Found pair: {orig_file.name} <-> {recons_file.name}")
        else:
            print(f"No reconstruction file found for {orig_file.name}")
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Analyze existing EEG reconstruction files with extended frequency range")
    
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing the .npy files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: input_dir/extended_analysis)")
    parser.add_argument("--sampling_freq", type=float, default=100.0,
                       help="Sampling frequency in Hz (default: 100)")
    parser.add_argument("--max_freq", type=float, default=35.0,
                       help="Maximum frequency for analysis in Hz (default: 35)")
    parser.add_argument("--nperseg", type=int, default=256,
                       help="Window size for PSD calculation (default: 256)")
    
    args = parser.parse_args()
    
    # Set up directories
    input_dir = Path(args.input_dir)
    if args.output_dir is None:
        output_dir = input_dir / "extended_analysis"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sampling frequency: {args.sampling_freq} Hz")
    print(f"Maximum frequency: {args.max_freq} Hz")
    print(f"Window size: {args.nperseg}")
    print("-" * 50)
    
    # Find file pairs
    pairs = find_file_pairs(input_dir)
    
    if not pairs:
        print("No matching file pairs found!")
        print("Make sure you have files named like:")
        print("  - original_XXXXX.npy")
        print("  - reconstr_XXXXX.npy (or recons_XXXXX.npy)")
        return
    
    print(f"Found {len(pairs)} file pairs to analyze")
    print("-" * 50)
    
    # Analyze each pair
    for i, (orig_file, recons_file) in enumerate(pairs, 1):
        print(f"Processing pair {i}/{len(pairs)}")
        try:
            analyze_reconstruction_pair(
                original_file=orig_file,
                recons_file=recons_file,
                output_dir=output_dir,
                sampling_freq=args.sampling_freq,
                max_freq=args.max_freq,
                nperseg=args.nperseg
            )
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            continue
        print()
    
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()