#!/usr/bin/env python
"""
Optimized script to generate synthetic EEG data with the same structure as the original dataset.
This version is adapted for PURE diffusion models (no autoencoder) and includes significant 
performance optimizations, proper PSD scaling, and clear progress tracking.
Modified to only generate synthetic data for existing feature files.
"""

import os
import argparse
import numpy as np
import torch
import torch.cuda.amp as amp
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import time
import json
import gc
import sys
from scipy import signal
import matplotlib.pyplot as plt

# Import necessary model components
print("START OF SCRIPT: Importing required modules...")
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer
from models.ldm import UNetModel
from omegaconf import OmegaConf
print("All modules imported successfully")

def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            used_mem = total_mem - free_mem
            return f"GPU Memory: {used_mem / 1024**2:.1f}MB used / {total_mem / 1024**2:.1f}MB total"
        except:
            return "GPU memory info not available"
    else:
        return "CUDA not available"

def parse_args():
    print("Parsing command line arguments...")
    parser = argparse.ArgumentParser(description="Generate synthetic EEG data matching original dataset structure")
    
    # Model paths for each category (pure diffusion models)
    parser.add_argument("--hc_model_path", type=str, required=True, 
                      help="Path to the healthy controls diffusion model weights")
    parser.add_argument("--mci_model_path", type=str, required=True,
                      help="Path to the MCI diffusion model weights")
    parser.add_argument("--dementia_model_path", type=str, required=True, 
                      help="Path to the dementia diffusion model weights")
    
    # Config files
    parser.add_argument("--diffusion_config", type=str, required=True,
                      help="Path to the diffusion model config file")
    
    # Original dataset information
    parser.add_argument("--original_label_path", type=str, required=True,
                      help="Path to the original label.npy file (e.g., dataset/CAUEEG2/Label/label.npy)")
    parser.add_argument("--original_data_path", type=str, required=True,
                       help="Path to original dataset features (e.g., dataset/CAUEEG2/Feature)")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="synthetic_eeg_matched",
                      help="Directory to save the generated samples")
    
    # Performance optimization parameters
    parser.add_argument("--diffusion_steps", type=int, default=100,
                      help="Number of diffusion steps (fewer = faster)")
    parser.add_argument("--batch_epochs", type=int, default=16,
                      help="Number of epochs to generate in a single batch")
    parser.add_argument("--category", type=str, choices=["hc", "mci", "dementia", "all"], default="all",
                      help="Which category to generate (for parallel jobs)")
    
    # Generation parameters
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                      help="Use GPU for generation if available")
    
    # Fixed dimensions (adjust based on your training data)
    parser.add_argument("--num_epochs", type=int, default=71,
                      help="Number of epochs per sample")
    parser.add_argument("--num_channels", type=int, default=19,
                      help="Number of EEG channels")
    parser.add_argument("--num_timepoints", type=int, default=6000,
                      help="Number of timepoints per epoch")
    
    # PSD scaling parameters
    parser.add_argument("--normalize_psd", action="store_true", default=True,
                      help="Apply PSD normalization to match real data statistics")
    parser.add_argument("--per_channel", action="store_true", default=True,
                      help="Apply normalization per EEG channel separately")
    parser.add_argument("--plot_psd", action="store_true", default=True,
                      help="Plot and save PSD comparisons during generation")
    parser.add_argument("--reference_sample_count", type=int, default=5,
                      help="Number of real samples to use as reference for PSD matching")
    parser.add_argument("--sampling_rate", type=int, default=200,
                      help="EEG sampling rate in Hz")
    
    args = parser.parse_args()
    print(f"Arguments parsed successfully: Category={args.category}, Diffusion steps={args.diffusion_steps}, Batch epochs={args.batch_epochs}")
    return args

def ensure_correct_data_format(data, expected_channels=19, expected_timepoints=6000):
    """
    Ensure data is in the expected format (epochs, channels, timepoints).
    Will automatically detect and transpose if needed.
    """
    # First check if dimensions are already as expected
    if data.ndim == 3:
        # Check if transposition is needed
        if data.shape[1] == expected_timepoints and data.shape[2] == expected_channels:
            print(f"Detected data in shape {data.shape}, transposing to format (epochs, channels, timepoints)")
            return np.transpose(data, (0, 2, 1))
        elif data.shape[1] == expected_channels and data.shape[2] == expected_timepoints:
            print(f"Data already in correct format with shape {data.shape}")
            return data
        else:
            print(f"Warning: Unusual dimensions {data.shape}, attempting to infer correct format")
            # Try to infer format - channels should be smaller than timepoints
            if data.shape[1] < data.shape[2]:
                return data  # Likely already in (epochs, channels, timepoints)
            else:
                return np.transpose(data, (0, 2, 1))  # Convert from (epochs, timepoints, channels)
    else:
        print(f"Warning: Expected 3D array but got {data.ndim}D array with shape {data.shape}")
        return data

def read_original_labels(label_path, data_path):
    """
    Read the original label.npy file to extract feature numbering and labels.
    Only include feature IDs that have corresponding feature files.
    """
    print(f"Reading original labels from: {label_path}")
    print(f"Checking for existing feature files in: {data_path}")
    
    try:
        # Load the original labels
        labels = np.load(label_path)
        print(f"Successfully loaded original label file with shape: {labels.shape}")
        
        # Group by label and check file existence
        label_to_features = defaultdict(list)
        missing_files = []
        
        for entry in labels:
            label = int(entry[0])  # First column is label
            subject_id = int(entry[1])  # Second column is subject_id
            
            # Check if the feature file exists
            feature_filename = f"feature_{subject_id:02d}.npy"
            feature_path = os.path.join(data_path, feature_filename)
            
            if os.path.exists(feature_path):
                label_to_features[label].append(subject_id)
            else:
                missing_files.append((label, subject_id, feature_path))
        
        # Print summary
        print(f"\nExisting files summary:")
        print(f"  HC (label 0): {len(label_to_features[0])} files exist")
        print(f"  MCI (label 1): {len(label_to_features[1])} files exist")
        print(f"  Dementia (label 2): {len(label_to_features[2])} files exist")
        
        if missing_files:
            print(f"\nSkipping {len(missing_files)} missing files:")
            for label, subject_id, path in missing_files[:10]:  # Show first 10
                label_name = ['HC', 'MCI', 'Dementia'][label]
                print(f"  - {label_name} (label {label}): feature_{subject_id:02d}.npy")
            if len(missing_files) > 10:
                print(f"  ... and {len(missing_files) - 10} more")
        
        return label_to_features
    except Exception as e:
        print(f"ERROR loading labels: {str(e)}")
        raise

def load_reference_data(data_path, label_to_features, label, num_samples=5):
    """
    Load reference data samples with handling for varying epoch counts.
    """
    print(f"Loading {num_samples} reference samples for label {label}...")
    
    # Sample feature IDs from this label category
    feature_ids = label_to_features[label]
    if len(feature_ids) > num_samples:
        selected_ids = np.random.choice(feature_ids, num_samples, replace=False)
    else:
        selected_ids = feature_ids
        print(f"Warning: Only {len(feature_ids)} samples available for label {label}")
    
    # Load the selected features
    all_epochs = []
    for feature_id in selected_ids:
        feature_path = os.path.join(data_path, f"feature_{feature_id:02d}.npy")
        try:
            # Load the feature data
            feature_data = np.load(feature_path)
            
            # Check feature shape and ensure correct format
            feature_data = ensure_correct_data_format(feature_data)
            
            print(f"Loaded reference feature_{feature_id:02d}.npy with shape {feature_data.shape}")
            
            # Add all epochs from this feature
            for epoch in range(feature_data.shape[0]):
                all_epochs.append(feature_data[epoch])
        except Exception as e:
            print(f"Error loading reference data for feature {feature_id}: {str(e)}")
    
    # Check if we have loaded any data
    if len(all_epochs) == 0:
        print("ERROR: No reference data could be loaded!")
        return None
    
    # Stack all epochs into a single array
    reference_data = np.stack(all_epochs)
    print(f"Combined reference data shape: {reference_data.shape} (total epochs from all samples)")
    return reference_data

def calculate_psd(data, fs=200, nperseg=512):
    """
    Calculate Power Spectral Density (PSD) of EEG data in dB.
    """
    # Make sure data is numpy array
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # Get original shape
    orig_shape = data.shape
    
    # Reshape to [total_samples, timepoints]
    data_reshaped = data.reshape(-1, orig_shape[-1])
    
    # Calculate PSD for each sample
    psds = []
    for i in range(data_reshaped.shape[0]):
        f, pxx = signal.welch(data_reshaped[i], fs=fs, nperseg=nperseg)
        # Convert to dB
        pxx_db = 10 * np.log10(pxx + 1e-10)  # Add small constant to avoid log(0)
        psds.append(pxx_db)
    
    # Average across all samples
    avg_psd = np.mean(np.array(psds), axis=0)
    
    return f, avg_psd

def plot_psd_comparison(real_f, real_psd, syn_f, syn_psd, norm_f, norm_psd, category, output_dir):
    """
    Plot and save a comparison of PSDs between real, synthetic and normalized data.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(real_f, real_psd, 'b-', linewidth=2, label='Real Data')
    plt.plot(syn_f, syn_psd, 'r--', linewidth=2, label='Synthetic Data (Raw)')
    plt.plot(norm_f, norm_psd, 'g-', linewidth=2, label='Synthetic Data (Normalized)')
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set labels and title
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Power (dB μV²/Hz)', fontsize=14)
    plt.title(f'PSD Comparison for {category.upper()} Data', fontsize=16)
    
    # Limit x-axis to relevant frequencies (e.g., 0-30 Hz for EEG)
    plt.xlim(0, 30)
    
    # Create plot directory if it doesn't exist
    plot_dir = os.path.join(output_dir, 'psd_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join(plot_dir, f'psd_comparison_{category}_{timestamp}.png')
    
    # Save the plot
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"PSD comparison plot saved to {plot_filename}")
    plt.close()

def normalize_to_reference(synthetic_data, reference_data, per_channel=True):
    """
    Normalize synthetic data to match the statistical properties of reference data.
    """
    print(f"Synthetic data shape: {synthetic_data.shape}, Reference data shape: {reference_data.shape}")
    print("Applying statistical normalization to match real data properties...")
    
    # Create a copy of synthetic data to modify
    normalized_data = synthetic_data.copy()
    
    if per_channel:
        # Normalize each channel separately
        print("Normalizing each EEG channel separately...")
        for c in range(synthetic_data.shape[1]):
            # Calculate statistics for this channel
            ref_mean = reference_data[:, c].mean()
            ref_std = reference_data[:, c].std()
            syn_mean = synthetic_data[:, c].mean()
            syn_std = synthetic_data[:, c].std()
            
            # Check for extremely small standard deviations
            if syn_std < 1e-10:
                print(f"Warning: Very small std in synthetic data channel {c}: {syn_std}")
                syn_std = 1e-10
            
            # Normalize synthetic data to match reference statistics
            normalized_data[:, c] = ((synthetic_data[:, c] - syn_mean) / syn_std) * ref_std + ref_mean
            
            # Log the changes
            print(f"Channel {c}: Changed mean from {syn_mean:.2f} to {ref_mean:.2f}, " +
                  f"std from {syn_std:.2f} to {ref_std:.2f}")
    else:
        # Normalize the entire data at once
        print("Normalizing all EEG channels together...")
        ref_mean = reference_data.mean()
        ref_std = reference_data.std()
        syn_mean = synthetic_data.mean()
        syn_std = synthetic_data.std()
        
        # Check for extremely small standard deviations
        if syn_std < 1e-10:
            print(f"Warning: Very small std in synthetic data: {syn_std}")
            syn_std = 1e-10
        
        # Normalize synthetic data to match reference statistics
        normalized_data = ((synthetic_data - syn_mean) / syn_std) * ref_std + ref_mean
        
        # Log the changes
        print(f"Overall: Changed mean from {syn_mean:.2f} to {ref_mean:.2f}, " +
              f"std from {syn_std:.2f} to {ref_std:.2f}")
    
    return normalized_data

def load_checkpoint(output_dir, category):
    """Load checkpoint to resume generation from where it left off."""
    print(f"Checking for checkpoint for category: {category}")
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{category}.json")
    completed_ids = []
    
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                completed_ids = checkpoint_data.get('completed_ids', [])
                print(f"Checkpoint found: {len(completed_ids)} samples already generated")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
    
    return completed_ids

def save_checkpoint(output_dir, category, completed_ids):
    """Save checkpoint of completed feature IDs."""
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{category}.json")
    checkpoint_data = {
        'completed_ids': completed_ids,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)

def load_diffusion_model(config_path, model_path, device):
    """Load the pure diffusion model."""
    print(f"Loading diffusion model config from: {config_path}")
    
    try:
        # Load diffusion config
        config = OmegaConf.load(config_path)
        print("Successfully loaded diffusion config")
        
        # Extract UNet parameters from config
        print("Extracting UNet parameters...")
        if hasattr(config, 'model') and hasattr(config.model, 'params') and hasattr(config.model.params, 'unet_config') and hasattr(config.model.params.unet_config, 'params'):
            parameters = dict(config.model.params.unet_config.params)
            print("Parameters successfully extracted from config")
        else:
            # Fallback to basic parameters
            print("WARNING: Couldn't find full UNet parameters in config, using defaults")
            parameters = {
                'model_channels': 64,
                'num_res_blocks': 2,
                'attention_resolutions': [4],
                'channel_mult': [1, 2, 4],
                'conv_resample': True,
                'num_heads': 1,
                'use_scale_shift_norm': False,
                'resblock_updown': True,
                'dropout': 0.0
            }
        
        # Set channels for direct EEG generation (no latent space)
        parameters['in_channels'] = 19
        parameters['out_channels'] = 19
        
        print(f"Diffusion model configuration: {parameters}")
        
        # Create diffusion model
        print(f"Creating diffusion model instance...")
        diffusion = UNetModel(**parameters)
        print("Diffusion model created successfully")
        
        # Load weights
        print(f"Loading diffusion model weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel wrapper
        if any(key.startswith('module.') for key in state_dict.keys()):
            # Remove 'module.' prefix
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        diffusion.load_state_dict(state_dict)
        print("Weights loaded successfully, applying to model...")
        print(f"Moving diffusion model to device: {device}")
        diffusion = diffusion.to(device)
        diffusion.eval()
        print(f"Diffusion model ready on device: {device}")
        
        # Create scheduler
        print("Creating diffusion scheduler...")
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            schedule="linear_beta",
            beta_start=0.0015,
            beta_end=0.0195
        )
        scheduler = scheduler.to(device)
        print("Scheduler created and moved to device")
        
        # Create inferer
        inferer = DiffusionInferer(scheduler)
        
        print(f"Current GPU memory: {get_gpu_memory()}")
        
        return diffusion, scheduler, inferer
    except Exception as e:
        print(f"ERROR loading diffusion model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def generate_sample_batch(diffusion_model, scheduler, inferer, num_epochs, channels, timepoints, 
                       batch_size, diffusion_steps, reference_data=None, normalize=True, 
                       per_channel=True, plot_psd=False, category=None, output_dir=None, fs=200):
    """
    Generate a batch of epochs using pure diffusion model.
    """
    device = next(diffusion_model.parameters()).device
    use_autocast = torch.cuda.is_available()
    
    # Create container for all epochs
    sample = np.zeros((num_epochs, channels, timepoints))
    
    # Process in batches for memory efficiency
    for i in range(0, num_epochs, batch_size):
        batch_size_current = min(batch_size, num_epochs - i)
        
        try:
            with torch.no_grad():
                # Generate random noise for this batch
                noise_shape = (batch_size_current, channels, timepoints)
                noise = torch.randn(noise_shape, device=device)
                
                print(f"  Generating batch {i//batch_size + 1}, epochs {i+1}-{i+batch_size_current}")
                
                # Use the inferer to generate samples
                if use_autocast:
                    with amp.autocast():
                        generated = inferer.sample(
                            input_noise=noise,
                            diffusion_model=diffusion_model,
                            scheduler=scheduler,
                            save_intermediates=False,
                            intermediate_steps=diffusion_steps
                        )
                else:
                    generated = inferer.sample(
                        input_noise=noise,
                        diffusion_model=diffusion_model,
                        scheduler=scheduler,
                        save_intermediates=False,
                        intermediate_steps=diffusion_steps
                    )
                
                # Move to CPU and convert to numpy
                generated_np = generated.cpu().float().numpy()
                
                # Add to sample
                for j in range(batch_size_current):
                    if i + j < num_epochs:
                        sample[i + j] = generated_np[j]
                
                # Cleanup to free memory
                del noise, generated
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"ERROR in generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    print(f"Raw sample generation completed, shape: {sample.shape}")
    
    # Ensure sample has the correct format (epochs, channels, timepoints)
    sample = ensure_correct_data_format(sample, expected_channels=channels, expected_timepoints=timepoints)
    
    # Calculate and save raw PSD before normalization
    if plot_psd and reference_data is not None and category is not None and output_dir is not None:
        print("Calculating PSD for raw synthetic data...")
        syn_f, syn_psd = calculate_psd(sample, fs=fs)
        print("Calculating PSD for reference data...")
        ref_f, ref_psd = calculate_psd(reference_data, fs=fs)
    
    # Apply normalization if requested and reference data is available
    if normalize and reference_data is not None:
        print("Applying normalization to match reference data statistics...")
        
        # Ensure reference data is in the correct format
        reference_data = ensure_correct_data_format(reference_data, expected_channels=channels, expected_timepoints=timepoints)
        
        normalized_sample = normalize_to_reference(sample, reference_data, per_channel=per_channel)
        
        # Calculate and plot PSD after normalization
        if plot_psd and category is not None and output_dir is not None:
            print("Calculating PSD for normalized synthetic data...")
            norm_f, norm_psd = calculate_psd(normalized_sample, fs=fs)
            print("Plotting PSD comparison...")
            plot_psd_comparison(ref_f, ref_psd, syn_f, syn_psd, norm_f, norm_psd, 
                             category, output_dir)
        
        print("Normalization completed")
        return normalized_sample
    else:
        if normalize and reference_data is None:
            print("WARNING: Normalization requested but no reference data provided. Returning raw sample.")
        return sample

def save_sample(sample, feature_id, output_dir):
    """Save a single generated sample with a specific feature ID."""
    feature_dir = os.path.join(output_dir, "Feature")
    os.makedirs(feature_dir, exist_ok=True)
    
    # Create filename with the exact same format as original
    filename = f"feature_{feature_id:02d}.npy"
    filepath = os.path.join(feature_dir, filename)
    
    # Save the sample
    try:
        np.save(filepath, sample)
    except Exception as e:
        print(f"ERROR saving sample: {str(e)}")
        raise
    
    return feature_id

def save_labels(label_to_features, output_dir):
    """Save the label file that matches the original structure."""
    print("Saving label file to match original structure")
    label_dir = os.path.join(output_dir, "Label")
    os.makedirs(label_dir, exist_ok=True)
    
    # Create label data array
    label_data = []
    for label, feature_ids in label_to_features.items():
        for feature_id in feature_ids:
            label_data.append([label, feature_id])
    
    # Convert to numpy array and save
    label_data = np.array(label_data)
    label_path = os.path.join(label_dir, "label.npy")
    
    try:
        np.save(label_path, label_data)
        print(f"Labels saved successfully to {label_path}")
    except Exception as e:
        print(f"ERROR saving labels: {str(e)}")
        raise

def generate_category(category_name, label, feature_ids, diffusion, scheduler, inferer, 
                     output_dir, args, reference_data=None, completed_ids=None):
    """Generate all samples for a specific category."""
    print(f"\n===== STARTING GENERATION FOR CATEGORY: {category_name} =====")
    
    if completed_ids is None:
        completed_ids = []
    
    # Filter out already completed feature IDs
    pending_ids = [fid for fid in feature_ids if fid not in completed_ids]
    print(f"Generating {len(pending_ids)}/{len(feature_ids)} remaining {category_name} samples")
    
    # Track timing information
    start_time = time.time()
    completed_this_run = []
    
    # Generate each pending sample
    for i, feature_id in enumerate(pending_ids):
        # Generate the sample
        try:
            sample_start_time = time.time()
            
            sample = generate_sample_batch(
                diffusion, scheduler, inferer,
                args.num_epochs, args.num_channels, args.num_timepoints,
                args.batch_epochs, args.diffusion_steps,
                reference_data=reference_data,
                normalize=args.normalize_psd,
                per_channel=args.per_channel,
                plot_psd=args.plot_psd and i == 0,  # Only plot for first sample to save time
                category=category_name,
                output_dir=output_dir,
                fs=args.sampling_rate
            )
            
            sample_generation_time = time.time() - sample_start_time
            
            # Save the sample
            save_sample(sample, feature_id, output_dir)
            completed_this_run.append(feature_id)
            
            # Update and save checkpoint every sample
            save_checkpoint(output_dir, category_name, completed_ids + completed_this_run)
            
            # Calculate and display ETA with clear progress indicator
            elapsed = time.time() - start_time
            samples_per_hour = (i + 1) / (elapsed / 3600)
            remaining = len(pending_ids) - (i + 1)
            eta_hours = remaining / samples_per_hour if samples_per_hour > 0 else 0
            
            # More visible progress report with percentage
            progress_pct = ((i + 1) / len(pending_ids)) * 100
            print("\n" + "=" * 80)
            print(f"🔄 PROGRESS: {progress_pct:.1f}% ({i+1}/{len(pending_ids)}) {category_name} samples")
            print(f"⏱️ Speed: {samples_per_hour:.2f} samples/hour | ETA: {eta_hours:.1f} hours")
            print(f"🆔 Last completed: feature_{feature_id:02d}.npy | Generation time: {sample_generation_time:.2f}s")
            print(f"🧠 {get_gpu_memory()}")
            print("=" * 80 + "\n")
            
            # Force garbage collection
            del sample
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"ERROR generating sample {feature_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\nCompleted {len(completed_this_run)} {category_name} samples " + 
          f"in {total_time/3600:.2f} hours")
    print(f"===== FINISHED GENERATION FOR CATEGORY: {category_name} =====\n")
    
    return completed_ids + completed_this_run

def main():
    print("========== PURE LDM GENERATION SCRIPT STARTING ==========")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    print(f"Setting random seed to {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    print(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read original labels to get structure (only includes existing files)
    print("Reading original labels and checking for existing files...")
    label_to_features = read_original_labels(args.original_label_path, args.original_data_path)
    
    # Save labels file immediately (it's the same regardless of which samples we generate)
    print("Saving labels file...")
    save_labels(label_to_features, args.output_dir)
    
    # Load reference data for PSD matching if normalization is enabled
    reference_data = {}
    if args.normalize_psd:
        print("Loading reference data for PSD normalization...")
        for label, category in zip([0, 1, 2], ["hc", "mci", "dementia"]):
            if args.category == "all" or args.category == category:
                reference_data[label] = load_reference_data(
                    args.original_data_path, 
                    label_to_features, 
                    label, 
                    args.reference_sample_count
                )
    
    # Determine which categories to generate
    print(f"Determining categories to generate based on --category={args.category}")
    categories_to_generate = []
    if args.category == "all" or args.category == "hc":
        categories_to_generate.append(("hc", 0, label_to_features[0]))
    if args.category == "all" or args.category == "mci":
        categories_to_generate.append(("mci", 1, label_to_features[1]))
    if args.category == "all" or args.category == "dementia":
        categories_to_generate.append(("dementia", 2, label_to_features[2]))
    
    print(f"Will generate {len(categories_to_generate)} categories: {[c[0] for c in categories_to_generate]}")
    
    # Process each category
    for category_name, label, feature_ids in categories_to_generate:
        print(f"\n========== PROCESSING CATEGORY: {category_name} ==========")
        
        # Load checkpoint
        completed_ids = load_checkpoint(args.output_dir, category_name)
        
        # Check if all samples for this category are already completed
        if len(completed_ids) >= len(feature_ids):
            print(f"All {category_name} samples already generated, skipping")
            continue
        
        # Load models for this category
        print(f"\nLoading {category_name} models...")
        if category_name == "hc":
            print("Starting to load HC diffusion model")
            diffusion, scheduler, inferer = load_diffusion_model(
                args.diffusion_config,
                args.hc_model_path,
                device
            )
        elif category_name == "mci":
            print("Starting to load MCI diffusion model")
            diffusion, scheduler, inferer = load_diffusion_model(
                args.diffusion_config,
                args.mci_model_path,
                device
            )
        elif category_name == "dementia":
            print("Starting to load Dementia diffusion model")
            diffusion, scheduler, inferer = load_diffusion_model(
                args.diffusion_config,
                args.dementia_model_path,
                device
            )
        
        print(f"Models loaded successfully for {category_name}")
        print(f"Current GPU memory after loading models: {get_gpu_memory()}")
        
        # Get reference data for this category if available
        category_reference_data = reference_data.get(label) if args.normalize_psd else None
        
        # Generate samples for this category
        print(f"Starting generation for {category_name}...")
        generate_category(
            category_name, label, feature_ids, diffusion, scheduler, inferer, 
            args.output_dir, args, category_reference_data, completed_ids
        )
        
        # Clear GPU memory
        print(f"Clearing GPU memory after {category_name} generation")
        del diffusion, scheduler, inferer
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"\n========== GENERATION COMPLETE ==========")
    print(f"Synthetic dataset saved to {args.output_dir}")
    print(f"Script execution completed successfully")

if __name__ == "__main__":
    print("Script executed as main program")
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)