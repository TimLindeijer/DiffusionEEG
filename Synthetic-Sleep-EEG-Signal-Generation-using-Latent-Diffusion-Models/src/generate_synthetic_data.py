#!/usr/bin/env python
"""
Generate synthetic EEG data using trained Latent Diffusion Models.
This script loads trained models for different categories (HC, MCI, Dementia)
and generates synthetic EEG data with matching labels.
The output shape matches the original data structure (epochs, channels, timepoints).
"""

import os
import argparse
import numpy as np
import torch
import torch.cuda.amp as amp
from pathlib import Path
from tqdm import tqdm
import random
import glob
import datetime

# Import necessary model components
from generative.networks.nets import AutoencoderKL
from generative.networks.schedulers import DDPMScheduler
from models.ldm import UNetModel
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic EEG data from trained models")
    
    # Model paths
    parser.add_argument("--hc_model_path", type=str, required=True, 
                      help="Path to the healthy controls model weights")
    parser.add_argument("--mci_model_path", type=str, required=True,
                      help="Path to the MCI model weights")
    parser.add_argument("--dementia_model_path", type=str, required=True, 
                      help="Path to the dementia model weights")
    
    # Autoencoder paths
    parser.add_argument("--hc_autoencoder_path", type=str, required=True,
                      help="Path to the healthy controls autoencoder weights")
    parser.add_argument("--mci_autoencoder_path", type=str, required=True,
                      help="Path to the MCI autoencoder weights")
    parser.add_argument("--dementia_autoencoder_path", type=str, required=True,
                      help="Path to the dementia autoencoder weights")
    
    # Config files
    parser.add_argument("--autoencoder_config", type=str, required=True,
                      help="Path to the autoencoder config file")
    parser.add_argument("--diffusion_config", type=str, required=True,
                      help="Path to the diffusion model config file")
    
    # Number of samples to generate
    parser.add_argument("--num_hc", type=int, default=100,
                      help="Number of healthy control samples to generate")
    parser.add_argument("--num_mci", type=int, default=100,
                      help="Number of MCI samples to generate")
    parser.add_argument("--num_dementia", type=int, default=100,
                      help="Number of dementia samples to generate")
    
    # Original data directory for mean length calculation
    parser.add_argument("--original_data_dir", type=str, required=True,
                      help="Directory containing the original EEG data for length calculation")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="synthetic_eeg_data",
                      help="Directory to save the generated samples")
    
    # Original label file for reference structure (optional)
    parser.add_argument("--reference_label_file", type=str, default=None,
                      help="Path to original label.npy file to match structure")
    
    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                      help="Use GPU for generation if available")
    parser.add_argument("--fixed_length", type=int, default=None,
                      help="Fixed number of epochs to use instead of calculating mean (optional)")
    parser.add_argument("--timepoints", type=int, default=1000,
                      help="Number of timepoints in each epoch")
    
    return parser.parse_args()

def interpolate_timepoints(samples, target_timepoints=928):
    """
    Resize temporal dimension of generated samples using linear interpolation.
    
    Args:
        samples: Numpy array with shape (batch_size, channels, timepoints)
        target_timepoints: Desired number of timepoints in output
        
    Returns:
        Resized samples with shape (batch_size, channels, target_timepoints)
    """
    current_timepoints = samples.shape[2]
    if current_timepoints == target_timepoints:
        return samples
        
    # Create output array
    resized = np.zeros((samples.shape[0], samples.shape[1], target_timepoints))
    
    # Interpolate each channel for each sample
    for b in range(samples.shape[0]):
        for c in range(samples.shape[1]):
            # Linear interpolation
            resized[b, c] = np.interp(
                np.linspace(0, current_timepoints-1, target_timepoints),
                np.arange(current_timepoints),
                samples[b, c]
            )
    return resized

def load_autoencoder(config_path, model_path, device):
    """Load the trained autoencoder model."""
    config = OmegaConf.load(config_path)
    autoencoder_args = config.autoencoderkl.params
    
    # Create and load the autoencoder
    autoencoder = AutoencoderKL(**autoencoder_args)
    state_dict = torch.load(model_path, map_location=device)
    autoencoder.load_state_dict(state_dict)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    
    # Calculate scale factor based on a random input
    # This is important for correctly scaling the latent space
    with torch.no_grad():
        # Create a random input tensor with the expected shape
        # Note: Adjust shape as needed based on your model
        random_input = torch.randn(1, autoencoder_args['in_channels'], 1000).to(device)
        z = autoencoder.encode_stage_2_inputs(random_input)
        scale_factor = 1.0 / torch.std(z)
        
    return autoencoder, scale_factor, autoencoder_args


def load_diffusion_model(config_path, model_path, latent_channels, device):
    """Load the trained diffusion model."""
    config = OmegaConf.load(config_path)
    parameters = config['model']['params']['unet_config']['params']
    parameters['in_channels'] = latent_channels
    parameters['out_channels'] = latent_channels
    
    # Create and load the diffusion model
    diffusion = UNetModel(**parameters)
    state_dict = torch.load(model_path, map_location=device)
    diffusion.load_state_dict(state_dict)
    diffusion = diffusion.to(device)
    diffusion.eval()
    
    # Create the scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        schedule="linear_beta",
        beta_start=0.0015, 
        beta_end=0.0195
    )
    scheduler = scheduler.to(device)
    
    return diffusion, scheduler


def calculate_mean_epochs(data_dir, label_file=None):
    """Calculate the mean number of epochs across all EEG recordings."""
    print("Calculating mean number of epochs across all data...")
    
    # Array to store epoch counts
    all_epoch_counts = []
    
    # Load label file if provided
    labels_dict = {}
    if label_file and os.path.exists(label_file):
        try:
            labels = np.load(label_file)
            for entry in labels:
                label, sample_id = entry[0], entry[1]
                labels_dict[int(sample_id)] = int(label)
            print(f"Loaded {len(labels_dict)} labels from {label_file}")
        except Exception as e:
            print(f"Error loading label file: {e}")
    
    # Find all feature files
    feature_dir = os.path.join(data_dir, "Feature")
    feature_files = glob.glob(os.path.join(feature_dir, "feature_*.npy"))
    print(f"Found {len(feature_files)} feature files")
    
    # Process each file
    for file_path in tqdm(feature_files, desc="Analyzing file shapes"):
        try:
            # Load the data
            data = np.load(file_path)
            
            # Get the number of epochs (first dimension)
            # For CAUEEG2, shape should be (epochs, timepoints, channels)
            num_epochs = data.shape[0]
            
            # Add to the list
            all_epoch_counts.append(num_epochs)
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Calculate mean number of epochs
    if all_epoch_counts:
        mean_epochs = int(np.mean(all_epoch_counts))
        print(f"Mean number of epochs: {mean_epochs} (from {len(all_epoch_counts)} samples)")
        print(f"Min epochs: {min(all_epoch_counts)}, Max epochs: {max(all_epoch_counts)}")
    else:
        mean_epochs = 50  # Default
        print("No samples found, using default of 50 epochs")
    
    return mean_epochs


def generate_single_epoch(diffusion_model, autoencoder, scheduler, 
                          batch_size, latent_shape, device, scale_factor):
    """Generate a single EEG epoch using the trained models."""
    
    # Check if CUDA is available for autocast
    use_autocast = torch.cuda.is_available()
    
    with torch.no_grad():
        # Start with random noise in the latent space
        latents = torch.randn((batch_size, *latent_shape), device=device)
        
        # Set timesteps
        scheduler.set_timesteps(num_inference_steps=1000)
        timesteps = scheduler.timesteps
        
        # Diffusion denoising process
        for t in timesteps:
            # Ensure timestep is a single scalar tensor for each item in batch
            t_batch = torch.ones(batch_size, dtype=torch.long, device=device) * t
            
            # Use autocast only when CUDA is available
            if use_autocast:
                with amp.autocast():
                    # Get model prediction
                    model_output = diffusion_model(latents, timesteps=t_batch)
                    
                    # Update latents
                    # Handle the case where step returns a tuple
                    step_output = scheduler.step(model_output, t, latents)
                    if isinstance(step_output, tuple):
                        # If it's a tuple, first element is usually the predicted sample
                        latents = step_output[0]
                    else:
                        # If it's an object with attributes (backward compatibility)
                        latents = step_output.prev_sample
            else:
                # No autocast for CPU
                model_output = diffusion_model(latents, timesteps=t_batch)
                # Handle the case where step returns a tuple
                step_output = scheduler.step(model_output, t, latents)
                if isinstance(step_output, tuple):
                    # If it's a tuple, first element is usually the predicted sample
                    latents = step_output[0]
                else:
                    # If it's an object with attributes (backward compatibility)
                    latents = step_output.prev_sample
        
        # Scale latents according to the scale factor
        latents = latents / scale_factor
        
        # Decode the latents to get the synthetic EEG signals
        if use_autocast:
            with amp.autocast():
                samples = autoencoder.decode_stage_2_outputs(latents)
        else:
            samples = autoencoder.decode_stage_2_outputs(latents)
        
        # Move to CPU and convert to numpy
        samples = samples.cpu().float().numpy()
        
        # Add this line to resize from 464 to 928 timepoints
        samples = interpolate_timepoints(samples, target_timepoints=928)
        
        return samples


def interpolate_timepoints(samples, target_timepoints=928):
    """
    Resize temporal dimension of generated samples using linear interpolation.
    
    Args:
        samples: Numpy array with shape (batch_size, channels, timepoints)
        target_timepoints: Desired number of timepoints in output
        
    Returns:
        Resized samples with shape (batch_size, channels, target_timepoints)
    """
    current_timepoints = samples.shape[2]
    if current_timepoints == target_timepoints:
        return samples
        
    # Create output array
    resized = np.zeros((samples.shape[0], samples.shape[1], target_timepoints))
    
    # Interpolate each channel for each sample
    for b in range(samples.shape[0]):
        for c in range(samples.shape[1]):
            # Linear interpolation
            resized[b, c] = np.interp(
                np.linspace(0, current_timepoints-1, target_timepoints),
                np.arange(current_timepoints),
                samples[b, c]
            )
    return resized

def generate_samples_3d(diffusion_model, autoencoder, scheduler, 
                       num_samples, batch_size, latent_shape, device, scale_factor,
                       num_epochs, timepoints=1000, channels=19):
    """Generate synthetic EEG samples with shape (num_epochs, channels, timepoints)."""
    
    # Store all generated samples
    all_samples = []
    
    # Generate samples in batches
    for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
        # Adjust batch size for the last batch if needed
        current_batch_size = min(batch_size, num_samples - i)
        
        # Create a tensor to store all epochs for this batch
        # Shape: [batch_size, num_epochs, channels, timepoints]
        batch_samples = np.zeros((current_batch_size, num_epochs, channels, timepoints))
        
        # Generate epochs for each sample
        for epoch_idx in tqdm(range(num_epochs), desc=f"Generating epochs for batch {i//batch_size+1}", leave=False):
            # Generate one epoch for the entire batch
            epoch_samples = generate_single_epoch(
                diffusion_model, autoencoder, scheduler,
                current_batch_size, latent_shape, device, scale_factor
            )
            
            # Add this epoch to each sample in the batch
            for sample_idx in range(current_batch_size):
                batch_samples[sample_idx, epoch_idx] = epoch_samples[sample_idx]
        
        all_samples.append(batch_samples)
    
    # Concatenate all generated samples
    all_samples = np.concatenate(all_samples, axis=0)
    
    # Ensure we only return the requested number of samples
    return all_samples[:num_samples]


def save_samples_and_labels(samples_dict, output_dir, reference_label_file=None):
    """Save the generated samples and create a label file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Feature directory for saving the EEG data
    feature_dir = os.path.join(output_dir, "Feature")
    os.makedirs(feature_dir, exist_ok=True)
    
    # Create Label directory for saving the labels
    label_dir = os.path.join(output_dir, "Label")
    os.makedirs(label_dir, exist_ok=True)
    
    # Initialize label data
    label_data = []
    
    # Counter for file naming
    file_counter = 0
    
    # Process each category
    for label, (label_name, samples) in enumerate([
        (0, samples_dict['hc']), 
        (1, samples_dict['mci']), 
        (2, samples_dict['dementia'])
    ]):
        print(f"Saving {len(samples)} {label_name} samples...")
        
        for i, sample in enumerate(samples):
            # Create filename
            filename = f"feature_{file_counter:02d}.npy"
            filepath = os.path.join(feature_dir, filename)
            
            # Create label entry [label, subject_id]
            # Use subject_id = file_counter for simplicity
            label_entry = [label, file_counter]
            label_data.append(label_entry)
            
            # Save the sample
            np.save(filepath, sample)
            
            # Increment counter
            file_counter += 1
    
    # Convert label_data to numpy array and save
    label_data = np.array(label_data)
    label_path = os.path.join(label_dir, "label.npy")
    np.save(label_path, label_data)
    
    # If reference label file is provided, match its structure
    if reference_label_file and os.path.exists(reference_label_file):
        try:
            ref_labels = np.load(reference_label_file)
            print(f"Reference label shape: {ref_labels.shape}")
            print(f"Generated label shape: {label_data.shape}")
            
            # Check if structures match or need adjustment
            if ref_labels.shape[1] != label_data.shape[1]:
                print("Warning: Reference label structure doesn't match generated structure.")
                print("Attempting to adjust to match reference structure...")
                
                # Create new label array with reference structure
                new_label_data = np.zeros((len(label_data), ref_labels.shape[1]))
                
                # Copy the label and subject_id columns we already have
                new_label_data[:, 0] = label_data[:, 0]  # Label
                new_label_data[:, 1] = label_data[:, 1]  # Subject ID
                
                # If there are more columns in the reference, fill with placeholder values
                for col in range(2, ref_labels.shape[1]):
                    # Use mean values from reference for each label class
                    for label in [0, 1, 2]:
                        mask = label_data[:, 0] == label
                        ref_mask = ref_labels[:, 0] == label
                        
                        if np.any(ref_mask):
                            col_values = ref_labels[ref_mask, col]
                            mean_val = np.mean(col_values)
                            new_label_data[mask, col] = mean_val
                
                # Save the adjusted label file
                adjusted_label_path = os.path.join(label_dir, "label_adjusted.npy")
                np.save(adjusted_label_path, new_label_data)
                print(f"Adjusted label file saved to {adjusted_label_path}")
                
        except Exception as e:
            print(f"Error matching reference label structure: {e}")
            print("Using basic label structure instead.")
    
    print(f"All samples and labels saved to {output_dir}")


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Set number of timepoints
    timepoints = args.timepoints
    print(f"Using {timepoints} timepoints per epoch")
    
    # If a fixed number of epochs is specified, use that
    if args.fixed_length:
        print(f"Using fixed number of {args.fixed_length} epochs for all categories")
        num_epochs = args.fixed_length
    else:
        # Calculate mean number of epochs across all categories
        num_epochs = calculate_mean_epochs(
            args.original_data_dir, 
            args.reference_label_file
        )
    
    print(f"Using {num_epochs} epochs for all categories")
    
    # Load autoencoders for each category
    print("Loading autoencoder models...")
    hc_autoencoder, hc_scale, hc_args = load_autoencoder(
        args.autoencoder_config, args.hc_autoencoder_path, device
    )
    mci_autoencoder, mci_scale, mci_args = load_autoencoder(
        args.autoencoder_config, args.mci_autoencoder_path, device
    )
    dementia_autoencoder, dementia_scale, dementia_args = load_autoencoder(
        args.autoencoder_config, args.dementia_autoencoder_path, device
    )
    
    # Get latent dimensions from autoencoders
    latent_channels = hc_autoencoder.latent_channels
    channels = hc_args['in_channels']  # Number of EEG channels (should be 19)
    
    # Calculate the latent shape (same for all categories)
    # For a 1000-point signal, the latent shape would typically be around 125 points (1000/8)
    # assuming 3 downsampling layers in the autoencoder
    latent_timepoints = timepoints // 8
    latent_shape = (latent_channels, latent_timepoints)
    print(f"Using latent shape: {latent_shape}")
    print(f"Output shape will be: ({num_epochs}, {channels}, {timepoints})")
    
    # Load diffusion models for each category
    print("Loading diffusion models...")
    hc_diffusion, hc_scheduler = load_diffusion_model(
        args.diffusion_config, args.hc_model_path, latent_channels, device
    )
    mci_diffusion, mci_scheduler = load_diffusion_model(
        args.diffusion_config, args.mci_model_path, latent_channels, device
    )
    dementia_diffusion, dementia_scheduler = load_diffusion_model(
        args.diffusion_config, args.dementia_model_path, latent_channels, device
    )
    
    # Generate 3D samples for each category
    print(f"Generating {args.num_hc} HC samples with shape ({num_epochs}, {channels}, {timepoints})...")
    hc_samples = generate_samples_3d(
        hc_diffusion, hc_autoencoder, hc_scheduler, 
        args.num_hc, args.batch_size, latent_shape, device, hc_scale,
        num_epochs, timepoints, channels
    )
    
    print(f"Generating {args.num_mci} MCI samples with shape ({num_epochs}, {channels}, {timepoints})...")
    mci_samples = generate_samples_3d(
        mci_diffusion, mci_autoencoder, mci_scheduler, 
        args.num_mci, args.batch_size, latent_shape, device, mci_scale,
        num_epochs, timepoints, channels
    )
    
    print(f"Generating {args.num_dementia} Dementia samples with shape ({num_epochs}, {channels}, {timepoints})...")
    dementia_samples = generate_samples_3d(
        dementia_diffusion, dementia_autoencoder, dementia_scheduler, 
        args.num_dementia, args.batch_size, latent_shape, device, dementia_scale,
        num_epochs, timepoints, channels
    )
    
    # Organize samples
    samples_dict = {
        'hc': hc_samples,
        'mci': mci_samples,
        'dementia': dementia_samples
    }
    
    # Save samples and create label file
    save_samples_and_labels(
        samples_dict, args.output_dir, args.reference_label_file
    )
    
    # Create a summary file with metadata
    summary_path = os.path.join(args.output_dir, "generation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Synthetic EEG Data Generation Summary\n")
        f.write("====================================\n\n")
        f.write(f"Generated at: {datetime.datetime.now()}\n\n")
        
        f.write("Sample counts:\n")
        f.write(f"- Healthy Controls: {args.num_hc}\n")
        f.write(f"- MCI: {args.num_mci}\n")
        f.write(f"- Dementia: {args.num_dementia}\n")
        f.write(f"- Total: {args.num_hc + args.num_mci + args.num_dementia}\n\n")
        
        f.write(f"Data dimensions (epochs, channels, timepoints): ({num_epochs}, {channels}, {timepoints})\n\n")
        
        f.write("Models used:\n")
        f.write(f"- HC Model: {args.hc_model_path}\n")
        f.write(f"- MCI Model: {args.mci_model_path}\n")
        f.write(f"- Dementia Model: {args.dementia_model_path}\n\n")
        
        f.write(f"- HC Autoencoder: {args.hc_autoencoder_path}\n")
        f.write(f"- MCI Autoencoder: {args.mci_autoencoder_path}\n")
        f.write(f"- Dementia Autoencoder: {args.dementia_autoencoder_path}\n\n")
        
        f.write(f"Original data directory: {args.original_data_dir}\n")
        f.write(f"Reference label file: {args.reference_label_file}\n")
        f.write(f"Random seed: {args.seed}\n")
    
    print(f"Summary saved to {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()