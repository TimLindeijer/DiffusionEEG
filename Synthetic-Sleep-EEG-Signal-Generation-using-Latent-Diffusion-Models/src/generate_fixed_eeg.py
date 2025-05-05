#!/usr/bin/env python
"""
Generate synthetic EEG data with fixed dimensions (71 epochs, 19 channels, 928 timepoints).
This script loads trained models and generates synthetic EEG data with consistent shape.
"""

import os
import argparse
import numpy as np
import torch
import torch.cuda.amp as amp
from pathlib import Path
from tqdm import tqdm

# Import necessary model components
from generative.networks.nets import AutoencoderKL
from generative.networks.schedulers import DDPMScheduler
from models.ldm import UNetModel
from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic EEG data with fixed dimensions")
    
    # Model paths
    parser.add_argument("--model_path", type=str, required=True, 
                      help="Path to the diffusion model weights")
    parser.add_argument("--autoencoder_path", type=str, required=True,
                      help="Path to the autoencoder model weights")
    
    # Config files
    parser.add_argument("--autoencoder_config", type=str, required=True,
                      help="Path to the autoencoder config file")
    parser.add_argument("--diffusion_config", type=str, required=True,
                      help="Path to the diffusion model config file")
    
    # Number of samples to generate
    parser.add_argument("--num_samples", type=int, default=50,
                      help="Number of samples to generate")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="synthetic_fixed_eeg",
                      help="Directory to save the generated samples")
    
    # Label for generated data (0=HC, 1=MCI, 2=dementia)
    parser.add_argument("--label", type=int, default=0,
                      help="Label for generated data (0=HC, 1=MCI, 2=dementia)")
    
    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                      help="Use GPU for generation if available")
    
    return parser.parse_args()

def load_autoencoder(config_path, model_path, device):
    """Load the trained autoencoder model."""
    config = OmegaConf.load(config_path)
    autoencoder_args = config.autoencoderkl.params
    
    print(f"Autoencoder configuration: {autoencoder_args}")
    
    # Create and load the autoencoder
    autoencoder = AutoencoderKL(**autoencoder_args)
    state_dict = torch.load(model_path, map_location=device)
    autoencoder.load_state_dict(state_dict)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    
    # Calculate scale factor based on a random input
    latent_channels = autoencoder_args['latent_channels']
    with torch.no_grad():
        # Create a random input tensor with the expected shape (batch, channels, timepoints)
        # For 19 channels and 928 timepoints
        random_input = torch.randn(1, autoencoder_args['in_channels'], 928).to(device)
        z = autoencoder.encode_stage_2_inputs(random_input)
        scale_factor = 1.0 / torch.std(z)
        latent_shape = z.shape[2]  # Get actual latent dimension
        
    print(f"Loaded autoencoder model. Latent channels: {latent_channels}, Latent shape: {latent_shape}")
    print(f"Scale factor: {scale_factor}")
        
    return autoencoder, scale_factor, latent_channels, latent_shape

def load_diffusion_model(config_path, model_path, latent_channels, latent_shape, device):
    """Load the trained diffusion model."""
    config = OmegaConf.load(config_path)
    parameters = config['model']['params']['unet_config']['params']
    
    # Override parameters based on actual latent dimensions
    parameters['in_channels'] = latent_channels
    parameters['out_channels'] = latent_channels
    parameters['image_size'] = latent_shape
    
    print(f"Diffusion model configuration: {parameters}")
    
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
        for t in tqdm(timesteps, desc="Diffusion steps", leave=False):
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
    
    return samples

def generate_samples_fixed_shape(diffusion_model, autoencoder, scheduler, 
                              num_samples, batch_size, latent_shape, device, scale_factor,
                              num_epochs=71, channels=19, timepoints=928):
    """Generate synthetic EEG samples with fixed shape (num_epochs, channels, timepoints)."""
    
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
            
            # Check if the output shape matches expected dimensions
            if epoch_samples.shape[2] != timepoints:
                print(f"Warning: Generated data has {epoch_samples.shape[2]} timepoints instead of {timepoints}")
                # Resize to match expected timepoints
                resized_samples = np.zeros((epoch_samples.shape[0], epoch_samples.shape[1], timepoints))
                for b in range(epoch_samples.shape[0]):
                    for c in range(epoch_samples.shape[1]):
                        # Linear interpolation
                        resized_samples[b, c] = np.interp(
                            np.linspace(0, epoch_samples.shape[2]-1, timepoints),
                            np.arange(epoch_samples.shape[2]),
                            epoch_samples[b, c]
                        )
                epoch_samples = resized_samples
            
            # Add this epoch to each sample in the batch
            for sample_idx in range(current_batch_size):
                batch_samples[sample_idx, epoch_idx] = epoch_samples[sample_idx]
        
        all_samples.append(batch_samples)
    
    # Concatenate all generated samples
    all_samples = np.concatenate(all_samples, axis=0)
    
    # Ensure we only return the requested number of samples
    return all_samples[:num_samples]

def save_samples_and_labels(samples, output_dir, label=0):
    """Save the generated samples and create a label file."""
    # Create output directories
    feature_dir = os.path.join(output_dir, "Feature")
    label_dir = os.path.join(output_dir, "Label")
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    # Initialize label data
    label_data = []
    
    # Save each sample
    for i, sample in enumerate(samples):
        # Create filename
        filename = f"feature_{i:02d}.npy"
        filepath = os.path.join(feature_dir, filename)
        
        # Create label entry [label, subject_id]
        label_entry = [label, i]
        label_data.append(label_entry)
        
        # Save the sample
        np.save(filepath, sample)
        print(f"Saved sample {i+1}/{len(samples)} to {filepath}")
    
    # Convert label_data to numpy array and save
    label_data = np.array(label_data)
    label_path = os.path.join(label_dir, "label.npy")
    np.save(label_path, label_data)
    print(f"Saved labels to {label_path}")

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading autoencoder model...")
    autoencoder, scale_factor, latent_channels, latent_shape = load_autoencoder(
        args.autoencoder_config, args.autoencoder_path, device
    )
    
    print("Loading diffusion model...")
    diffusion, scheduler = load_diffusion_model(
        args.diffusion_config, args.model_path, latent_channels, latent_shape, device
    )
    
    # Get latent dimensions
    latent_dims = (latent_channels, latent_shape)
    print(f"Using latent dimensions: {latent_dims}")
    print(f"Target output shape: (71, 19, 928)")
    
    # Generate fixed-shape samples
    print(f"Generating {args.num_samples} samples...")
    samples = generate_samples_fixed_shape(
        diffusion, autoencoder, scheduler,
        args.num_samples, args.batch_size, latent_dims, device, scale_factor,
        num_epochs=71, channels=19, timepoints=928
    )
    
    print(f"Generated samples shape: {samples.shape}")
    
    # Save samples and labels
    print("Saving samples and labels...")
    save_samples_and_labels(samples, args.output_dir, label=args.label)
    
    print(f"Generation complete! {args.num_samples} samples saved to {args.output_dir}")

if __name__ == "__main__":
    main()