#!/usr/bin/env python
"""
Optimized script to generate synthetic EEG data with the same structure as the original dataset.
This version includes significant performance optimizations and clear progress tracking.
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

# Import necessary model components
print("START OF SCRIPT: Importing required modules...")
from generative.networks.nets import AutoencoderKL
from generative.networks.schedulers import DDPMScheduler
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
    
    # Model paths for each category
    parser.add_argument("--hc_model_path", type=str, required=True, 
                      help="Path to the healthy controls diffusion model weights")
    parser.add_argument("--mci_model_path", type=str, required=True,
                      help="Path to the MCI diffusion model weights")
    parser.add_argument("--dementia_model_path", type=str, required=True, 
                      help="Path to the dementia diffusion model weights")
    
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
    
    # Original dataset information
    parser.add_argument("--original_label_path", type=str, required=True,
                      help="Path to the original label.npy file (e.g., dataset/CAUEEG2/Label/label.npy)")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="synthetic_eeg_matched",
                      help="Directory to save the generated samples")
    
    # Performance optimization parameters
    parser.add_argument("--diffusion_steps", type=int, default=100,
                      help="Number of diffusion steps (fewer = faster)")
    parser.add_argument("--batch_epochs", type=int, default=64,
                      help="Number of epochs to generate in a single batch")
    parser.add_argument("--category", type=str, choices=["hc", "mci", "dementia", "all"], default="all",
                      help="Which category to generate (for parallel jobs)")
    
    # Generation parameters
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                      help="Use GPU for generation if available")
    
    # Fixed dimensions
    parser.add_argument("--num_epochs", type=int, default=71,
                      help="Number of epochs per sample")
    parser.add_argument("--num_channels", type=int, default=19,
                      help="Number of EEG channels")
    parser.add_argument("--num_timepoints", type=int, default=928,
                      help="Number of timepoints per epoch")
    
    args = parser.parse_args()
    print(f"Arguments parsed successfully: Category={args.category}, Diffusion steps={args.diffusion_steps}, Batch epochs={args.batch_epochs}")
    return args

def read_original_labels(label_path):
    """
    Read the original label.npy file to extract feature numbering and labels.
    
    Returns:
        Dictionary mapping label (0, 1, 2) to list of feature file numbers
    """
    print(f"Reading original labels from: {label_path}")
    try:
        # Load the original labels
        labels = np.load(label_path)
        print(f"Successfully loaded original label file with shape: {labels.shape}")
        
        # Group by label
        label_to_features = defaultdict(list)
        for entry in labels:
            label = int(entry[0])  # First column is label
            subject_id = int(entry[1])  # Second column is subject_id
            label_to_features[label].append(subject_id)
        
        # Print summary
        print(f"Found {len(label_to_features[0])} HC samples, " +
              f"{len(label_to_features[1])} MCI samples, " +
              f"{len(label_to_features[2])} Dementia samples")
        
        return label_to_features
    except Exception as e:
        print(f"ERROR loading labels: {str(e)}")
        raise

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

def load_autoencoder(config_path, autoencoder_path, device):
    """Load the autoencoder model and compute latent dimensions."""
    print(f"Loading autoencoder config from: {config_path}")
    
    try:
        # Load config
        config = OmegaConf.load(config_path)
        autoencoder_args = config.autoencoderkl.params
        print(f"Autoencoder configuration loaded successfully")
        
        # Create model
        print(f"Creating autoencoder model instance...")
        autoencoder = AutoencoderKL(**autoencoder_args)
        print(f"Autoencoder created successfully, loading weights from: {autoencoder_path}")
        
        # Load weights
        state_dict = torch.load(autoencoder_path, map_location=device)
        print(f"Weights loaded successfully, applying to model...")
        autoencoder.load_state_dict(state_dict)
        print(f"Moving model to device: {device}")
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
        print(f"Autoencoder ready on device: {device}")
        
        # Calculate scale factor 
        print("Calculating latent dimensions and scale factor...")
        latent_channels = autoencoder_args['latent_channels']
        with torch.no_grad():
            # Create a random input tensor
            print(f"Creating test input tensor with shape: [1, {autoencoder_args['in_channels']}, 928]")
            random_input = torch.randn(1, autoencoder_args['in_channels'], 928).to(device)
            print("Encoding test tensor to determine latent shape...")
            z = autoencoder.encode_stage_2_inputs(random_input)
            print(f"Latent tensor shape: {z.shape}")
            scale_factor = 1.0 / torch.std(z)
            latent_shape = z.shape[2]  # Get actual latent dimension
            
        print(f"Autoencoder loaded successfully. Latent channels: {latent_channels}, Latent shape: {latent_shape}")
        print(f"Scale factor: {scale_factor}")
        print(f"Current GPU memory: {get_gpu_memory()}")
            
        return autoencoder, scale_factor, latent_channels, latent_shape
    except Exception as e:
        print(f"ERROR loading autoencoder: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def load_diffusion_model(config_path, model_path, latent_channels, latent_shape, device):
    """Load the diffusion model with correct parameter handling."""
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
        
        # Override latent space parameters
        parameters['in_channels'] = latent_channels
        parameters['out_channels'] = latent_channels
        parameters['image_size'] = latent_shape
        
        print(f"Diffusion model configuration: {parameters}")
        
        # Create diffusion model
        print(f"Creating diffusion model instance...")
        diffusion = UNetModel(**parameters)
        print("Diffusion model created successfully")
        
        # Load weights
        print(f"Loading diffusion model weights from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        print("Weights loaded successfully, applying to model...")
        diffusion.load_state_dict(state_dict)
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
        print(f"Current GPU memory: {get_gpu_memory()}")
        
        return diffusion, scheduler
    except Exception as e:
        print(f"ERROR loading diffusion model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def generate_sample_batch(diffusion_model, autoencoder, scheduler, scale_factor, latent_shape, 
                       num_epochs, channels, timepoints, batch_size, diffusion_steps):
    """Generate a batch of epochs all at once with optimized settings."""
    device = next(diffusion_model.parameters()).device
    use_autocast = torch.cuda.is_available()
    
    # Create container for all epochs
    sample = np.zeros((num_epochs, channels, timepoints))
    
    # Process in batches for memory efficiency
    for i in range(0, num_epochs, batch_size):
        batch_size_current = min(batch_size, num_epochs - i)
        
        try:
            with torch.no_grad():
                # Generate all latents
                latents = torch.randn((batch_size_current, latent_shape[0], latent_shape[1]), device=device)
                
                # Optimized diffusion process with fewer steps
                scheduler.set_timesteps(num_inference_steps=diffusion_steps)
                timesteps = scheduler.timesteps
                
                # Diffusion denoising process
                for step_idx, t in enumerate(timesteps):
                    if step_idx % 20 == 0:  # Print progress every 20 steps
                        print(f"  Diffusion progress: step {step_idx+1}/{len(timesteps)}")
                    
                    t_batch = torch.ones(batch_size_current, dtype=torch.long, device=device) * t
                    
                    if use_autocast:
                        with amp.autocast():
                            model_output = diffusion_model(latents, timesteps=t_batch)
                            step_output = scheduler.step(model_output, t, latents)
                            latents = step_output[0] if isinstance(step_output, tuple) else step_output.prev_sample
                    else:
                        model_output = diffusion_model(latents, timesteps=t_batch)
                        step_output = scheduler.step(model_output, t, latents)
                        latents = step_output[0] if isinstance(step_output, tuple) else step_output.prev_sample
                
                # Scale and decode
                latents = latents / scale_factor
                
                if use_autocast:
                    with amp.autocast():
                        epochs_batch = autoencoder.decode_stage_2_outputs(latents)
                else:
                    epochs_batch = autoencoder.decode_stage_2_outputs(latents)
                
                # Move to CPU
                epochs_batch = epochs_batch.cpu().float().numpy()
                
                # Handle dimension mismatch
                if epochs_batch.shape[2] != timepoints:
                    print(f"Warning: Generated data has {epochs_batch.shape[2]} timepoints instead of {timepoints}, resizing...")
                    resized_batch = np.zeros((epochs_batch.shape[0], epochs_batch.shape[1], timepoints))
                    for b in range(epochs_batch.shape[0]):
                        for c in range(epochs_batch.shape[1]):
                            resized_batch[b, c] = np.interp(
                                np.linspace(0, epochs_batch.shape[2] - 1, timepoints),
                                np.arange(epochs_batch.shape[2]),
                                epochs_batch[b, c]
                            )
                    epochs_batch = resized_batch
                
                # Add to sample
                for j in range(batch_size_current):
                    if i + j < num_epochs:
                        sample[i + j] = epochs_batch[j]
                
                # Cleanup to free memory
                del latents, model_output, step_output, epochs_batch
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"ERROR in generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    print(f"Sample generation completed, shape: {sample.shape}")
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

def generate_category(category_name, label, feature_ids, diffusion, autoencoder, scheduler, scale_factor, latent_shape, 
                     output_dir, args, completed_ids=None):
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
                diffusion, autoencoder, scheduler, scale_factor, latent_shape,
                args.num_epochs, args.num_channels, args.num_timepoints,
                args.batch_epochs, args.diffusion_steps
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
    print("========== SCRIPT STARTING ==========")
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
    
    # Read original labels to get structure
    print("Reading original labels...")
    label_to_features = read_original_labels(args.original_label_path)
    
    # Save labels file immediately (it's the same regardless of which samples we generate)
    print("Saving labels file...")
    save_labels(label_to_features, args.output_dir)
    
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
            print("Starting to load HC models")
            # Load autoencoder
            print("Loading HC autoencoder...")
            autoencoder, scale_factor, latent_channels, latent_shape = load_autoencoder(
                args.autoencoder_config, 
                args.hc_autoencoder_path,
                device
            )
            # Load diffusion model
            print("Loading HC diffusion model...")
            diffusion, scheduler = load_diffusion_model(
                args.diffusion_config,
                args.hc_model_path,
                latent_channels,
                (latent_channels, latent_shape),
                device
            )
        elif category_name == "mci":
            print("Starting to load MCI models")
            # Load autoencoder
            print("Loading MCI autoencoder...")
            autoencoder, scale_factor, latent_channels, latent_shape = load_autoencoder(
                args.autoencoder_config, 
                args.mci_autoencoder_path,
                device
            )
            # Load diffusion model
            print("Loading MCI diffusion model...")
            diffusion, scheduler = load_diffusion_model(
                args.diffusion_config,
                args.mci_model_path,
                latent_channels,
                (latent_channels, latent_shape),
                device
            )
        elif category_name == "dementia":
            print("Starting to load Dementia models")
            # Load autoencoder
            print("Loading Dementia autoencoder...")
            autoencoder, scale_factor, latent_channels, latent_shape = load_autoencoder(
                args.autoencoder_config, 
                args.dementia_autoencoder_path,
                device
            )
            # Load diffusion model
            print("Loading Dementia diffusion model...")
            diffusion, scheduler = load_diffusion_model(
                args.diffusion_config,
                args.dementia_model_path,
                latent_channels,
                (latent_channels, latent_shape),
                device
            )
        
        print(f"Models loaded successfully for {category_name}")
        print(f"Current GPU memory after loading models: {get_gpu_memory()}")
        
        # Generate samples for this category
        print(f"Starting generation for {category_name}...")
        generate_category(
            category_name, label, feature_ids, diffusion, autoencoder, scheduler, 
            scale_factor, (latent_channels, latent_shape), args.output_dir, args, completed_ids
        )
        
        # Clear GPU memory
        print(f"Clearing GPU memory after {category_name} generation")
        del diffusion, autoencoder, scheduler
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