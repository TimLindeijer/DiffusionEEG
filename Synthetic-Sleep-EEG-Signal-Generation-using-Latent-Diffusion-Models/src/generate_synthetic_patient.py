"""
Generate synthetic EEG data for a single patient using a trained diffusion model.
This script runs on CPU and allows generating a patient with a specified number of epochs.
All epochs are generated together as an interconnected unit, preserving temporal relationships.
"""

import argparse
import os
import torch
import numpy as np
import time
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from generative.networks.schedulers import DDPMScheduler
from models.ldm import UNetModel  # Your original UNet model

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic EEG data for a single patient")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file)"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the model config file (.yaml file)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_patients",
        help="Directory to save the generated data"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=60,
        help="Number of epochs to generate for the patient"
    )
    
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=1000,
        help="Number of denoising steps for inference"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--class_label",
        type=str,
        default="hc",
        choices=["hc", "mci", "dementia"],
        help="Class label for the generated patient (hc=healthy, mci=MCI, dementia=Dementia)"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot sample epochs from the generated data"
    )
    
    return parser.parse_args()

def load_model(model_path, config_path):
    """
    Load the trained UNet model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        config_path: Path to the model config file
        
    Returns:
        The loaded UNet model
    """
    # Load config
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    
    # Get model parameters
    parameters = config['model']['params']['unet_config']['params']
    parameters['in_channels'] = 19
    parameters['out_channels'] = 19
    parameters['image_size'] = 1000
    
    print(f"Creating UNet model with parameters:")
    for key, value in parameters.items():
        print(f"  {key}: {value}")
    
    # Initialize the UNet model
    model = UNetModel(**parameters)
    
    # Load checkpoint
    print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Handle both normal and DataParallel saved models
    if "module" in list(checkpoint["model_state_dict"].keys())[0]:
        # Model was saved with DataParallel
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        # Check if this is the wrapper model or just the UNet
        if "unet_model" in list(checkpoint["model_state_dict"].keys())[0]:
            # This is the wrapper model, extract just the UNet part
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["model_state_dict"].items():
                if "unet_model." in k:
                    name = k.replace("unet_model.", "")
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            # This is just the UNet model
            model.load_state_dict(checkpoint["model_state_dict"])
    
    # Put model in eval mode
    model.eval()
    print("Model loaded successfully!")
    
    return model

def map_class_to_numeric(class_label):
    """Map string class label to numeric value"""
    mapping = {"hc": 0, "mci": 1, "dementia": 2}
    return mapping.get(class_label.lower(), 0)

def generate_patient(model, num_epochs=60, inference_steps=1000, seed=None, device="cpu"):
    """
    Generate synthetic EEG data for a single patient with all epochs at once.
    This preserves any learned temporal relationships between epochs.
    
    Args:
        model: The UNet model
        num_epochs: Number of epochs to generate
        inference_steps: Number of denoising steps
        seed: Random seed for reproducibility
        device: Device to run the model on
        
    Returns:
        Tensor with shape [num_epochs, 19, 1000]
    """
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Using random seed: {seed}")
    
    print(f"Generating patient with {num_epochs} interconnected epochs, using {inference_steps} inference steps")
    print(f"Using device: {device}")
    
    # Create a scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="linear_beta",
        beta_start=0.0015,
        beta_end=0.0195
    )
    
    # Set the scheduler timesteps
    scheduler.set_timesteps(inference_steps)
    
    # Generate initial noise for each epoch
    sample = torch.randn(num_epochs, 19, 1000, device=device)
    
    # Track generation time
    start_time = time.time()
    
    # Denoise the sample step by step
    print(f"Starting denoising process with {len(scheduler.timesteps)} steps")
    for i, timestep in enumerate(scheduler.timesteps):
        # Print progress
        if (i + 1) % 100 == 0 or i == 0 or i == len(scheduler.timesteps) - 1:
            elapsed_time = time.time() - start_time
            progress = (i + 1) / len(scheduler.timesteps) * 100
            remaining_time = (elapsed_time / (i + 1)) * (len(scheduler.timesteps) - i - 1)
            print(f"Step {i+1}/{len(scheduler.timesteps)} ({progress:.1f}%) - "
                  f"Time elapsed: {elapsed_time:.2f}s - "
                  f"Est. remaining: {remaining_time:.2f}s")
            sys.stdout.flush()
        
        # Process each epoch individually
        all_noise_preds = []
        for epoch_idx in range(num_epochs):
            # Extract a single epoch (add batch dimension)
            single_epoch = sample[epoch_idx:epoch_idx+1]
            
            # Predict noise for this epoch with the model
            with torch.no_grad():
                noise_pred = model(single_epoch, timestep.unsqueeze(0))
            
            all_noise_preds.append(noise_pred)
        
        # Stack all noise predictions
        noise_pred_batch = torch.cat(all_noise_preds, dim=0)
        
        # Update sample with scheduler
        # The scheduler might return a tuple or an object, handle both cases
        step_output = scheduler.step(noise_pred_batch, timestep, sample)
        
        # Handle both cases: when step_output is a tuple and when it's an object with prev_sample attribute
        if isinstance(step_output, tuple):
            # If it's a tuple, the first element should be the sample
            sample = step_output[0]
        else:
            # If it's an object with prev_sample attribute
            sample = step_output.prev_sample
    
    # Report completion and timing
    total_time = time.time() - start_time
    print(f"Generation complete! Total time: {total_time:.2f} seconds")
    
    return sample

def save_patient_data(patient_data, output_dir, class_label, seed=None):
    """
    Save the generated patient data to a file.
    
    Args:
        patient_data: Tensor with shape [num_epochs, 19, 1000]
        output_dir: Directory to save the data
        class_label: Class label for the generated patient (hc, mci, dementia)
        seed: Random seed used for generation
    
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy array
    patient_np = patient_data.detach().cpu().numpy()
    
    # Get shape information
    num_epochs, num_channels, num_timepoints = patient_np.shape
    
    # Current timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create filename with metadata
    if seed is not None:
        filename = f"synthetic_patient_{class_label}_e{num_epochs}_seed{seed}_{timestamp}.npy"
    else:
        filename = f"synthetic_patient_{class_label}_e{num_epochs}_{timestamp}.npy"
    
    # Full path
    filepath = os.path.join(output_dir, filename)
    
    # Save the data
    np.save(filepath, patient_np)
    print(f"Saved patient data to {filepath}")
    print(f"Data shape: {patient_np.shape} (epochs, channels, timepoints)")
    
    # Save metadata
    metadata = {
        "class_label": class_label,
        "num_epochs": num_epochs,
        "num_channels": num_channels,
        "num_timepoints": num_timepoints,
        "seed": seed,
        "timestamp": timestamp
    }
    
    metadata_path = filepath.replace(".npy", "_metadata.txt")
    with open(metadata_path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    return filepath

def plot_sample_epochs(patient_data, filepath, class_label):
    """
    Plot sample epochs from the generated patient data.
    
    Args:
        patient_data: Tensor with shape [num_epochs, 19, 1000]
        filepath: Path where the plot will be saved
        class_label: String label of the class (hc, mci, dementia)
    """
    # Convert to numpy if tensor
    if isinstance(patient_data, torch.Tensor):
        patient_data = patient_data.detach().cpu().numpy()
    
    # Get number of epochs
    num_epochs = patient_data.shape[0]
    
    # Select a few epochs to plot
    num_sample_epochs = min(5, num_epochs)
    sample_indices = np.linspace(0, num_epochs-1, num_sample_epochs, dtype=int)
    
    # Map class label to display name
    class_display = {
        "hc": "Healthy Control",
        "mci": "Mild Cognitive Impairment",
        "dementia": "Dementia"
    }.get(class_label.lower(), class_label)
    
    # Create plot
    fig, axes = plt.subplots(num_sample_epochs, 1, figsize=(12, num_sample_epochs * 2.5), sharex=True)
    
    if num_sample_epochs == 1:
        axes = [axes]
    
    for i, epoch_idx in enumerate(sample_indices):
        ax = axes[i]
        
        # Plot 3 sample channels (front, middle, and back)
        channel_indices = [0, 9, 18]  # Channels 0, 9, 18
        channel_names = ["Front", "Middle", "Back"]
        colors = ['b', 'g', 'r']
        
        for c, cname, color in zip(channel_indices, channel_names, colors):
            if c < patient_data.shape[1]:
                ax.plot(patient_data[epoch_idx, c, :], color=color, label=cname)
        
        ax.set_title(f"Epoch {epoch_idx+1}/{num_epochs}")
        ax.set_ylabel("Amplitude")
        
        # Only add legend to first plot
        if i == 0:
            ax.legend()
    
    axes[-1].set_xlabel("Timepoints")
    fig.suptitle(f"Sample Epochs from Generated Patient ({class_display})", fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plot_path = filepath.replace(".npy", "_plot.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    
    # Close figure to free memory
    plt.close(fig)
    
    return plot_path

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device to CPU
    device = "cpu"
    
    # Load model
    model = load_model(args.model_path, args.config_path)
    
    # Get numeric class label for internal use if needed
    numeric_class = map_class_to_numeric(args.class_label)
    print(f"Generating patient with class label: {args.class_label} (numeric: {numeric_class})")
    
    # Generate patient data
    patient_data = generate_patient(
        model=model,
        num_epochs=args.num_epochs,
        inference_steps=args.inference_steps,
        seed=args.seed,
        device=device
    )
    
    # Save the generated data
    filepath = save_patient_data(
        patient_data=patient_data,
        output_dir=args.output_dir,
        class_label=args.class_label,
        seed=args.seed
    )
    
    # Plot sample epochs if requested
    if args.plot:
        plot_sample_epochs(patient_data, filepath, args.class_label)
    
    print("Done!")

if __name__ == "__main__":
    main()