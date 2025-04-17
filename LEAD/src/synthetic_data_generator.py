import argparse
import os
import torch
import numpy as np
import random
from tqdm import tqdm
import h5py
import pickle
import matplotlib.pyplot as plt
import datetime

# Import the required components from your codebase
# Adjust these imports based on your project structure
import sys
sys.path.append("/mnt/beegfs/home/timlin/DiffusionEEG/LEAD/src/")
from models.LEAD import Model, DenoiseDiffusion
from utils.tools import dotdict

def load_model_and_diffusion(args):
    """
    Load the pretrained LEAD diffusion model and create the diffusion process
    """
    print("Loading model from checkpoint:", args.checkpoints_path)
    
    # Initialize the model with the configuration
    model = Model(args)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoints_path, map_location=args.device)
    
    # The checkpoint appears to be from DataParallel (has 'module.' prefix)
    # Handle this case specifically
    if all(k.startswith('module.') for k in checkpoint.keys()):
        print("Detected DataParallel checkpoint - removing 'module.' prefix")
        # Create a new state dict without the "module." prefix
        new_state_dict = {}
        for key, value in checkpoint.items():
            name = key[7:]  # Remove 'module.' prefix
            new_state_dict[name] = value
        # Load the new state dict
        model.load_state_dict(new_state_dict, strict=False)
    else:
        # Try other loading strategies (for completeness)
        if 'model' in checkpoint:
            print("Found 'model' key in checkpoint")
            model.load_state_dict(checkpoint['model'], strict=False)
        elif 'model_state_dict' in checkpoint:
            print("Found 'model_state_dict' key in checkpoint")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            # Direct loading if it's already a state dict
            model.load_state_dict(checkpoint, strict=False)
    
    # Move model to device
    model = model.to(args.device)
    model.eval()
    
    # Create the diffusion process
    diffusion = DenoiseDiffusion(
        eps_model=model,
        n_steps=args.n_steps,
        device=args.device,
        time_diff_constraint=args.time_diff_constraint
    )
    
    return model, diffusion

# Rest of the code remains the same
def generate_samples_for_subject(diffusion, config, subject_id, num_samples, batch_size=16):
    """
    Generate synthetic EEG samples for a specific subject using the diffusion model
    
    Args:
        diffusion: The diffusion model
        config: Configuration parameters
        subject_id: Subject ID for conditional generation
        num_samples: Number of samples to generate for this subject
        batch_size: Batch size for generation (to avoid OOM)
        
    Returns:
        Generated samples for this subject
    """
    all_samples = []
    total_batches = (num_samples + batch_size - 1) // batch_size
    
    # Convert to tensor and expand to batch size
    subject_tensor = torch.tensor([subject_id], device=config.device, dtype=torch.long)
    
    with torch.no_grad():
        for i in tqdm(range(total_batches), desc=f"Generating samples for subject {subject_id}"):
            # Calculate actual batch size for this iteration
            curr_batch_size = min(batch_size, num_samples - i * batch_size)
            if curr_batch_size <= 0:
                break
                
            # Define sample shape [batch_size, channels, seq_len]
            sample_shape = (curr_batch_size, config.enc_in, config.seq_len)
            
            # Repeat subject ID for batch
            batch_subject_ids = subject_tensor.repeat(curr_batch_size)
            
            # Generate samples with subject conditioning
            samples = diffusion.sample_with_subject(sample_shape, batch_subject_ids, sample_steps=config.sample_steps)
            
            # Move samples to CPU and convert to numpy for storage
            samples = samples.cpu().numpy()
            all_samples.append(samples)
    
    # Concatenate all batches
    all_samples = np.concatenate(all_samples, axis=0)
    return all_samples


def save_samples(samples, save_path, format='hdf5', metadata=None):
    """
    Save generated samples to disk
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if metadata is None:
        metadata = {}
    
    if format == 'hdf5':
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('eeg_data', data=samples)
            # Save metadata as attributes
            for key, value in metadata.items():
                if isinstance(value, dict):
                    # Convert nested dict to string
                    f.attrs[key] = str(value)
                else:
                    f.attrs[key] = value
        print(f"Saved {len(samples)} samples to {save_path} in HDF5 format")
    
    elif format == 'numpy':
        np.save(save_path, samples)
        # Save metadata separately
        if metadata:
            meta_path = save_path.replace('.npy', '_metadata.npy')
            with open(meta_path, 'wb') as f:
                pickle.dump(metadata, f)
        print(f"Saved {len(samples)} samples to {save_path} in NumPy format")
    
    elif format == 'pickle':
        data_dict = {
            'samples': samples,
            'metadata': metadata
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Saved {len(samples)} samples to {save_path} in Pickle format")
    
    else:
        raise ValueError(f"Unknown format '{format}'. Use 'hdf5', 'numpy', or 'pickle'.")


def visualize_samples(samples, subject_id, num_to_display=4, save_path=None):
    """
    Visualize some of the generated samples for a specific subject
    """
    num_to_display = min(num_to_display, len(samples))
    
    fig, axes = plt.subplots(num_to_display, 1, figsize=(12, 3 * num_to_display))
    if num_to_display == 1:
        axes = [axes]
    
    for i in range(num_to_display):
        sample = samples[i]
        
        # Plot each channel with a different color
        for ch_idx in range(min(10, sample.shape[0])):  # Limit to first 10 channels
            axes[i].plot(sample[ch_idx], label=f'Ch {ch_idx+1}' if i == 0 else "", alpha=0.7)
        
        axes[i].set_title(f'Subject {subject_id} - Generated Sample {i+1}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Amplitude')
        
        # Only add legend to the first plot to avoid clutter
        if i == 0:
            axes[i].legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.close()  # Close the figure to avoid display in notebooks


def get_config():
    """
    Create a configuration object with default values
    """
    parser = argparse.ArgumentParser(description='Generate synthetic EEG data for multiple subjects')
    
    # Model loading parameters
    parser.add_argument('--checkpoints_path', type=str, required=True, 
                        help='Path to the trained diffusion model checkpoint')
    
    # Generation parameters
    parser.add_argument('--samples_per_subject', type=int, default=100,
                        help='Number of synthetic samples to generate per subject')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for generation (to avoid OOM)')
    parser.add_argument('--n_steps', type=int, default=1000,
                        help='Number of steps for diffusion process')
    parser.add_argument('--sample_steps', type=int, default=50,
                        help='Number of steps for sampling from diffusion model')
    parser.add_argument('--time_diff_constraint', action='store_true',
                        help='Whether to use time difference constraint in diffusion')
    parser.add_argument('--num_subjects', type=int, default=33,
                        help='Number of subjects to generate data for (0 to num_subjects-1)')
    parser.add_argument('--start_subject', type=int, default=0,
                        help='Starting subject ID')
    
    # Data parameters - should match the training configuration
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length of the generated data')
    parser.add_argument('--enc_in', type=int, default=19,
                        help='Number of channels in the generated data')
    parser.add_argument('--dec_in', type=int, default=19,
                        help='Decoder input size')
    parser.add_argument('--c_out', type=int, default=19,
                        help='Output size')
    
    # Model parameters (matching your configs in main.py)
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension parameter from training')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of heads from training')
    parser.add_argument('--e_layers', type=int, default=12,
                        help='Number of encoder layers from training')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='Number of decoder layers from training')
    parser.add_argument('--d_ff', type=int, default=256,
                        help='Feed-forward dimension from training')
    parser.add_argument('--patch_len_list', type=str, default="4",
                        help='Patch lengths from training')
    parser.add_argument('--up_dim_list', type=str, default="76",
                        help='Up dimensions from training')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate from training')
    parser.add_argument('--output_attention', action='store_true',
                        help='Output attention like in training')
    parser.add_argument('--task_name', type=str, default='diffusion',
                        help='Task name (should be diffusion)')
    parser.add_argument('--model', type=str, default='LEAD',
                        help='Model name (should be LEAD)')
    parser.add_argument('--augmentations', type=str, 
                       default="flip,frequency,jitter,mask,channel,drop",
                       help='Augmentations for data')
    parser.add_argument('--activation', type=str, default='gelu',
                       help='Activation function')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--num_class', type=int, default=3,
                        help='Number of classes for classification')
    
    # Architecture-specific parameters
    parser.add_argument('--no_inter_attn', action='store_true',
                        default=False,
                        help='Whether to use inter-attention')
    parser.add_argument('--no_temporal_block', action='store_true',
                        default=False, 
                        help='Whether to use temporal block')
    parser.add_argument('--no_channel_block', action='store_true',
                        default=False,
                        help='Whether to use channel block')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./generated_samples/',
                        help='Directory to save generated samples')
    parser.add_argument('--file_format', type=str, default='hdf5',
                        choices=['hdf5', 'numpy', 'pickle'],
                        help='File format to save generated samples')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize some of the generated samples')
    parser.add_argument('--num_visualize', type=int, default=4,
                        help='Number of samples to visualize')
    
    # Device parameters
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU for generation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Additional required parameters
    parser.add_argument('--subject_conditional', action='store_true', default=True,
                        help='Enable subject conditioning for diffusion')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert to dotdict for easier access
    config = dotdict()
    for arg in vars(args):
        setattr(config, arg, getattr(args, arg))
    
    # Set device
    config.device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() and config.use_gpu else 'cpu')
    
    return config


def main():
    # Get configuration
    config = get_config()
    
    # Set random seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Using device: {config.device}")
    print(f"Generating {config.samples_per_subject} samples for each of {config.num_subjects} subjects")
    
    # Load the model and diffusion process
    model, diffusion = load_model_and_diffusion(config)
    
    # Timestamp for output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate data for each subject
    for subject_id in range(config.start_subject, config.start_subject + config.num_subjects):
        print(f"\n===== Generating data for subject {subject_id} =====")
        
        # Generate samples for this subject
        samples = generate_samples_for_subject(
            diffusion=diffusion,
            config=config,
            subject_id=subject_id,
            num_samples=config.samples_per_subject,
            batch_size=config.batch_size
        )
        
        # Prepare metadata
        metadata = {
            'subject_id': subject_id,
            'num_samples': config.samples_per_subject,
            'seq_len': config.seq_len,
            'channels': config.enc_in,
            'timestamp': timestamp,
            'model_config': str({k: str(v) for k, v in vars(config).items() if k != 'device'})
        }
        
        # Create output filename
        output_filename = f"subject_{subject_id}_eeg_{timestamp}"
        
        if config.file_format == 'hdf5':
            output_path = os.path.join(config.output_dir, f"{output_filename}.h5")
        elif config.file_format == 'numpy':
            output_path = os.path.join(config.output_dir, f"{output_filename}.npy")
        elif config.file_format == 'pickle':
            output_path = os.path.join(config.output_dir, f"{output_filename}.pkl")
        
        # Save samples
        save_samples(samples, output_path, config.file_format, metadata)
        
        # Visualize samples if requested
        if config.visualize:
            viz_path = os.path.join(config.output_dir, f"{output_filename}_viz.png")
            visualize_samples(samples, subject_id, config.num_visualize, viz_path)
    
    print("\nAll subject data generated successfully!")


if __name__ == "__main__":
    main()