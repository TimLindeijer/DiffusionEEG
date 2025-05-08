"""
Author: Bruno Aristimunha (Modified)
Training LDM with CAUEEG2 data, processing batches of patients while keeping each patient separate.
Based on the tutorial from Monai Generative.
"""
import argparse
import os
import torch
import torch.nn as nn

from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from generative.inferers import DiffusionInferer

# Import from local modules
from dataset.dataset import get_trans, get_caueeg2_datalist
from models.ldm import UNetModel  # Your original UNet model
from patient_diffusion_wrapper import (
    BatchedPatientDiffusionWrapper, 
    collate_patients_fn, 
    train_diffusion_batched_patients
)
from util import log_mlflow, ParseListAction, setup_run_dir

# Custom DataLoader that uses our custom collate function
from torch.utils.data import DataLoader
from monai.data import PersistentDataset

# Set determinism for reproducibility
set_determinism(42)

if os.path.exists('/project'):
    base_path = '/project/'
    base_path_data = '/data/'
else:
    base_path = '/home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models'
    base_path_data = base_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=f"{base_path}/config/config_dm.yaml",
        help="Path to config file with all the training parameters needed",
    )
    parser.add_argument(
        "--path_train_ids",
        type=str,
        default=f"{base_path}/data/ids/ids_sleep_edfx_cassette_double_train.csv",
    )

    parser.add_argument(
        "--path_valid_ids",
        type=str,
        default=f"{base_path}/data/ids/ids_sleep_edfx_cassette_double_valid.csv",
    )
    parser.add_argument(
        "--path_cached_data",
        type=str,
        default=f"{base_path_data}/pre",
    )

    parser.add_argument(
        "--path_pre_processed",
        type=str,
        default="/data/physionet-sleep-data-npy",
    )
        
    parser.add_argument(
        "--spe",
        type=str,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["edfx", "shhs", "shhsh", "caueeg2"],
        default="caueeg2",
        help="Dataset to use for training"
    )
    
    # Add label filter parameter for CAUEEG2 dataset
    parser.add_argument(
        "--label_filter",
        type=str,
        help="Filter to include only specific labels: 'hc'/'0' (Healthy Controls), 'mci'/'1' (MCI), or 'dementia'/'2' (Dementia). Can use comma-separated values."
    )
    
    # Add batch size parameter
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of patients to process in parallel"
    )

    args = parser.parse_args()
    
    # Process label_filter if provided
    if args.label_filter and ',' in args.label_filter:
        args.label_filter = args.label_filter.split(',')
    
    return args


def create_dataloaders(config, args, transforms_list):
    """
    Create train and validation dataloaders with our custom collate function.
    
    Args:
        config: Configuration object
        args: Command line arguments
        transforms_list: Transforms to apply to data
        
    Returns:
        train_loader, val_loader
    """
    # Get label filter
    label_filter = args.label_filter if hasattr(args, 'label_filter') else None
    
    # Get all data dictionaries
    all_dicts = get_caueeg2_datalist(args.path_pre_processed, label_filter)
    
    # Handle empty dataset case
    if not all_dicts:
        raise ValueError("Dataset is empty after applying label filter. Check available labels.")
        
    # Split into train and validation
    train_size = int(0.8 * len(all_dicts))
    train_dicts = all_dicts[:train_size]
    valid_dicts = all_dicts[train_size:]
    
    print(f"Using {len(train_dicts)} patients for training")
    print(f"Using {len(valid_dicts)} patients for validation")
    
    # Create datasets
    train_ds = PersistentDataset(
        data=train_dicts,
        transform=transforms_list,
        cache_dir=None
    )
    
    valid_ds = PersistentDataset(
        data=valid_dicts,
        transform=transforms_list,
        cache_dir=None
    )
    
    # Batch size for patient batching
    batch_size = args.batch_size
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        drop_last=config.train.drop_last,
        pin_memory=False,
        persistent_workers=True,
        collate_fn=collate_patients_fn  # Use our custom collate function
    )
    
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        drop_last=config.train.drop_last,
        pin_memory=False,
        persistent_workers=True,
        collate_fn=collate_patients_fn  # Use our custom collate function
    )
    
    return train_loader, valid_loader


def main(args):
    config = OmegaConf.load(args.config_file)

    set_determinism(seed=config.train.seed)
    print_config()

    run_dir, resume = setup_run_dir(config=config, args=args,
                                   base_path=base_path)

    # Setting up tensorboard writers
    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))
    
    # Get dataset-specific transforms
    trans = get_trans(args.dataset)
    
    # Getting data loaders with our custom collate function
    train_loader, val_loader = create_dataloaders(config, args, trans)

    # Sanity check: print shape of first batch
    sample_batch = next(iter(train_loader))
    num_patients = len(sample_batch)
    print(f"Sample batch contains {num_patients} patients")
    for i, patient in enumerate(sample_batch):
        print(f"  Patient {i+1}: {patient['eeg'].shape} (epochs, channels, timepoints)")
    
    # Defining device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Configure model parameters
    parameters = config['model']['params']['unet_config']['params']
    
    # Adjust parameters for CAUEEG2 dataset - Note: UNet still expects [batch, channels, timepoints]
    parameters['in_channels'] = 19
    parameters['out_channels'] = 19
    parameters['image_size'] = 1000  # Set image_size to match timepoints dimension

    # Print model parameters
    print(f"UNet parameters: {parameters}")

    # Initialize the diffusion model (the original UNet)
    print("Initializing base UNet model...")
    unet_model = UNetModel(**parameters)
    
    # Wrap the UNet in our batched patient wrapper
    print("Creating batched patient wrapper model...")
    diffusion = BatchedPatientDiffusionWrapper(unet_model)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        diffusion = torch.nn.DataParallel(diffusion)

    diffusion.to(device)

    # Set up the diffusion scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta",
                            beta_start=0.0015, beta_end=0.0195)
    
    scheduler.to(device)
    
    # Determine whether to use spectral loss
    if args.spe == 'spectral':
        spectral_loss = True
        print("Using spectral loss")
    else:
        spectral_loss = False

    # Initialize the diffusion inferer
    inferer = DiffusionInferer(scheduler)

    # Set up the optimizer
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

    best_loss = float("inf")
    start_epoch = 0

    print(f"Starting Training")
    # Use our modified training function for batched patients
    val_loss = train_diffusion_batched_patients(
        model=diffusion,
        scheduler=scheduler,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=config.train.n_epochs,
        eval_freq=config.train.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        inferer=inferer,
        spectral_loss=spectral_loss,
        spectral_weight=1E-6
    )

    # Log the results
    log_mlflow(
        model=diffusion,
        config=config,
        args=args,
        run_dir=run_dir,
        val_loss=val_loss,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)