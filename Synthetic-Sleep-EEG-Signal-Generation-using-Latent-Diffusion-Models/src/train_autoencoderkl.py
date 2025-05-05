"""
Author: Bruno Aristimunha (Modified with ultra-stable training)
Training AutoEncoder KL with SleepEDFx or SHHS data.
Based on the tutorial from Monai Generative.
"""
import argparse
import ast
import time
import os
import math
from generative.losses import JukeboxLoss, PatchAdversarialLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.config import print_config
from monai.utils import first, set_determinism
from omegaconf import OmegaConf
from torch.nn import L1Loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.dataset import train_dataloader, valid_dataloader, get_trans, get_caueeg2_datalist
from util import log_mlflow, log_reconstructions, log_spectral, ParseListAction, print_gpu_memory_report, setup_run_dir

# for reproducibility purposes set a seed
set_determinism(42)

import torch
import torch.nn.functional as F

if os.path.exists('/project'):
    base_path = '/project/'
    base_path_data = '/data/'
else:
    base_path = '/home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models'
    base_path_data = base_path

class ParseListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parsed_list = ast.literal_eval(values)
        setattr(namespace, self.dest, parsed_list)

def init_weights(m):
    """Initialize model weights for better training stability"""
    if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default="/home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/config/config_encoder_eeg.yaml",
        help="Path to config file with all the training parameters needed",
    )
    parser.add_argument(
        "--path_train_ids",
        type=str,
        default="/home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/data/ids/ids_sleep_edfx_cassette_double_train.csv",
    )

    parser.add_argument(
        "--path_valid_ids",
        type=str,
        default="/home/stud/timlin/bhome/DiffusionEEG/Synthetic-Sleep-EEG-Signal-Generation-using-Latent-Diffusion-Models/project/data/ids/ids_sleep_edfx_cassette_double_valid.csv",
    )
    parser.add_argument(
        "--path_cached_data",
        type=str,
        default="/data/pre",
    )

    parser.add_argument(
        "--path_pre_processed",
        type=str,
        default="/data/physionet-sleep-data-npy",
    )

    parser.add_argument(
        "--num_channels",
        type=str, action=ParseListAction
    )

    parser.add_argument(
        "--spe",
        type=str, 
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
    )
    parser.add_argument(
        "--type_dataset",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["edfx", "shhs", "shhsh", "caueeg2"],
    )
    
    # New argument for label filtering in CAUEEG2 dataset
    parser.add_argument(
        "--label_filter",
        type=str,
        help="Label filter for CAUEEG2 dataset. Can be 'hc' (healthy), 'mci', 'dementia', or a comma-separated list"
    )
    
    # Arguments for numerical stability
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=1.0,
        help="Maximum norm for gradient clipping"
    )
    
    parser.add_argument(
        "--spectral_cap",
        type=float,
        default=1e2,
        help="Maximum value for spectral weight to prevent instability"
    )
    
    # New stability parameters
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizers"
    )
    
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0.1,
        help="Scale factor for losses to prevent NaN"
    )
    
    parser.add_argument(
        "--clamp_z_mu",
        type=float,
        default=5.0,
        help="Clamp z_mu values to +/- this value"
    )
    
    parser.add_argument(
        "--clamp_z_sigma",
        type=float,
        default=10.0,
        help="Clamp z_sigma values between 0.1 and this value"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size for stability"
    )
    
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of warmup epochs for learning rate"
    )
    
    args = parser.parse_args()
    
    # Process label_filter if it's a comma-separated string
    if hasattr(args, 'label_filter') and args.label_filter and ',' in args.label_filter:
        args.label_filter = args.label_filter.split(',')
    
    return args


def main(args):
    config = OmegaConf.load(args.config_file)
    set_determinism(seed=config.train.seed)
    print_config()

    # Override batch size if specified
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
        print(f"Overriding batch size to: {config.train.batch_size}")

    # Add suffix to run_dir based on label filter for easier identification
    if args.dataset == "caueeg2" and hasattr(args, 'label_filter') and args.label_filter:
        if isinstance(args.label_filter, list):
            label_suffix = '_'.join(str(l) for l in args.label_filter)
        else:
            label_suffix = str(args.label_filter)
        config.train.run_dir = f"{config.train.run_dir}_label_{label_suffix}"
        print(f"Using modified run directory: {config.train.run_dir}")
        
    run_dir, resume = setup_run_dir(config=config, args=args, base_path=base_path)

    # Getting write training and validation data
    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    trans = get_trans(args.dataset)
    # Getting data loaders
    train_loader = train_dataloader(config=config, args=args, transforms_list=trans, dataset=args.dataset)
    val_loader = valid_dataloader(config=config, args=args, transforms_list=trans, dataset=args.dataset)

    # Defining device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Defining model
    autoencoder_args = config.autoencoderkl.params
    if args.num_channels is not None:
        autoencoder_args['num_channels'] = args.num_channels
    if args.latent_channels is not None:
        autoencoder_args['latent_channels'] = args.latent_channels

    print(f"Autoencoder args: {autoencoder_args}")

    model = AutoencoderKL(**autoencoder_args)
    # including extra parameters for the discriminator from a dictionary
    discriminator_dict = config.patchdiscriminator.params

    discriminator = PatchDiscriminator(**discriminator_dict)
    
    # Apply weight initialization for better stability
    if not resume:
        print("Applying custom weight initialization for stability")
        model.apply(init_weights)
        discriminator.apply(init_weights)

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        print("Putting the model to run in more that 1 GPU")
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)

    model = model.to(device)
    discriminator = discriminator.to(device)

    # Use AdamW instead of Adam for better regularization
    optimizer_g = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.models.optimizer_g_lr,
        weight_decay=args.weight_decay,
        eps=1e-8  # Increased epsilon for numerical stability
    )
    
    optimizer_d = torch.optim.AdamW(
        params=discriminator.parameters(),
        lr=config.models.optimizer_d_lr,
        weight_decay=args.weight_decay,
        eps=1e-8  # Increased epsilon for numerical stability
    )
    
    # Add learning rate schedulers for better stability
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='min', factor=0.5, patience=5
    )
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode='min', factor=0.5, patience=5
    )

    # %%
    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = config.models.adv_weight
    jukebox_loss = JukeboxLoss(spatial_dims=1, reduction="sum")

    # ## Model Training
    kl_weight = config.models.kl_weight
    n_epochs = config.train.n_epochs
    val_interval = config.train.val_interval
    
    # Cap spectral weight to prevent instability
    spectral_weight = config.models.spectral_weight
    spectral_weight = min(spectral_weight, args.spectral_cap)
    print(f"Using spectral weight: {spectral_weight} (capped at {args.spectral_cap})")
    
    # Initialize loss tracking lists
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    epoch_spectral_loss_list = []
    val_recon_epoch_loss_list = []
    best_loss = float("inf")
    start_epoch = 0
    
    # Initialize EMA trackers for smoother loss trends
    ema_recon_loss = None
    ema_gen_loss = None
    ema_disc_loss = None
    ema_spectral_loss = None
    ema_decay = 0.95  # Decay factor for EMA
    
    # Function to update EMA values
    def update_ema(current_ema, new_value, decay=0.95):
        if current_ema is None:
            return new_value
        return current_ema * decay + new_value * (1 - decay)
    
    # Function to get learning rate factor for warmup
    def get_lr_factor(current_epoch, warmup_epochs):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        return 1.0
    
    # Resume from checkpoint if available
    if resume:
        print(f"Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        model.load_state_dict(checkpoint["state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        init_batch = checkpoint["init_batch"]
        
        # Try to load EMA values if they exist
        if "ema_recon_loss" in checkpoint:
            ema_recon_loss = checkpoint["ema_recon_loss"]
            ema_gen_loss = checkpoint["ema_gen_loss"]
            ema_disc_loss = checkpoint["ema_disc_loss"]
            ema_spectral_loss = checkpoint["ema_spectral_loss"]
    else:
        print(f"No checkpoint found.")

    total_start = time.time()

    # Get the initial batch and fix dimensions for CAUEEG2 data
    init_batch_raw = first(train_loader)['eeg'].to(device)
    
    # Handle data dimensions
    if args.dataset == "caueeg2":
        # For CAUEEG2, reshape from [batch, 1, 19, 1000] to [batch, 19, 1000]
        init_batch = init_batch_raw.squeeze(1)
        # Apply padding slicing if needed
        if init_batch.dim() == 3:  # Check if dimensions are correct
            init_batch = init_batch[:, :, 36:-36]
    else:
        # For other datasets, keep the original behavior
        init_batch = init_batch_raw[:, :, 36:-36]

    # Define epsilon for numerical stability
    eps = 1e-8

    for epoch in range(start_epoch, n_epochs):
        model.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        spectral_epoch_loss = 0
        
        # Count batches with NaN values for monitoring
        nan_batches = 0
        total_batches = 0
        
        # Get learning rate warmup factor
        lr_factor = get_lr_factor(epoch, args.warmup_epochs)
        if lr_factor < 1.0:
            # Apply warmup to learning rates
            for param_group in optimizer_g.param_groups:
                param_group['lr'] = config.models.optimizer_g_lr * lr_factor
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = config.models.optimizer_d_lr * lr_factor
            print(f"Warmup epoch {epoch+1}/{args.warmup_epochs}, LR factor: {lr_factor:.4f}")
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=150)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            total_batches += 1
            try:
                eeg_data_raw = batch['eeg'].to(device)
                
                if args.dataset == "caueeg2":
                    # Simply transpose the data to match what the model expects
                    # From [batch, 1, 19, 1000] to [batch, 19, 1000]
                    eeg_data = eeg_data_raw.squeeze(1)
                    
                    # Print shape for debugging
                    if step == 0 and epoch == 0:
                        print(f"EEG data shape after processing: {eeg_data.shape}")
                        print(f"Model expects input with 19 channels (current first layer weight: [32, 19, 3])")
                else:
                    eeg_data = eeg_data_raw
                    
                # Print shape for debugging
                if step == 0 and epoch == 0:
                    print(f"EEG data shape after processing: {eeg_data.shape}")

                # Check for NaN in input data
                if torch.isnan(eeg_data).any():
                    print(f"Warning: NaN detected in input data (batch {step}), skipping")
                    nan_batches += 1
                    continue

                optimizer_g.zero_grad(set_to_none=True)
                reconstruction, z_mu, z_sigma = model(eeg_data)

                # Check for NaN in model outputs
                if torch.isnan(reconstruction).any() or torch.isnan(z_mu).any() or torch.isnan(z_sigma).any():
                    print(f"Warning: NaN detected in model outputs (batch {step}), skipping")
                    nan_batches += 1
                    continue
                
                # Clamp z_mu and z_sigma values to prevent extreme values
                z_mu = torch.clamp(z_mu, -args.clamp_z_mu, args.clamp_z_mu)
                z_sigma = torch.clamp(z_sigma, 0.1, args.clamp_z_sigma)

                # Calculate losses with numerical stability
                recons_loss = l1_loss(reconstruction.float(), eeg_data.float())
                # Scale down loss for stability
                recons_loss = recons_loss * args.loss_scale
                
                # Use try-except blocks for potentially unstable operations
                try:
                    recons_spectral = jukebox_loss(reconstruction.float(), eeg_data.float())
                    # Scale down for stability
                    recons_spectral = recons_spectral * args.loss_scale
                    
                    if torch.isnan(recons_spectral):
                        print(f"Warning: NaN in spectral loss, using default value")
                        recons_spectral = torch.tensor(1.0 * args.loss_scale, device=device)
                except Exception as e:
                    print(f"Exception in spectral loss: {e}, using default value")
                    recons_spectral = torch.tensor(1.0 * args.loss_scale, device=device)
                
                # Calculate KL loss with numerical stability
                try:
                    # Use a more numerically stable version
                    kl_loss = 0.5 * torch.sum(
                        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1, 
                        dim=[1]
                    )
                    kl_loss = torch.mean(kl_loss)  # Use mean instead of sum for better stability
                    # Scale down for stability
                    kl_loss = kl_loss * args.loss_scale
                    
                    if torch.isnan(kl_loss):
                        print(f"Warning: NaN in KL loss, using default value")
                        kl_loss = torch.tensor(0.01 * args.loss_scale, device=device)
                except Exception as e:
                    print(f"Exception in KL loss: {e}, using default value")
                    kl_loss = torch.tensor(0.01 * args.loss_scale, device=device)

                # Adversarial loss
                try:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    # Scale down for stability
                    generator_loss = generator_loss * args.loss_scale
                    
                    if torch.isnan(generator_loss):
                        print(f"Warning: NaN in generator loss, using default value")
                        generator_loss = torch.tensor(0.1 * args.loss_scale, device=device)
                except Exception as e:
                    print(f"Exception in generator loss: {e}, using default value")
                    generator_loss = torch.tensor(0.1 * args.loss_scale, device=device)

                # Combine losses with spectral weight cap and dynamic factor
                if args.spe == "spectral":
                    # Start with very small spectral weight and gradually increase
                    effective_spectral_weight = min(spectral_weight, 
                                                   spectral_weight * lr_factor * 0.1)  # Further reduce in early epochs
                    
                    loss_g = (recons_loss + 
                             kl_weight * kl_loss + 
                             adv_weight * generator_loss + 
                             effective_spectral_weight * recons_spectral)
                else:
                    loss_g = recons_loss + kl_weight * kl_loss + adv_weight * generator_loss 
                
                # Skip if loss is NaN
                if torch.isnan(loss_g):
                    print(f"Warning: NaN in generator combined loss (batch {step}), skipping")
                    nan_batches += 1
                    continue
                
                # Backward pass with gradient clipping
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                
                # Check for NaN in gradients before step
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                        
                if has_nan_grad:
                    print(f"Warning: NaN in generator gradients (batch {step}), skipping optimizer step")
                    nan_batches += 1
                    continue
                    
                optimizer_g.step()

                # Discriminator part
                optimizer_d.zero_grad(set_to_none=True)

                try:
                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(eeg_data.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                    
                    # Scale down for stability
                    discriminator_loss = discriminator_loss * args.loss_scale
                    
                    if torch.isnan(discriminator_loss):
                        print(f"Warning: NaN in discriminator loss (batch {step}), skipping")
                        discriminator_loss = torch.tensor(0.1 * args.loss_scale, device=device)
                        
                    loss_d = adv_weight * discriminator_loss
                    
                    if torch.isnan(loss_d):
                        print(f"Warning: NaN in weighted discriminator loss, skipping")
                        continue
                        
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=args.clip_grad)
                    
                    # Check for NaN in gradients before step
                    has_nan_grad = False
                    for param in discriminator.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            has_nan_grad = True
                            break
                            
                    if has_nan_grad:
                        print(f"Warning: NaN in discriminator gradients (batch {step}), skipping optimizer step")
                        continue
                        
                    optimizer_d.step()
                except Exception as e:
                    print(f"Exception in discriminator update: {e}, skipping")
                    continue

                # Update metrics using original loss values
                epoch_loss += recons_loss.item() / args.loss_scale  # Scale back to original magnitude
                gen_epoch_loss += generator_loss.item() / args.loss_scale
                disc_epoch_loss += discriminator_loss.item() / args.loss_scale
                spectral_epoch_loss += recons_spectral.item() / args.loss_scale
                
                # Update EMA values for smoother loss tracking
                ema_recon_loss = update_ema(ema_recon_loss, recons_loss.item() / args.loss_scale, ema_decay)
                ema_gen_loss = update_ema(ema_gen_loss, generator_loss.item() / args.loss_scale, ema_decay)
                ema_disc_loss = update_ema(ema_disc_loss, discriminator_loss.item() / args.loss_scale, ema_decay)
                ema_spectral_loss = update_ema(ema_spectral_loss, recons_spectral.item() / args.loss_scale, ema_decay)

                # Show EMA values in progress bar for smoother updates
                progress_bar.set_postfix(
                    {
                        "l1_loss": ema_recon_loss,
                        "gen_loss": ema_gen_loss,
                        "disc_loss": ema_disc_loss,
                        "spec_loss": ema_spectral_loss,
                    }
                )
            except Exception as e:
                print(f"Unexpected error in batch {step}: {e}, skipping")
                nan_batches += 1
                continue

            if (epoch + 1) % val_interval == 0:
                with torch.no_grad():
                    # For visualization, we may need to adjust slicing based on data format
                    if args.dataset == "caueeg2":
                        vis_eeg = eeg_data
                        vis_recon = reconstruction
                        # Add slicing if needed
                        if vis_eeg.dim() == 3 and vis_eeg.shape[2] > 72:  # Check if padding should be removed
                            vis_eeg = vis_eeg[:, :, 36:-36]
                            vis_recon = vis_recon[:, :, 36:-36]
                    else:
                        vis_eeg = eeg_data[:, :, 36:-36]
                        vis_recon = reconstruction[:, :, 36:-36]
                        
                    log_reconstructions(img=vis_eeg,
                                        recons=vis_recon,
                                        writer=writer_train,
                                        step=epoch+1,
                                        name="RECONSTRUCTION_TRAIN",
                                        run_dir=run_dir)
        
                    reconstruction_init, _, _ = model(init_batch)

                    # For visualization of init batch
                    if args.dataset == "caueeg2":
                        vis_init = init_batch
                        vis_init_recon = reconstruction_init
                        # Check if slicing is needed based on the shape
                        if vis_init.dim() == 3 and vis_init.shape[2] > 72:
                            # Torch tensors don't have a way to check if slicing was applied
                            # So just check the actual size to determine if slicing is needed
                            vis_init = vis_init[:, :, 36:-36]
                            vis_init_recon = vis_init_recon[:, :, 36:-36]
                    else:
                        vis_init = init_batch[:, :, 36:-36]
                        vis_init_recon = reconstruction_init[:, :, 36:-36]

                    log_reconstructions(img=vis_init,
                                        recons=vis_init_recon,
                                        writer=writer_train,
                                        step=epoch+1,
                                        name="RECONSTRUCTION_TRAIN_OVERTIME",
                                        run_dir=run_dir)

                    log_spectral(eeg=vis_eeg,
                                 recons=vis_init_recon,
                                 writer=writer_val,
                                 step=epoch+1,
                                 name="SPECTROGRAM_OVERTIME", 
                                 run_dir=run_dir)

        # Log NaN batch statistics
        nan_percentage = (nan_batches / max(total_batches, 1)) * 100
        print(f"Epoch {epoch} had {nan_batches}/{total_batches} batches with NaN values ({nan_percentage:.2f}%)")
        
        # Skip logging if all batches had NaN
        if total_batches == nan_batches:
            print(f"Warning: All batches in epoch {epoch} had NaN values, skipping TensorBoard logging")
            continue
            
        writer_train.add_scalar("loss_g", gen_epoch_loss / max(step + 1 - nan_batches, 1), epoch * len(train_loader) + step)
        writer_train.add_scalar("loss_d", disc_epoch_loss / max(step + 1 - nan_batches, 1), epoch * len(train_loader) + step)
        writer_train.add_scalar("recons_loss", epoch_loss / max(step + 1 - nan_batches, 1), epoch * len(train_loader) + step)
        writer_train.add_scalar("recons_spectral", spectral_epoch_loss / max(step + 1 - nan_batches, 1), epoch * len(train_loader) + step)
        writer_train.add_scalar("nan_percentage", nan_percentage, epoch)
        writer_train.add_scalar("lr_factor", lr_factor, epoch)

        epoch_recon_loss_list.append(epoch_loss / max(step + 1 - nan_batches, 1))
        epoch_gen_loss_list.append(gen_epoch_loss / max(step + 1 - nan_batches, 1))
        epoch_disc_loss_list.append(disc_epoch_loss / max(step + 1 - nan_batches, 1))
        epoch_spectral_loss_list.append(spectral_epoch_loss / max(step + 1 - nan_batches, 1))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    try:
                        eeg_data_raw = batch['eeg'].to(device)
                        
                        # Fix dimensions for CAUEEG2 data during validation
                        if args.dataset == "caueeg2":
                            eeg_data = eeg_data_raw.squeeze(1)
                        else:
                            eeg_data = eeg_data_raw
                            
                        # Skip if NaN in input
                        if torch.isnan(eeg_data).any():
                            print(f"Warning: NaN in validation input (batch {val_step}), skipping")
                            continue
                            
                        reconstruction_eeg, _, _ = model(eeg_data)
                        
                        # Skip if NaN in output
                        if torch.isnan(reconstruction_eeg).any():
                            print(f"Warning: NaN in validation output (batch {val_step}), skipping")
                            continue

                        # Handle visualization
                        if args.dataset == "caueeg2":
                            vis_val_eeg = eeg_data
                            vis_val_recon = reconstruction_eeg
                            # Add slicing if needed
                            if vis_val_eeg.dim() == 3 and vis_val_eeg.shape[2] > 72:
                                vis_val_eeg = vis_val_eeg[:, :, 36:-36]
                                vis_val_recon = vis_val_recon[:, :, 36:-36]
                        else:
                            vis_val_eeg = eeg_data[:, :, 36:-36]
                            vis_val_recon = reconstruction_eeg[:, :, 36:-36]

                        log_reconstructions(img=vis_val_eeg,
                                            recons=vis_val_recon,
                                            writer=writer_val,
                                            step=epoch+1,
                                            name="RECONSTRUCTION_VAL",
                                            run_dir=run_dir)

                        log_spectral(eeg=vis_val_eeg,
                                    recons=vis_val_recon,
                                    writer=writer_val,
                                    step=epoch+1,
                                    name="SPECTROGRAM_VAL", 
                                    run_dir=run_dir)

                        recons_loss = l1_loss(reconstruction_eeg.float(),
                                            eeg_data.float())

                        val_loss += recons_loss.item()
                        val_batches += 1

                        if val_batches > 0 and val_loss / val_batches <= best_loss:
                            print(f"New best val loss {val_loss / val_batches}")
                            best_loss = val_loss / val_batches
                            torch.save(model.state_dict(), str(run_dir / "best_model.pth"))
                    except Exception as e:
                        print(f"Error in validation batch {val_step}: {e}, skipping")
                        continue
                    
                    # Save checkpoint
                    checkpoint = {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "optimizer_g": optimizer_g.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        "best_loss": best_loss,
                        "init_batch": init_batch,
                        "ema_recon_loss": ema_recon_loss,
                        "ema_gen_loss": ema_gen_loss,
                        "ema_disc_loss": ema_disc_loss,
                        "ema_spectral_loss": ema_spectral_loss
                    }
                    torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

                if val_batches > 0:
                    final_val_loss = val_loss / val_batches
                    writer_val.add_scalar("recons_loss", final_val_loss, epoch * len(val_loader) + step)
                    val_recon_epoch_loss_list.append(final_val_loss)
                    
                    # Store old learning rates for comparison
                    old_lr_g = optimizer_g.param_groups[0]['lr']
                    old_lr_d = optimizer_d.param_groups[0]['lr']

                    # Update learning rate schedulers only after warmup
                    if epoch >= args.warmup_epochs:
                        scheduler_g.step(final_val_loss)
                        scheduler_d.step(final_val_loss)

                    # Manual verbose logging
                    new_lr_g = optimizer_g.param_groups[0]['lr']
                    new_lr_d = optimizer_d.param_groups[0]['lr']
                    if old_lr_g != new_lr_g:
                        print(f"Generator learning rate changed from {old_lr_g} to {new_lr_g}")
                    if old_lr_d != new_lr_d:
                        print(f"Discriminator learning rate changed from {old_lr_d} to {new_lr_d}")

    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")
    torch.save(model.state_dict(), str(run_dir / "final_model.pth"))

    if val_batches > 0:
        log_mlflow(
            model=model,
            config=config,
            args=args,
            run_dir=run_dir,
            val_loss=val_loss / val_batches,
        )
    else:
        print("Warning: No valid validation batches, skipping MLflow logging")


if __name__ == "__main__":
    args = parse_args()
    main(args)