import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
import sys
from typing import List, Dict, Any, Optional, Tuple


class BatchedPatientDiffusionWrapper(nn.Module):
    """
    A wrapper model that processes batches of patients through a UNet diffusion model.
    Each patient's epochs are kept separate, but multiple patients are processed in parallel.
    """
    def __init__(self, unet_model):
        super().__init__()
        self.unet_model = unet_model
        
    def forward(self, patient_batch: List[Dict[str, torch.Tensor]], timesteps=None, context=None, y=None, **kwargs):
        """
        Process a batch of patients, keeping each patient's epochs separate.
        
        Args:
            patient_batch: List of dictionaries, each containing one patient's data
                           Each dict has 'eeg' with shape [epochs, channels, timepoints]
            timesteps: Single timestep or batch of timesteps
            context: Optional conditioning
            y: Optional class labels
            
        Returns:
            List of processed patient data, each with shape [epochs, channels, timepoints]
        """
        # Extract the 'eeg' data from each patient dict
        all_patient_outputs = []
        
        for i, patient_data in enumerate(patient_batch):
            # Get this patient's EEG data
            x = patient_data['eeg']
            num_epochs = x.shape[0]
            
            # Handle timesteps for this patient
            patient_timesteps = None
            if timesteps is not None:
                # If timesteps is a single value, repeat it for all epochs
                if timesteps.shape[0] == 1:
                    patient_timesteps = timesteps.repeat(num_epochs)
                else:
                    # If timesteps is a batch, use the corresponding one
                    patient_timesteps = timesteps[i:i+1].repeat(num_epochs)
            
            # Process each epoch for this patient
            outputs = []
            for j in range(num_epochs):
                # Extract a single epoch
                epoch = x[j:j+1]
                
                # Get the corresponding timestep
                if patient_timesteps is not None:
                    t = patient_timesteps[j:j+1]
                else:
                    t = None
                
                # Process through UNet
                output = self.unet_model(epoch, t, context, y, **kwargs)
                outputs.append(output)
            
            # Stack all epochs for this patient
            patient_output = torch.cat(outputs, dim=0)
            
            # Add processed data back to patient dict
            patient_result = patient_data.copy()
            patient_result['processed_eeg'] = patient_output
            
            all_patient_outputs.append(patient_result)
        
        return all_patient_outputs

def collate_patients_fn(batch):
    """
    Custom collate function that keeps each patient's data separate.
    Instead of stacking tensors, it returns a list of patient dictionaries.
    
    Args:
        batch: List of dictionaries, each containing one patient's data
        
    Returns:
        List of dictionaries, preserving patient structure
    """
    # Return batch as is, without stacking
    return batch

def train_diffusion_batched_patients(
    model,
    scheduler,
    start_epoch,
    best_loss,
    train_loader,
    val_loader,
    optimizer,
    n_epochs,
    eval_freq,
    writer_train,
    writer_val,
    device,
    run_dir,
    inferer,
    spectral_loss=False,
    spectral_weight=1e-6,
):
    """
    Training function that handles batches of patients while keeping each patient's epochs separate.
    
    Args:
        model: The BatchedPatientDiffusionWrapper model
        scheduler: The diffusion scheduler
        start_epoch: Starting epoch number
        best_loss: Best validation loss so far
        train_loader: DataLoader that returns batches of patients
        val_loader: DataLoader for validation
        optimizer: Optimizer
        n_epochs: Total number of epochs
        eval_freq: Evaluate every n epochs
        writer_train: TensorBoard writer for training
        writer_val: TensorBoard writer for validation
        device: Device to use
        run_dir: Directory to save models
        inferer: DiffusionInferer object
        spectral_loss: Whether to use spectral loss
        spectral_weight: Weight for spectral loss
    """
    # Import here to ensure it's in the local namespace
    from torch.cuda.amp import GradScaler, autocast
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Compute the next power of 2 for the timepoints dimension (for FFT)
    timepoints = 1000  # Your timepoints dimension
    fft_size = 2**10  # 1024, the next power of 2 after 1000
    
    # Print training setup details
    print(f"\n{'='*80}")
    print(f"Starting training of Batched Patient-Level Diffusion Model")
    print(f"Epochs: {n_epochs}, Starting from: {start_epoch}")
    print(f"Device: {device}")
    print(f"Spectral loss: {'Enabled' if spectral_loss else 'Disabled'}")
    if spectral_loss:
        print(f"Spectral weight: {spectral_weight}")
    print(f"Training set size: {len(train_loader)} batches")
    print(f"Batch size: {train_loader.batch_size} patients per batch")
    print(f"Validation set size: {len(val_loader)} batches")
    print(f"{'='*80}\n")
    
    # Track total training time
    start_time = time.time()
    
    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0
        total_patients = 0
        total_epochs = 0
        
        print(f"Epoch {epoch+1}/{n_epochs} - Training:")
        
        for step, patient_batch in enumerate(train_loader):
            # Move all patient data to device
            for i in range(len(patient_batch)):
                patient_batch[i]['eeg'] = patient_batch[i]['eeg'].to(device)
            
            # Count patients and total epochs
            batch_patients = len(patient_batch)
            batch_epochs = sum(p['eeg'].shape[0] for p in patient_batch)
            total_patients += batch_patients
            total_epochs += batch_epochs
            
            # Sample a single timestep for the batch
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (1,)).long().to(device)
            
            # Start the forward/backward pass
            optimizer.zero_grad()
            
            with autocast(enabled=True):
                # Process each patient individually but in parallel
                all_noise = []
                all_noise_preds = []
                
                # Create noise and add it to each patient's data
                for i, patient_data in enumerate(patient_batch):
                    # Get patient's data
                    images = patient_data['eeg']
                    num_epochs = images.shape[0]
                    
                    # Create noise for this patient
                    noise = torch.randn_like(images).to(device)
                    
                    # Forward diffusion to add noise
                    noisy_images = scheduler.add_noise(images, noise, timesteps.repeat(num_epochs))
                    
                    # Update patient data for model input
                    patient_batch[i]['eeg'] = noisy_images
                    
                    # Store noise for loss calculation
                    all_noise.append(noise)
                
                # Predict noise for all patients
                with torch.cuda.amp.autocast():
                    processed_patients = model(patient_batch, timesteps)
                
                # Extract predictions and calculate loss
                total_loss = 0
                for i, processed_patient in enumerate(processed_patients):
                    noise_pred = processed_patient['processed_eeg']
                    noise = all_noise[i]
                    
                    # Calculate loss for this patient
                    if spectral_loss:
                        # MSE loss in time domain
                        time_loss = torch.nn.functional.mse_loss(noise_pred, noise)
                        
                        # Disable autocast for FFT computation
                        with autocast(enabled=False):
                            # Pad signals to power of 2 for FFT
                            noise_padded = F.pad(noise, (0, fft_size - noise.shape[2]))
                            pred_padded = F.pad(noise_pred, (0, fft_size - noise_pred.shape[2]))
                            
                            # Compute FFT
                            noise_fft = torch.fft.rfft(noise_padded, dim=2)
                            pred_fft = torch.fft.rfft(pred_padded, dim=2)
                            
                            # Compute magnitude and phase difference
                            spec_loss = torch.mean(torch.abs(noise_fft - pred_fft))
                        
                        # Combined loss for this patient
                        patient_loss = time_loss + spectral_weight * spec_loss
                    else:
                        # Regular MSE loss
                        patient_loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    
                    # Add to total loss
                    total_loss += patient_loss
                
                # Average loss over patients in batch
                loss = total_loss / batch_patients
            
            # Backpropagate and update model
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss * batch_patients  # Scale by number of patients
            
            # Print progress
            if (step + 1) % 5 == 0 or step == 0:
                print(f"  Batch {step+1}/{len(train_loader)} - Loss: {batch_loss:.6f} - Patients: {batch_patients}, Epochs: {batch_epochs}")
                sys.stdout.flush()
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / total_patients
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch loss to TensorBoard
        writer_train.add_scalar("loss", avg_loss, epoch)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{n_epochs} completed - Avg Loss: {avg_loss:.6f} - Time: {epoch_time:.2f}s")
        print(f"Processed {total_patients} patients with {total_epochs} total epochs")
        
        # Evaluate periodically
        if (epoch + 1) % eval_freq == 0:
            print(f"\nEvaluating model at epoch {epoch+1}...")
            val_loss = eval_diffusion_batched_patients(
                model=model,
                scheduler=scheduler,
                data_loader=val_loader,
                device=device,
                inferer=inferer,
                epoch=epoch,
                writer=writer_val,
                spectral_loss=spectral_loss,
                spectral_weight=spectral_weight,
                fft_size=fft_size,
            )
            
            # Check if this is the best model so far
            is_best = val_loss < best_loss
            
            # Save the best model
            if is_best:
                best_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss,
                    },
                    str(run_dir / "best_model.pth"),
                )
                print(f"New best model saved with validation loss: {best_loss:.6f}")
            
            # Save checkpoint
            checkpoint_path = str(run_dir / f"checkpoint_epoch{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                checkpoint_path
            )
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Print divider for better readability
            print(f"\n{'-'*80}\n")
        
        # Force flush output for log files
        sys.stdout.flush()
    
    # Print training summary
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Model saved to: {run_dir}")
    print(f"{'='*80}\n")
    
    return best_loss

def eval_diffusion_batched_patients(
    model,
    scheduler,
    data_loader,
    device,
    inferer,
    epoch,
    writer,
    spectral_loss=False,
    spectral_weight=1e-6,
    fft_size=1024,
):
    """
    Evaluation function for batched patient-level diffusion models.
    """
    # Import here to ensure it's in the local namespace
    from torch.cuda.amp import GradScaler, autocast
    
    model.eval()
    val_loss = 0
    total_patients = 0
    total_epochs = 0
    
    print(f"\nValidation - Epoch {epoch+1}:")
    
    with torch.no_grad():
        for step, patient_batch in enumerate(data_loader):
            # Move all patient data to device
            for i in range(len(patient_batch)):
                patient_batch[i]['eeg'] = patient_batch[i]['eeg'].to(device)
            
            # Count patients and total epochs
            batch_patients = len(patient_batch)
            batch_epochs = sum(p['eeg'].shape[0] for p in patient_batch)
            total_patients += batch_patients
            total_epochs += batch_epochs
            
            # Sample a single timestep for the batch
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (1,)).long().to(device)
            
            # Process each patient individually but in parallel
            all_noise = []
            
            # Create noise and add it to each patient's data
            for i, patient_data in enumerate(patient_batch):
                # Get patient's data
                images = patient_data['eeg']
                num_epochs = images.shape[0]
                
                # Create noise for this patient
                noise = torch.randn_like(images).to(device)
                
                # Forward diffusion to add noise
                noisy_images = scheduler.add_noise(images, noise, timesteps.repeat(num_epochs))
                
                # Update patient data for model input
                patient_batch[i]['eeg'] = noisy_images
                
                # Store noise for loss calculation
                all_noise.append(noise)
            
            # Predict noise for all patients
            with torch.cuda.amp.autocast():
                processed_patients = model(patient_batch, timesteps)
            
            # Extract predictions and calculate loss
            batch_loss = 0
            for i, processed_patient in enumerate(processed_patients):
                noise_pred = processed_patient['processed_eeg']
                noise = all_noise[i]
                
                # Calculate loss for this patient
                if spectral_loss:
                    # MSE loss in time domain
                    time_loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='sum')
                    
                    # Disable autocast for FFT computation
                    with autocast(enabled=False):
                        # Pad signals to power of 2 for FFT
                        noise_padded = F.pad(noise, (0, fft_size - noise.shape[2]))
                        pred_padded = F.pad(noise_pred, (0, fft_size - noise_pred.shape[2]))
                        
                        # Compute FFT
                        noise_fft = torch.fft.rfft(noise_padded, dim=2)
                        pred_fft = torch.fft.rfft(pred_padded, dim=2)
                        
                        # Compute magnitude and phase difference
                        spec_loss = torch.sum(torch.abs(noise_fft - pred_fft))
                    
                    # Combined loss for this patient
                    patient_loss = time_loss + spectral_weight * spec_loss
                else:
                    # Regular MSE loss
                    patient_loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='sum')
                
                # Add to batch loss
                batch_loss += patient_loss.item()
            
            val_loss += batch_loss
            
            # Print progress
            if (step + 1) % 5 == 0 or step == 0:
                print(f"  Batch {step+1}/{len(data_loader)} - Patients: {batch_patients}, Epochs: {batch_epochs}")
                sys.stdout.flush()
    
    # Calculate average loss per epoch
    avg_loss = val_loss / total_epochs
    
    # Log to tensorboard
    writer.add_scalar("loss", avg_loss, epoch)
    
    # Print validation summary
    print(f"Validation complete - Avg Loss: {avg_loss:.6f}")
    print(f"Processed {total_patients} patients with {total_epochs} total epochs")
    sys.stdout.flush()
    
    return avg_loss




class PatientDiffusionWrapper(nn.Module):
    """
    A wrapper model that processes all epochs from a single patient through a UNet diffusion model.
    This wrapper handles variable numbers of epochs per patient.
    """
    def __init__(self, unet_model):
        super().__init__()
        self.unet_model = unet_model
        
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Process all epochs from a single patient.
        
        Args:
            x: Tensor with shape [epochs, channels, timepoints] where epochs varies per patient
            timesteps: Tensor with shape [1] containing the current timestep
            context: Optional conditioning
            y: Optional class labels
            
        Returns:
            Tensor with shape [epochs, channels, timepoints]
        """
        # Get the number of epochs for this patient
        num_epochs = x.shape[0]
        
        # Expand timesteps to match the number of epochs
        if timesteps is not None:
            # If timesteps is a single value, expand it to match the number of epochs
            if timesteps.shape[0] == 1:
                timesteps = timesteps.repeat(num_epochs)
        
        # Process each epoch through the UNet model
        outputs = []
        for i in range(num_epochs):
            # Extract a single epoch (add batch dimension)
            epoch = x[i:i+1]
            
            # Get the corresponding timestep
            if timesteps is not None:
                t = timesteps[i:i+1]
            else:
                t = None
            
            # Process through UNet
            output = self.unet_model(epoch, t, context, y, **kwargs)
            
            # Store the result
            outputs.append(output)
        
        # Stack the results along the epoch dimension
        return torch.cat(outputs, dim=0)

def train_diffusion_patient(
    model,
    scheduler,
    start_epoch,
    best_loss,
    train_loader,
    val_loader,
    optimizer,
    n_epochs,
    eval_freq,
    writer_train,
    writer_val,
    device,
    run_dir,
    inferer,
    spectral_loss=False,
    spectral_weight=1e-6,
):
    """
    Modified training function that handles patient-level data (all epochs at once).
    Added detailed console output for monitoring in log files.
    
    Each batch contains all epochs from a single patient.
    """
    # Import here to ensure it's in the local namespace
    from torch.cuda.amp import GradScaler, autocast
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Compute the next power of 2 for the timepoints dimension (for FFT)
    timepoints = 1000  # Your timepoints dimension
    fft_size = 2**10  # 1024, the next power of 2 after 1000
    
    # Print training setup details
    print(f"\n{'='*80}")
    print(f"Starting training of Patient-Level Diffusion Model")
    print(f"Epochs: {n_epochs}, Starting from: {start_epoch}")
    print(f"Device: {device}")
    print(f"Spectral loss: {'Enabled' if spectral_loss else 'Disabled'}")
    if spectral_loss:
        print(f"Spectral weight: {spectral_weight}")
    print(f"Training set size: {len(train_loader)} patients")
    print(f"Validation set size: {len(val_loader)} patients")
    print(f"{'='*80}\n")
    
    # Track total training time
    start_time = time.time()
    
    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0
        num_patients = 0
        
        print(f"Epoch {epoch+1}/{n_epochs} - Training:")
        
        for step, batch in enumerate(train_loader):
            # Get the patient's EEG data (shape: [epochs, channels, timepoints])
            images = batch["eeg"].to(device)
            
            # Create a batch of the same noise for all epochs
            num_epochs = images.shape[0]
            noise = torch.randn_like(images).to(device)
            
            # Sample a single timestep for all epochs
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (1,)).long().to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=True):
                # Forward diffusion to add noise
                noisy_images = scheduler.add_noise(images, noise, timesteps.repeat(num_epochs))
                
                # Predict the noise with the model
                with torch.cuda.amp.autocast():
                    noise_pred = model(noisy_images, timesteps, None)
                
                # Calculate loss
                if spectral_loss:
                    # MSE loss in time domain
                    time_loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    
                    # Disable autocast for FFT computation to avoid half-precision issue
                    with autocast(enabled=False):
                        # Pad signals to power of 2 for FFT
                        noise_padded = F.pad(noise, (0, fft_size - noise.shape[2]))
                        pred_padded = F.pad(noise_pred, (0, fft_size - noise_pred.shape[2]))
                        
                        # Compute FFT
                        noise_fft = torch.fft.rfft(noise_padded, dim=2)
                        pred_fft = torch.fft.rfft(pred_padded, dim=2)
                        
                        # Compute magnitude and phase difference
                        spec_loss = torch.mean(torch.abs(noise_fft - pred_fft))
                    
                    # Combined loss
                    loss = time_loss + spectral_weight * spec_loss
                else:
                    # Regular MSE loss
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Backpropagate and update model
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            num_patients += 1
            
            # Print progress every 10 batches
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  Batch {step+1}/{len(train_loader)} - Loss: {batch_loss:.6f} - Patient epochs: {num_epochs}")
                sys.stdout.flush()  # Force output to be written to file immediately
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_patients
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch loss to TensorBoard
        writer_train.add_scalar("loss", avg_loss, epoch)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{n_epochs} completed - Avg Loss: {avg_loss:.6f} - Time: {epoch_time:.2f} seconds")
        
        # Evaluate periodically
        if (epoch + 1) % eval_freq == 0:
            print(f"\nEvaluating model at epoch {epoch+1}...")
            val_loss = eval_diffusion_patient(
                model=model,
                scheduler=scheduler,
                data_loader=val_loader,
                device=device,
                inferer=inferer,
                epoch=epoch,
                writer=writer_val,
                spectral_loss=spectral_loss,
                spectral_weight=spectral_weight,
                fft_size=fft_size,
            )
            
            # Check if this is the best model so far
            is_best = val_loss < best_loss
            
            # Save the best model
            if is_best:
                best_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss,
                    },
                    str(run_dir / "best_model.pth"),
                )
                print(f"New best model saved with validation loss: {best_loss:.6f}")
            
            # Save checkpoint
            checkpoint_path = str(run_dir / f"checkpoint_epoch{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                checkpoint_path
            )
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Print divider for better readability
            print(f"\n{'-'*80}\n")
        
        # Force flush output for log files
        sys.stdout.flush()
    
    # Print training summary
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Model saved to: {run_dir}")
    print(f"{'='*80}\n")
    
    return best_loss

def eval_diffusion_patient(
    model,
    scheduler,
    data_loader,
    device,
    inferer,
    epoch,
    writer,
    spectral_loss=False,
    spectral_weight=1e-6,
    fft_size=1024,
):
    """
    Evaluation function for patient-level diffusion models.
    Added detailed console output for monitoring in log files.
    """
    # Import here to ensure it's in the local namespace
    from torch.cuda.amp import GradScaler, autocast
    
    model.eval()
    val_loss = 0
    total_samples = 0
    
    print(f"\nValidation - Epoch {epoch+1}:")
    
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            # Get the patient's EEG data (shape: [epochs, channels, timepoints])
            images = batch["eeg"].to(device)
            num_epochs = images.shape[0]
            total_samples += num_epochs
            
            # Create noise and sample timesteps
            noise = torch.randn_like(images).to(device)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (1,)).long().to(device)
            
            # Add noise to images
            noisy_images = scheduler.add_noise(images, noise, timesteps.repeat(num_epochs))
            
            # Predict noise
            with torch.cuda.amp.autocast():
                noise_pred = model(noisy_images, timesteps, None)
            
            # Calculate loss
            if spectral_loss:
                # MSE loss in time domain
                time_loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='sum')
                
                # Disable autocast for FFT computation
                with autocast(enabled=False):
                    # Pad signals to power of 2 for FFT
                    noise_padded = F.pad(noise, (0, fft_size - noise.shape[2]))
                    pred_padded = F.pad(noise_pred, (0, fft_size - noise_pred.shape[2]))
                    
                    # Compute FFT
                    noise_fft = torch.fft.rfft(noise_padded, dim=2)
                    pred_fft = torch.fft.rfft(pred_padded, dim=2)
                    
                    # Compute magnitude and phase difference
                    spec_loss = torch.sum(torch.abs(noise_fft - pred_fft))
                
                # Combined loss
                patient_loss = time_loss + spectral_weight * spec_loss
            else:
                # Regular MSE loss
                patient_loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='sum')
            
            batch_loss = patient_loss.item()
            val_loss += batch_loss
            
            # Print progress for every 5 batches
            if (step + 1) % 5 == 0 or step == 0:
                print(f"  Batch {step+1}/{len(data_loader)} - Patient epochs: {num_epochs}")
                sys.stdout.flush()  # Force output to be written to file immediately
    
    # Calculate average loss
    avg_loss = val_loss / total_samples
    
    # Log to tensorboard
    writer.add_scalar("loss", avg_loss, epoch)
    
    # Print validation summary
    print(f"Validation complete - Avg Loss: {avg_loss:.6f}")
    sys.stdout.flush()  # Force output to be written to file immediately
    
    return avg_loss

# Function to generate samples for a single patient
def generate_patient_samples(model, scheduler, patient_data, device, num_inference_steps=1000):
    """
    Generate synthetic EEG data for a patient based on their epoch count.
    
    Args:
        model: The diffusion model
        scheduler: The diffusion scheduler
        patient_data: Tensor with shape [epochs, channels, timepoints]
        device: Device to use for generation
        num_inference_steps: Number of denoising steps
        
    Returns:
        Tensor with shape [epochs, channels, timepoints]
    """
    model.eval()
    
    print(f"Generating samples for patient with {patient_data.shape[0]} epochs...")
    
    # Get the number of epochs in the patient data
    num_epochs = patient_data.shape[0]
    
    # Create initial noise
    sample = torch.randn_like(patient_data).to(device)
    
    # Denoise the sample step by step
    scheduler.set_timesteps(num_inference_steps)
    
    start_time = time.time()
    
    for i, t in enumerate(scheduler.timesteps):
        # Expand timestep for all epochs
        timesteps = t.repeat(num_epochs).to(device)
        
        # Get model prediction
        with torch.no_grad():
            model_output = model(sample, timesteps)
        
        # Update sample with scheduler
        sample = scheduler.step(model_output, t, sample).prev_sample
        
        # Print progress
        if (i + 1) % 100 == 0 or i == 0 or i == len(scheduler.timesteps) - 1:
            progress = (i + 1) / len(scheduler.timesteps) * 100
            elapsed = time.time() - start_time
            print(f"  Step {i+1}/{len(scheduler.timesteps)} ({progress:.1f}%) - Time elapsed: {elapsed:.2f}s")
            sys.stdout.flush()
    
    total_time = time.time() - start_time
    print(f"Sample generation complete! Time taken: {total_time:.2f} seconds")
    
    return sample