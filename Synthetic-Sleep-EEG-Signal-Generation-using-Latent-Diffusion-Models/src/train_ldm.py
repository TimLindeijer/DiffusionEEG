"""
Author: Bruno Aristimunha
Training LDM with SleepEDFx, SHHS, or CAUEEG2 data.
Based on the tutorial from Monai Generative.
"""
import argparse

import torch
import torch.nn as nn

from generative.networks.nets import DiffusionModelUNet
from generative.networks.nets import AutoencoderKL
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import first, set_determinism
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
import os

from dataset.dataset import train_dataloader, valid_dataloader, get_trans
from models.ldm import UNetModel
from training import train_ldm
from util import log_mlflow, ParseListAction, setup_run_dir

# for reproducibility purposes set a seed
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
        default="/project/config/config_ldm.yaml",
        help="Path to config file with all the training parameters needed",
    )
    parser.add_argument(
        "--path_train_ids",
        type=str,
        default="/project/data/ids/ids_sleep_edfx_cassette_double_train.csv",
    )

    parser.add_argument(
        "--path_valid_ids",
        type=str,
        default="/project/data/ids/ids_sleep_edfx_cassette_double_valid.csv",
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
        type=str, action=ParseListAction,
    )
    parser.add_argument(
        "--autoencoderkl_config_file_path",
        help="Path to the .pth model from the stage1.",
        default="/project/config/config_aekl_eeg.yaml"
    )

    parser.add_argument(
        "--best_model_path",
        help="Path to the .pth model from the stage1.",
    )
    parser.add_argument(
        "--spe",
        type=str,
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
    )
    # Add dataset parameter
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["edfx", "shhs", "shhsh", "caueeg2"],
        help="Dataset to use for training",
    )
    
    # Add label filter parameter for CAUEEG2
    parser.add_argument(
        "--label_filter",
        type=str,
        help="Label filter for CAUEEG2 dataset. Can be 'hc' (healthy), 'mci', 'dementia', or a comma-separated list"
    )

    args = parser.parse_args()
    
    # Process label_filter if it's a comma-separated string
    if hasattr(args, 'label_filter') and args.label_filter and ',' in args.label_filter:
        args.label_filter = args.label_filter.split(',')
    
    return args


class Stage1Wrapper(nn.Module):
    """Wrapper for stage 1 model."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Ensure model is in float32
        self.model = self.model.float()

    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        z_mu, z_sigma = self.model.encode(x)
        z = self.model.sampling(z_mu, z_sigma)
        return z
    
def main(args):
    config = OmegaConf.load(args.config_file)

    set_determinism(seed=config.train.seed)
    print_config()

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

    # Get transforms for the specified dataset
    trans = get_trans(args.dataset)
    
    # Getting data loaders with dataset parameter
    train_loader = train_dataloader(config=config, args=args, transforms_list=trans, dataset=args.dataset)
    val_loader = valid_dataloader(config=config, args=args, transforms_list=trans, dataset=args.dataset)
    
    # Defining device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Defining model
    config_aekl = OmegaConf.load(args.autoencoderkl_config_file_path)
    autoencoder_args = config_aekl.autoencoderkl.params
    if args.num_channels is not None:
        autoencoder_args['num_channels'] = args.num_channels
    if args.latent_channels is not None:
        autoencoder_args['latent_channels'] = args.latent_channels

    stage1 = AutoencoderKL(**autoencoder_args)

    state_dict = torch.load(args.best_model_path+"/best_model.pth",
                        map_location=torch.device('cpu'))

    # Convert all tensors in state_dict to float32
    for key in state_dict:
        if isinstance(state_dict[key], torch.Tensor):
            state_dict[key] = state_dict[key].float()

    stage1.load_state_dict(state_dict)
    stage1 = stage1.float()  # Ensure the model is in float32
    stage1.to(device)
    
    with torch.no_grad():
        with autocast(enabled=True):
            check_data = first(train_loader)['eeg'].to(device)
            
            # Handle CAUEEG2 data dimensions
            if args.dataset == "caueeg2":
                # For CAUEEG2, reshape from [batch, 1, 19, 1000] to [batch, 19, 1000]
                check_data = check_data.squeeze(1) if check_data.dim() == 4 else check_data
                # Apply padding slicing if needed
                if check_data.dim() == 3 and check_data.shape[2] > 72:  # 72 = 36*2
                    check_data = check_data[:, :, 36:-36]
                    
            z = stage1.encode_stage_2_inputs(check_data)

    autoencoderkl = Stage1Wrapper(model=stage1)

    #########################################################################
    # Diffusion model part
    parameters = config['model']['params']['unet_config']['params']
    parameters['in_channels'] = args.latent_channels
    parameters['out_channels'] = args.latent_channels
    
    # Update image_size parameter for CAUEEG2 if needed
    if args.dataset == "caueeg2":
        # Check the shape of z to determine the correct image_size
        if hasattr(z, 'shape') and len(z.shape) >= 3:
            # z should have shape [batch, latent_channels, time_dim]
            parameters['image_size'] = z.shape[2]
            print(f"Setting diffusion model image_size to match latent dimension: {parameters['image_size']}")

    diffusion = UNetModel(**parameters)

    if torch.cuda.device_count() > 1:
        autoencoderkl = torch.nn.DataParallel(autoencoderkl)
        diffusion = torch.nn.DataParallel(diffusion)

    autoencoderkl.eval()
    
    autoencoderkl = autoencoderkl.to(device)
    diffusion.to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta",
                            beta_start=0.0015, beta_end=0.0195)
    
    scheduler.to(device)
    print(f"Scaling factor set to {1 / torch.std(z)}")
    scale_factor = 1 / torch.std(z)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

    best_loss = float("inf")
    start_epoch = 0
    
    # Handle resume from checkpoint
    if resume:
        print(f"Loading checkpoint from {run_dir}")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        diffusion.load_state_dict(checkpoint["diffusion"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        scale_factor = checkpoint.get("scale_factor", scale_factor)
        print(f"Resumed from epoch {start_epoch} with best loss {best_loss}")

    print(f"Starting Training")
    val_loss = train_ldm(
        model=diffusion,
        stage1=autoencoderkl,
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
        scale_factor=scale_factor,
    )

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