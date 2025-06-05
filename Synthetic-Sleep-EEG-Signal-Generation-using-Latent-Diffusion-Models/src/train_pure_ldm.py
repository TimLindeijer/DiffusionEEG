"""
Author: Bruno Aristimunha
Training LDM with SleepEDFx, SHHS, or CAUEEG2 data.
Based on the tutorial from Monai Generative.

"""
import argparse
import os
import torch
import torch.nn as nn

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import first, set_determinism
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from generative.inferers import DiffusionInferer


from dataset.dataset import train_dataloader, valid_dataloader, get_trans
from models.ldm import UNetModel
from training.training_diffusion import train_diffusion_
from util import log_mlflow, ParseListAction, setup_run_dir
from generative.networks.nets import DiffusionModelUNet
# print_config()
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
        #default="/home/bru/PycharmProjects/DDPM-EEG/data/data_test",
        default="/data/physionet-sleep-data-npy",
        help="Path to preprocessed data. For CAUEEG2, this should be the base directory containing Feature/ and Label/ folders"
    )
        
    parser.add_argument(
        "--spe",
        type=str,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["edfx", "shhs", "shhsh", "caueeg2"],
        help="Dataset to use for training"
    )

    # Add label filter argument for CAUEEG2
    parser.add_argument(
        "--label_filter",
        type=str,
        nargs='*',
        default=None,
        help="Filter CAUEEG2 dataset by labels. Options: 'hc'/'healthy' (0), 'mci' (1), 'dementia' (2), or numeric values. Can specify multiple: --label_filter hc mci"
    )

    args = parser.parse_args()
    
    # Process label_filter for CAUEEG2
    if args.dataset == "caueeg2" and args.label_filter is not None:
        processed_filters = []
        for filt in args.label_filter:
            if filt.lower() in ['hc', 'healthy', 'healthy_controls']:
                processed_filters.append(0)
            elif filt.lower() == 'mci':
                processed_filters.append(1)
            elif filt.lower() == 'dementia':
                processed_filters.append(2)
            else:
                try:
                    processed_filters.append(int(filt))
                except ValueError:
                    print(f"Warning: Unrecognized label filter '{filt}'. Skipping.")
        args.label_filter = processed_filters if processed_filters else None
        
        if args.label_filter:
            label_names = {0: 'Healthy', 1: 'MCI', 2: 'Dementia'}
            selected_labels = [label_names.get(l, f'Unknown-{l}') for l in args.label_filter]
            print(f"Using label filter: {selected_labels}")
    
    return args


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

    run_dir, resume = setup_run_dir(config=config, args=args,
                                    base_path=base_path)

    # Getting write training and validation data

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))
    trans = get_trans(args.dataset)

    # Getting data loaders
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "caueeg2":
        if not os.path.exists(os.path.join(args.path_pre_processed, 'Feature')):
            raise ValueError(f"CAUEEG2 Feature directory not found at {os.path.join(args.path_pre_processed, 'Feature')}. "
                           "Please check --path_pre_processed points to the correct CAUEEG2 base directory.")
        print(f"CAUEEG2 data path: {args.path_pre_processed}")
        if args.label_filter:
            print(f"Using label filter: {args.label_filter}")
    
    train_loader = train_dataloader(config=config, args=args, transforms_list=trans, dataset=args.dataset)
    val_loader = valid_dataloader(config=config, args=args, transforms_list=trans, dataset=args.dataset)

    # Test the data loader to make sure it works
    print("Testing data loader...")
    first_batch = first(train_loader)
    print(f"First batch EEG shape: {first_batch['eeg'].shape}")
    if 'subject' in first_batch:
        print(f"First batch subjects shape: {first_batch['subject'].shape}")
    if 'label' in first_batch:
        print(f"First batch labels shape: {first_batch['label'].shape}")

    # Defining device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    parameters = config['model']['params']['unet_config']['params']
    parameters['in_channels'] = 19
    parameters['out_channels'] = 19

    diffusion = UNetModel(**parameters)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        diffusion = torch.nn.DataParallel(diffusion)

    diffusion.to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta",
                              beta_start=0.0015, beta_end=0.0195)
    
    scheduler.to(device)
    if args.spe == 'spectral':
        spectral_loss = True
        print("Using spectral loss")
    else:
        spectral_loss = False

    inferer = DiffusionInferer(scheduler)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

    best_loss = float("inf")
    start_epoch = 0

    print(f"Starting Training with {args.dataset} dataset")
    val_loss = train_diffusion_(
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