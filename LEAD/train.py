"""
---
title: A minimum implementation for DS_DDPM for start over with code comments
---
"""
import sys
sys.path.append("/home/stud/timlin/bhome/DiffusionEEG/LEAD/DS-DDPM")
sys.path.append("/home/stud/timlin/bhome/DiffusionEEG/LEAD/LEAD")
import comet_ml

from multiprocessing import reduction
from turtle import Turtle
from typing import List

import torch
import torch.utils.data
import torchvision
from PIL import Image

from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from src.EEGNet import EEG_Net_8_Stack
from src.unet_eeg_subject_emb import sub_gaussion
from src.unet_eeg import UNet
from typing import Tuple, Optional


import torch.nn.functional as F
import torch.utils.data
from torch import nn
from src.utils import gather
import numpy as np
import scipy.io as sio
import csv
import math

# LEAD
import os
from torch.utils.data import Dataset
import csv
from models.LEAD import Model as LEADModel

# subject_instance = UNet_sub()

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# (Assume gather() is defined elsewhere to index tensors by time-step t.)

class ArcMarginProduct(nn.Module):
    r"""Implementation of large margin arc distance.
    
    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: scaling factor.
        m: margin.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = torch.nn.parameter.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class ArcMarginHead(nn.Module):
    r"""ArcMarginHead that wraps an auxiliary backbone.
    
    It uses ArcMarginProduct to compute the final logits.
    """
    def __init__(self, in_features, out_features, load_backbone='./LEAD/DS-DDPM/assets/max_acc.pth', s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginHead, self).__init__()
        self.arcpro = ArcMarginProduct(in_features, out_features, s=s, m=m, easy_margin=easy_margin)
        self.auxback = EEG_Net_8_Stack(mtl=False)  # Assumes EEG_Net_8_Stack is defined elsewhere.
        pretrained_checkpoint = torch.load(load_backbone)
        print("Loading pretrained subject EEGNet weights: ", pretrained_checkpoint.keys())
        self.auxback.load_state_dict(pretrained_checkpoint['params'])
        print("Backbone loading successful.")

    def forward(self, input, label):
        emb = self.auxback(input)
        output = self.arcpro(emb, label)
        return output


class DenoiseDiffusion:
    """
    Denoise Diffusion class using the LEAD model's supervised branch.
    
    Instead of calling the diffusion-specific forward pass of the LEAD model,
    we now call its supervised branch by providing a dummy marker tensor.
    """
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device,
                 sub_theta: nn.Module, sub_arc_head: nn.Module,
                 debug=False, time_diff_constraint=True):
        super().__init__()
        self.eps_model = eps_model
        self.sub_theta = sub_theta
        self.sub_arc_head = sub_arc_head
        self.time_diff_constraint = time_diff_constraint

        # Create beta schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta
        self.debug = True
        self.step_size = 75
        self.window_size = 224
        self.subject_noise_range = 9

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        # Instead of using a diffusion branch, call the supervised branch.
        dummy = torch.zeros(xt.size(0), self.eps_model.seq_len, 1, device=xt.device)
        eps_theta = self.eps_model.pretrain(xt, dummy)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** 0.5) * eps

    def p_sample_noise(self, xt: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        eps_theta = self.sub_theta(xt, t, s)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** 0.5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, debug=False):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        dummy = torch.zeros(xt.size(0), self.eps_model.seq_len, 1, device=xt.device)
        eps_theta = self.eps_model.pretrain(xt, dummy)
        return F.mse_loss(noise, eps_theta)

    def loss_with_diff_constraint(self, x0: torch.Tensor, label: torch.Tensor,
                                  noise: Optional[torch.Tensor] = None, debug=False,
                                  noise_content_kl_co=1, arc_subject_co=0.1, orgth_co=2):
         # Get batch size
        debug = self.debug
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        if debug:
            print("the shape of x0")
            print(x0.shape)
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        # s = torch.randint(0, self.subject_noise_range, (batch_size,), device=x0.device, dtype=torch.long)
        s = label
        if debug:
            print("the shape of t")
            print(t.shape)
            print(t)
        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)
            if debug:
                print("the shape of noise")
                print(noise.shape)
        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        if debug:
            print("the shape of xt")
            print(xt.shape)
        eps_theta = self.eps_model.pretrain(xt, t)
        subject_mu, subject_theta = self.sub_theta(xt, t, s)
        constraint_panelty = 0
        for i in range(eps_theta.shape[3] - 1):
            constraint_panelty += F.mse_loss(eps_theta[:, :, self.step_size:, i],
                                             eps_theta[:, :, :-self.step_size, i + 1],
                                             reduction='mean')
        noise_conent_kl = F.kl_div(eps_theta.softmax(dim=-1).log(),
                                   subject_theta.softmax(dim=-1),
                                   reduction='mean')
        organal_squad = torch.bmm(
            eps_theta.view(eps_theta.shape[0]*eps_theta.shape[1], eps_theta.shape[2], eps_theta.shape[3]),
            subject_theta.view(subject_theta.shape[0]*subject_theta.shape[1], subject_theta.shape[2], subject_theta.shape[3]).permute(0,2,1)
        )
        ones = torch.ones(eps_theta.shape[0]*eps_theta.shape[1], eps_theta.shape[2], eps_theta.shape[2],
                           dtype=torch.float32, device=xt.device)
        diag = torch.eye(eps_theta.shape[2], dtype=torch.float32, device=xt.device)
        loss_orth = ((organal_squad * (ones - diag)) ** 2).mean()
        subject_arc_logit = self.sub_arc_head(subject_theta.permute(0,3,2,1), s)
        subject_arc_loss = F.cross_entropy(subject_arc_logit, s.long())
        if self.time_diff_constraint:
            total_loss = (F.mse_loss(noise, eps_theta + subject_theta) +
                          orgth_co * loss_orth +
                          arc_subject_co * subject_arc_loss +
                          0.1 * constraint_panelty)
        else:
            total_loss = (F.mse_loss(noise, eps_theta + subject_theta) +
                          orgth_co * loss_orth +
                          arc_subject_co * subject_arc_loss)
        return total_loss, constraint_panelty, noise_content_kl_co * noise_conent_kl, arc_subject_co * subject_arc_loss, orgth_co * loss_orth

class Configs(BaseConfigs):
    """
    ## Configurations for Diffusion using the LEAD Model
    """
    # Device configuration (automatically selects CUDA if available)
    device: torch.device = DeviceConfigs()

    # Replace the UNet model with the LEAD model for εₜ(xₜ, t)
    eps_model: LEADModel  
    sub_theta: sub_gaussion
    sub_archead: ArcMarginHead
    diffusion: DenoiseDiffusion

    # Data/model dimensions
    eeg_channels: int = 22
    window_size: int = 224
    stack_size: int = 8
    n_channels: int = 64
    channel_multipliers: List[int] = [1, 2, 2, 4]
    is_attention: List[int] = [False, False, False, True]

    # Diffusion parameters
    n_steps: int = 1_000
    batch_size: int = 32
    n_samples: int = 16
    learning_rate: float = 2e-5
    arc_in = 4 * 2 * 14
    arc_out = 9

    # Training parameters
    epochs: int = 1_000

    # Dataset and dataloader
    dataset: torch.utils.data.Dataset
    data_loader: torch.utils.data.DataLoader

    # Optimizers
    optimizer: torch.optim.Adam
    optimizer_noise: torch.optim.Adam

    def init(self):
        # Set up a configuration for the LEAD model.
        lead_config = type("LeadConfig", (), {})()  # Create an empty config object
        lead_config.task_name = "diffusion"          # Use the diffusion branch in LEAD.
        lead_config.seq_len = self.window_size        # Use window_size as sequence length.
        lead_config.enc_in = self.eeg_channels          # Number of input channels.
        lead_config.d_model = self.n_channels           # Model dimension.
        lead_config.n_heads = 4                         # Number of attention heads (adjust as needed).
        lead_config.dropout = 0.1                       # Dropout probability.
        lead_config.patch_len_list = "4,4"              # Example patch lengths; tune as necessary.
        lead_config.up_dim_list = "64,128"              # Example up-dim values; tune as necessary.
        lead_config.e_layers = 4                        # Number of encoder layers.
        lead_config.d_ff = 256                          # Feed-forward network dimension.
        lead_config.output_attention = False            # Whether to output attention maps.
        lead_config.no_temporal_block = False
        lead_config.no_channel_block = False
        lead_config.augmentations = "none"              # No augmentations.
        lead_config.activation = "gelu"                 # Activation function.
        lead_config.no_inter_attn = False

        # Instantiate the LEAD model and move it to the configured device.
        self.eps_model = LEADModel(lead_config).to(self.device)

        # Instantiate the other components as before.
        self.sub_theta = sub_gaussion(
            eeg_channels=self.eeg_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        self.sub_archead = ArcMarginHead(
            self.arc_in, self.arc_out,
            load_backbone='./LEAD/DS-DDPM/assets/max_acc.pth',
            s=30.0, m=0.50, easy_margin=False
        ).to(self.device)

        # Create the diffusion process using the new LEAD model.
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
            sub_theta=self.sub_theta,
            sub_arc_head=self.sub_archead,
            debug=False,
        )

        # Create the dataloader.
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, self.batch_size, shuffle=True, pin_memory=True
        )
        # Create the optimizers.
        self.optimizer = torch.optim.Adam([
            {'params': self.eps_model.parameters(), 'lr': self.learning_rate},
            {'params': self.sub_theta.parameters(), 'lr': self.learning_rate},
            {'params': self.sub_archead.arcpro.parameters(), 'lr': self.learning_rate}
        ], lr=self.learning_rate)
        self.optimizer_noise = torch.optim.Adam(self.sub_theta.parameters(), lr=self.learning_rate)



    def sample(self):
        """
        ### Sample images
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn([self.n_samples, self.eeg_channels, self.window_size, self.stack_size],
                            device=self.device)

            # Remove noise for $T$ steps
            for t_ in monit.iterate('Sample', self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # Log samples
            # tracker.save('sample', x)

   

    def train(self):
        """
        ### Train
        """
        metric_file = '/home/stud/timlin/bhome/DiffusionEEG/LEAD/metric/test.csv'
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(metric_file), exist_ok=True)

        # Open the file in append mode, creating it if necessary
        with open(metric_file, 'a+') as loss_target:
            target_writer = csv.writer(loss_target)
            
            # If the file is empty, write the header
            if os.stat(metric_file).st_size == 0:
                target_writer.writerow(['loss', 'time_period_diff'])



        # Iterate through the dataset
        for data, label in monit.iterate('Train', self.data_loader):
            print("Data shape:", data.shape)
            print("Label shape:", label.shape)
            
            # Optional: Inspect the first few elements of the data and labels to ensure they look correct
            print("Data sample:", data[0])  # First sample in the batch
            print("Label sample:", label[0])  # Corresponding label for the first sample

            # Increment global step
            tracker.add_global_step()
            # Move data to device
            data = torch.permute(data, (0,3,2,1))
            data = data.to(self.device)
            label = label.float().to(self.device)
            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            # loss = self.diffusion.loss(data)
            loss, time_period_diff, noise_conent_kl, sub_arc_loss, loss_orth = self.diffusion.loss_with_diff_constraint(data, label)
            # return F.mse_loss(noise, eps_theta + subject_theta) + orgth_co * loss_orth + arc_subject_co *subject_arc_loss, constraint_panelty, noise_content_kl_co * noise_conent_kl, arc_subject_co *subject_arc_loss, orgth_co * loss_orth

            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            self.optimizer_noise.step()
            # Track the loss
            # tracker.save('loss', loss - time_period_diff)
            tracker.save('loss', loss)
            tracker.save('diffusion_loss', loss - loss_orth -sub_arc_loss)
            tracker.save('time_seg_diff', time_period_diff)
            tracker.save('noise_conent_kl', noise_conent_kl)
            tracker.save('sub_arc_loss', sub_arc_loss)
            tracker.save('loss_orth', loss_orth)
            target_writer.writerow([loss.detach().cpu().numpy(), (loss - loss_orth -sub_arc_loss).detach().cpu().numpy(), time_period_diff.detach().cpu().numpy(), noise_conent_kl.detach().cpu().numpy(), sub_arc_loss.detach().cpu().numpy(), loss_orth.detach().cpu().numpy()])
        loss_target.close()


    def run(self):
        """
        ### Training loop
        """
        for inicator in monit.loop(self.epochs):
            # Train the model
            print(inicator)
            self.train()
            # Sample some images
            self.sample()
            # New line in the console
            tracker.new_line()
            # Save the model
            if inicator % 30 == 0:
                experiment.save_checkpoint()


class DatasetLoader_BCI_IV_signle(torch.utils.data.Dataset):

    def __init__(self, setname, datafolder=None, train_aug=False, subject_id=3):

        subject_id = subject_id
        if datafolder is None:
            data_folder = '../data'
        else:
            data_folder = datafolder
        data = sio.loadmat(data_folder + "/single_sep/single_subject_data_" + str(subject_id) + ".mat")
        test_X = data["test_x"][:, :, 750:1500]  # [trials, channels, time length]
        train_X = data["train_x"][:, :, 750:1500]

        test_y = data["test_y"].ravel()
        train_y = data["train_y"].ravel()

        train_y -= 1
        test_y -= 1
        window_size = 224
        step = 75 # 这里必须保证产出的tensor 是偶数，这里是超大overlap的形式
        # window_size = 400
        # step = 50 # 这里必须保证产出的tensor 是偶数，这里是超大overlap的形式
        n_channel = 22

        def windows(data, size, step):
            start = 0
            while (start + size) < data.shape[0]:
                yield int(start), int(start + size)
                start += step

        def segment_signal_without_transition(data, window_size, step):
            segments = []
            for (start, end) in windows(data, window_size, step):
                if len(data[start:end]) == window_size:
                    segments = segments + [data[start:end]]
            return np.array(segments)

        def segment_dataset(X, window_size, step):
            win_x = []
            for i in range(X.shape[0]):
                win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
            win_x = np.array(win_x)
            return win_x

        train_raw_x = np.transpose(train_X, [0, 2, 1])
        test_raw_x = np.transpose(test_X, [0, 2, 1])

        train_win_x = segment_dataset(train_raw_x, window_size, step)
        test_win_x = segment_dataset(test_raw_x, window_size, step)
        train_win_y = train_y
        test_win_y = test_y

        expand_factor = train_win_x.shape[1]

        train_x = np.reshape(train_win_x, (-1, train_win_x.shape[2], train_win_x.shape[3]))
        test_x = np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
        train_y = np.repeat(train_y, expand_factor)
        test_y = np.repeat(test_y, expand_factor)

        train_x = np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
        train_y = np.reshape(train_y, [train_y.shape[0]]).astype('float32')

        test_x = np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        test_y = np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        # test_x = test_x[2000:, :, :, :]
        # test_y = test_y[2000:]

        # val_x = test_x[:2000, :, :, :]
        # val_y = test_y[:2000]

        train_win_x = train_win_x.astype('float32')
        
        ratio = 0.5
        idx = list(range(len(test_win_y)))
        np.random.shuffle(idx)
        test_win_x = test_win_x[idx]
        test_win_y = test_win_y[idx]

        val_win_x = test_win_x[:int(len(test_win_x)*0.5), :, :, :].astype('float32')
        val_win_y = test_win_y[:int(len(test_win_x)*0.5)]

        real_test_win_x = test_win_x[int(len(test_win_x)*0.5):, :, :, :].astype('float32')
        real_test_win_y = test_win_y[int(len(test_win_x)*0.5):]

        self.X_val = val_win_x
        self.y_val = val_win_y

        print("The shape of sample x0 is {}".format(test_win_x.shape))

        if setname == 'train':
            self.data = train_win_x
            self.label = train_win_y
        elif setname == 'val':
            self.data = val_win_x
            self.label = val_win_y
        elif setname == 'test':
            self.data = real_test_win_x
            self.label = real_test_win_y

        self.num_class = 4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        return data, label


class DatasetLoader_BCI_IV_mix(torch.utils.data.Dataset):

    def __init__(self, setname, datafolder=None, train_aug=False):

        subject_id = 'all'
        if datafolder is None:
            data_folder = '../data'
        else:
            data_folder = datafolder
        data = sio.loadmat(data_folder + "/mix_sub/mix_subject_data_" + str(subject_id) + ".mat")
        test_X = data["test_x"][:, :, 750:1500]  # [trials, channels, time length]
        train_X = data["train_x"][:, :, 750:1500]

        test_y = data["test_y"].ravel()
        train_y = data["train_y"].ravel()

        train_y -= 1
        test_y -= 1
        # window_size = 400
        # step = 45
        window_size = 224
        step = 75
        n_channel = 22

        def windows(data, size, step):
            start = 0
            while (start + size) < data.shape[0]:
                yield int(start), int(start + size)
                start += step

        def segment_signal_without_transition(data, window_size, step):
            segments = []
            for (start, end) in windows(data, window_size, step):
                if len(data[start:end]) == window_size:
                    segments = segments + [data[start:end]]
            return np.array(segments)

        def segment_dataset(X, window_size, step):
            win_x = []
            for i in range(X.shape[0]):
                win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
            win_x = np.array(win_x)
            return win_x

        train_raw_x = np.transpose(train_X, [0, 2, 1])
        test_raw_x = np.transpose(test_X, [0, 2, 1])

        train_win_x = segment_dataset(train_raw_x, window_size, step)
        test_win_x = segment_dataset(test_raw_x, window_size, step)
        train_win_y = train_y
        test_win_y = test_y

        expand_factor = train_win_x.shape[1]

        train_x = np.reshape(train_win_x, (-1, train_win_x.shape[2], train_win_x.shape[3]))
        test_x = np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
        train_y = np.repeat(train_y, expand_factor)
        test_y = np.repeat(test_y, expand_factor)

        train_x = np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
        train_y = np.reshape(train_y, [train_y.shape[0]]).astype('float32')

        test_x = np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        test_y = np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        # test_x = test_x[2000:, :, :, :]
        # test_y = test_y[2000:]

        # val_x = test_x[:2000, :, :, :]
        # val_y = test_y[:2000]

        train_win_x = train_win_x.astype('float32')
        
        ratio = 0.5
        idx = list(range(len(test_win_y)))
        np.random.shuffle(idx)
        test_win_x = test_win_x[idx]
        test_win_y = test_win_y[idx]

        val_win_x = test_win_x[:int(len(test_win_x)*0.5), :, :, :].astype('float32')
        val_win_y = test_win_y[:int(len(test_win_x)*0.5)]

        real_test_win_x = test_win_x[int(len(test_win_x)*0.5):, :, :, :].astype('float32')
        real_test_win_y = test_win_y[int(len(test_win_x)*0.5):]

        self.X_val = val_win_x
        self.y_val = val_win_y

        print('The shape of x is {}'.format(test_win_x.shape))

        if setname == 'train':
            self.data = train_win_x
            self.label = train_win_y
        elif setname == 'val':
            self.data = val_win_x
            self.label = val_win_y
        elif setname == 'test':
            self.data = real_test_win_x
            self.label = real_test_win_y

        self.num_class = 4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        return data, label


class DatasetLoader_BCI_IV_mix_subjects(torch.utils.data.Dataset):

    def __init__(self, setname, datafolder, train_aug=False):
        test_X_list = []
        train_X_list = []
        test_y_list = []
        train_y_list = []

        for i in range(9):
            subject_id = i + 1
            if datafolder is None:
                data_folder = '../data'
            else:
                data_folder = datafolder

            data = sio.loadmat(data_folder + "/single_sep/single_subject_data_" + str(subject_id) + ".mat")
            test_X = data["test_x"][:, :, 750:1500]  # [trials, channels, time length]
            train_X = data["train_x"][:, :, 750:1500]

            test_y = np.ones((test_X.shape[0],)) * i
            train_y = np.ones((train_X.shape[0],)) * i
            # print(test_y.shape)
            # print(train_y.shape)
            test_X_list.append(test_X)
            train_X_list.append(train_X)
            test_y_list.append(test_y)
            train_y_list.append(train_y)

        test_X = np.vstack(test_X_list)
        train_y = np.concatenate(train_y_list, axis=0)
        train_X = np.vstack(train_X_list)
        test_y = np.concatenate(test_y_list, axis=0)
        print(test_X.shape)
        print(test_y.shape)
        print(train_X.shape)
        print(train_y.shape)
        # train_y -= 1
        # test_y -= 1
        window_size = 224
        step = 75
        n_channel = 22
        # n_channel = 22

        def windows(data, size, step):
            start = 0
            while (start + size) < data.shape[0]:
                yield int(start), int(start + size)
                start += step

        def segment_signal_without_transition(data, window_size, step):
            segments = []
            for (start, end) in windows(data, window_size, step):
                if len(data[start:end]) == window_size:
                    segments = segments + [data[start:end]]
            return np.array(segments)

        def segment_dataset(X, window_size, step):
            win_x = []
            for i in range(X.shape[0]):
                win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
            win_x = np.array(win_x)
            return win_x

        train_raw_x = np.transpose(train_X, [0, 2, 1])
        test_raw_x = np.transpose(test_X, [0, 2, 1])

        train_win_x = segment_dataset(train_raw_x, window_size, step)
        test_win_x = segment_dataset(test_raw_x, window_size, step)
        train_win_y = train_y
        test_win_y = test_y

        expand_factor = train_win_x.shape[1]

        train_x = np.reshape(train_win_x, (-1, train_win_x.shape[2], train_win_x.shape[3]))
        test_x = np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
        train_y = np.repeat(train_y, expand_factor)
        test_y = np.repeat(test_y, expand_factor)

        train_x = np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
        train_y = np.reshape(train_y, [train_y.shape[0]]).astype('float32')

        test_x = np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        test_y = np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        # test_x = test_x[2000:, :, :, :]
        # test_y = test_y[2000:]

        # val_x = test_x[:2000, :, :, :]
        # val_y = test_y[:2000]

        train_win_x = train_win_x.astype('float32')
        
        ratio = 0.5
        idx = list(range(len(test_win_y)))
        np.random.shuffle(idx)
        test_win_x = test_win_x[idx]
        test_win_y = test_win_y[idx]

        idx = list(range(len(test_win_y)))
        np.random.shuffle(idx)
        train_x = train_x[idx]
        train_y = train_y[idx]

        val_win_x = test_win_x[:int(len(test_win_x)*0.5), :, :, :].astype('float32')
        val_win_y = test_win_y[:int(len(test_win_x)*0.5)]

        real_test_win_x = test_win_x[int(len(test_win_x)*0.5):, :, :, :].astype('float32')
        real_test_win_y = test_win_y[int(len(test_win_x)*0.5):]

        self.X_val = val_win_x
        self.y_val = val_win_y

        if setname == 'train':
            self.data = train_win_x
            self.label = train_win_y
        elif setname == 'val':
            self.data = val_win_x
            self.label = val_win_y
        elif setname == 'test':
            self.data = real_test_win_x
            self.label = real_test_win_y

        self.num_class = 9

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ## comment out F.normalize if you want unconstrained diffusion (without normalize to 1)
        # data, label = F.normalize(torch.Tensor(self.data[i]), p=2, dim=2), self.label[i]
        data, label = torch.Tensor(self.data[i]), self.label[i]
        return data, label


class DatasetLoader_BCI_EEG_npy(Dataset):
    """
    Dataset class for EEG data generated by the pre-processing script.
    It loads features from .npy files (one per subject) and labels from label.npy.
    
    Expected feature file format:
        feature_{subject_id:02d}.npy
    Expected label file format:
        label.npy, containing a 2D array with columns [label, subject_id].
    """
    def __init__(self, setname, feature_folder, label_folder, subject_id=3):
        """
        Args:
            setname (str): One of 'train', 'val', or 'test'. In this example we use all data as 'train'.
            feature_folder (str): Path to the folder containing feature files.
            label_folder (str): Path to the folder containing label.npy.
            subject_id (int): The subject id to load.
        """
        self.setname = setname
        self.subject_id = subject_id

        # Load labels (assumes label.npy is stored in label_folder)
        label_path = os.path.join(label_folder, 'label.npy')
        labels = np.load(label_path)  # shape: (num_subjects, 2) with columns [label, subject_id]
        
        # Load the feature file for this subject.
        feature_file = os.path.join(feature_folder, f'feature_{subject_id:02d}.npy')
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature file {feature_file} not found.")
        data = np.load(feature_file)  # shape: (num_segments, SAMPLE_LEN, num_channels)
        data = data.astype('float32')
        
        # Reshape to add a channel dimension: (num_segments, 1, SAMPLE_LEN, num_channels)
        self.data = data.reshape((data.shape[0], 1, data.shape[1], data.shape[2]))
        
        # Get the label for this subject.
        subj_label = labels[labels[:, 1] == subject_id, 0]
        if subj_label.size > 0:
            lab = subj_label[0]
        else:
            lab = 0  # default label if not found
        # Create a label for each segment.
        self.label = np.full((self.data.shape[0],), lab, dtype='float32')
        
        # Number of classes (assuming labels are coded from 0 to num_class-1)
        self.num_class = len(np.unique(self.label))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        # Return a tuple of (data, label) as torch tensors.
        x = torch.tensor(self.data[i])
        y = torch.tensor(self.label[i]).long()
        return x, y


# @option(Configs.dataset, 'bci_comp_iv')
# def datasetLoader_BCI_IV_signle(c: Configs):
#     """
#     Create BCI IV dataset
#     """
    
#     return DatasetLoader_BCI_IV_signle('train', datafolder='/home/yiqduan/Data/bci/EEG_MI_DARTS/Mudus_BCI/data/bci_iv/', subject_id=3)


# @option(Configs.dataset, 'bci_comp_iv_full_mix')
# def datasetLoader_BCI_IV_mix_subjects(c: Configs):
#     """
#     Create BCI IV dataset
#     """
#     return DatasetLoader_BCI_IV_mix_subjects('train', datafolder='/home/yiqduan/Data/bci/EEG_MI_DARTS/Mudus_BCI/data/bci_iv/')

@option(Configs.dataset, 'bci_eeg_npy')
def datasetLoader_BCI_EEG_npy(c: Configs):
    """
    Create the BCI EEG dataset from preprocessed .npy files.
    """
    feature_folder = '/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG/Feature'
    label_folder = '/home/stud/timlin/bhome/DiffusionEEG/dataset/CAUEEG/Label'
    # Here subject_id can be passed as needed (e.g., subject_id=3)
    return DatasetLoader_BCI_EEG_npy('train', feature_folder, label_folder, subject_id=3)



def main():
    # Create experiment with screen writer
    experiment.create(name='unet_subjects_double_step_classifier_free_baseline', writers={'screen'})

    # Create configurations
    configs = Configs()

    # Override configurations if needed
    experiment.configs(configs, {
        'dataset': 'bci_eeg_npy',
        'eeg_channels': 19,
        'epochs': 128,
    })

    # Initialize configurations (instantiates models, dataset, etc.)
    configs.init()

    # The following line is removed because add_pytorch_models is no longer available.
    # experiment.add_pytorch_models({'eps_model': configs.eps_model, 'subject_theta': configs.sub_theta})

    # Start and run the training loop
    with experiment.start():
        configs.run()




#
if __name__ == '__main__':
    main()
 