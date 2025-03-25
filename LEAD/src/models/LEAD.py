from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.ADformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ADformerLayer
from layers.Embed import TokenChannelEmbedding
import math
import numpy as np
from typing import Optional, Tuple

class DenoiseDiffusion:
    """
    Simplified Denoising Diffusion Probabilistic Model.
    
    This implements the core DDPM algorithm as described in 
    "Denoising Diffusion Probabilistic Models" by Ho et al. (2020)
    with all additional constraints removed.
    """
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device, 
                 sub_theta: nn.Module = None, sub_arc_head: nn.Module = None,
                 time_diff_constraint=False, debug=False):
        """
        Initialize the diffusion process.
        
        Args:
            eps_model: The model that predicts noise
            n_steps: Number of diffusion steps
            device: The device to place constants on
            sub_theta: Optional subject-specific noise prediction network (ignored in simplified version)
            sub_arc_head: Optional ArcMargin classification head for subjects (ignored in simplified version)
            time_diff_constraint: Whether to use time difference constraint (ignored in simplified version)
            debug: Whether to print debug information
        """
        self.eps_model = eps_model
        self.device = device
        self.debug = debug

        # Create beta schedule (linearly increasing variance)
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        
        # Alpha values (1 - beta)
        self.alpha = 1. - self.beta
        
        # Cumulative product of alpha values
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Number of timesteps
        self.n_steps = n_steps
        
        # Variance schedule
        self.sigma2 = self.beta
        
        # Keep params for compatibility with the original implementation
        self.sub_theta = sub_theta
        self.sub_arc_head = sub_arc_head
        self.time_diff_constraint = time_diff_constraint
        
        if debug:
            print(f"Initialized simplified diffusion model with {n_steps} steps")
            print(f"Beta range: {self.beta[0].item():.6f} to {self.beta[-1].item():.6f}")

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get q(x_t|x_0) distribution: the distribution of adding noise to x_0 to get x_t.
        
        Args:
            x0: Original clean data [batch, seq_len, channels] or [batch, channels, seq_len]
            t: Timesteps [batch]
            
        Returns:
            mean: Mean of the conditional distribution q(x_t|x_0)
            var: Variance of the conditional distribution q(x_t|x_0)
        """
        # Get alpha_bar_t for the given timesteps
        alpha_bar_t = self.alpha_bar[t]
        
        # Add dimensions for broadcasting
        # For 3D input [batch, seq_len, channels], we need [batch, 1, 1]
        for _ in range(len(x0.shape) - 1):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        
        # Mean: √(α̅_t) * x_0
        mean = torch.sqrt(alpha_bar_t) * x0
        
        # Variance: (1 - α̅_t)
        var = 1 - alpha_bar_t
        
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        Sample from q(x_t|x_0): add noise to x_0 according to the diffusion schedule.
        
        Args:
            x0: Original clean data [batch, seq_len, channels] or [batch, channels, seq_len]
            t: Timesteps [batch]
            eps: Optional pre-generated noise (if None, will be sampled)
            
        Returns:
            x_t: Noisy samples at timestep t
        """
        # Generate random noise if not provided
        if eps is None:
            eps = torch.randn_like(x0)
        
        # Get mean and variance of q(x_t|x_0)
        mean, var = self.q_xt_x0(x0, t)
        
        # Sample from q(x_t|x_0) = N(mean, var)
        # x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε
        return mean + torch.sqrt(var) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        Sample from p_θ(x_{t-1}|x_t): perform one denoising step.
        
        Args:
            xt: Noisy input at time step t [batch, seq_len, channels] or [batch, channels, seq_len]
            t: Timestep indices [batch]
            
        Returns:
            x_{t-1}: Sample with less noise (one step of denoising)
        """
        # Store original shape for consistent output
        original_shape = xt.shape
        batch_size = xt.shape[0]
        
        # Predict noise using the model
        predicted_noise = self.eps_model(xt, t)
        
        # Make sure predicted_noise has the same shape as xt
        if predicted_noise.shape != xt.shape:
            try:
                # Try direct reshaping
                predicted_noise = predicted_noise.view(xt.shape)
            except:
                try:
                    # Try transposing if dimensions are swapped
                    if predicted_noise.dim() == 3 and xt.dim() == 3:
                        if predicted_noise.shape[1] == xt.shape[2] and predicted_noise.shape[2] == xt.shape[1]:
                            predicted_noise = predicted_noise.transpose(1, 2)
                    # If still doesn't match, try to reshape anyway
                    if predicted_noise.shape != xt.shape:
                        predicted_noise = predicted_noise.reshape(xt.shape[0], -1).reshape(xt.shape)
                except:
                    # If reshape fails, log a warning
                    print(f"Warning: Shape mismatch in p_sample. Predicted: {predicted_noise.shape}, Expected: {xt.shape}")
        
        # Get parameters from the noise schedule
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = self.beta[t]
        
        # Add dimensions for broadcasting
        for _ in range(len(xt.shape) - 1):
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            beta_t = beta_t.unsqueeze(-1)
        
        # Calculate the mean for p_θ(x_{t-1}|x_t)
        # μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-α̅_t)) * ε_θ(x_t, t))
        eps_coef = beta_t / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (xt - eps_coef * predicted_noise)
        
        # Variance
        # In DDPM paper, the variance is either fixed to β_t or learned
        # Here we use the fixed variance schedule
        var = beta_t
        
        # Sample noise
        noise = torch.randn_like(xt)
        
        # Only add noise if t > 0, otherwise just return the mean
        # This ensures x_0 is deterministic and not noisy
        is_last_step = (t == 0).view((-1,) + (1,) * (len(xt.shape) - 1))
        noise = torch.where(is_last_step, 0.0, noise)
        
        # Calculate result
        result = mean + torch.sqrt(var) * noise
        
        # Ensure output shape consistency
        if result.shape != original_shape:
            try:
                result = result.view(original_shape)
            except:
                try:
                    if result.dim() == 3 and original_shape[1] == result.shape[2] and original_shape[2] == result.shape[1]:
                        result = result.transpose(1, 2)
                    else:
                        result = result.reshape(batch_size, -1).reshape(original_shape)
                except:
                    print(f"Error: Failed to maintain shape consistency in p_sample")
        
        return result

    def p_sample_noise(self, xt: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        """
        Sample from p_θ(x_{t-1}|x_t) with subject-specific conditioning.
        Simplified to use direct p_sample without subject conditioning.
        
        Args:
            xt: Noisy input at time step t [batch, seq_len, channels]
            t: Timestep indices [batch]
            s: Subject indices for conditioning [batch] (ignored in simplified version)
            
        Returns:
            x_{t-1}: Sample with less noise
        """
        # Simply use the regular p_sample without subject conditioning
        return self.p_sample(xt, t)

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, debug=False):
        """
        Calculate the simplified diffusion loss:
        L_simple = E_t,x_0,ε[||ε - ε_θ(x_t, t)||^2]
        
        Args:
            x0: Original clean data [batch, seq_len, channels]
            noise: Optional pre-generated noise
            debug: Whether to print debug information
            
        Returns:
            loss: The MSE loss between predicted and actual noise
        """
        local_debug = debug or self.debug
        batch_size = x0.shape[0]
        
        if local_debug:
            print(f"Input shape: {x0.shape}")
        
        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        
        # Generate random noise if not provided
        if noise is None:
            noise = torch.randn_like(x0)
            
        if local_debug:
            print(f"Noise shape: {noise.shape}")
        
        # Get noisy samples x_t at timestep t
        xt = self.q_sample(x0, t, eps=noise)
        
        if local_debug:
            print(f"Noisy sample shape: {xt.shape}")
            print(f"Timestep shape: {t.shape}")
        
        # Predict noise
        predicted_noise = self.eps_model(xt, t)
        
        if local_debug:
            print(f"Predicted noise shape: {predicted_noise.shape}")
        
        # Handle shape mismatch if necessary
        if predicted_noise.shape != noise.shape:
            if local_debug:
                print(f"Shape mismatch: Predicted {predicted_noise.shape}, Expected {noise.shape}")
            try:
                predicted_noise = predicted_noise.view(noise.shape)
            except:
                if local_debug:
                    print("Reshape failed, falling back to MSE loss between differently shaped tensors")
        
        # MSE loss between predicted and actual noise
        noise_loss = F.mse_loss(noise, predicted_noise)
        
        return noise_loss

    # Maintain compatibility with the original implementation
    def loss_with_diff_constraint(self, x0: torch.Tensor, label: torch.Tensor, 
                            noise: Optional[torch.Tensor] = None, debug=False, 
                            noise_content_kl_co=1.0, arc_subject_co=0.1, orgth_co=2.0):
        """
        Simplified to just use the basic loss function without additional constraints.
        
        Args:
            x0: Original clean data [batch, seq_len, channels]
            label: Subject labels for classification [batch] (ignored in simplified version)
            noise: Optional pre-generated noise
            debug: Whether to print debug information
            noise_content_kl_co: Weight for KL divergence loss (ignored in simplified version)
            arc_subject_co: Weight for subject classification loss (ignored in simplified version)
            orgth_co: Weight for orthogonality loss (ignored in simplified version)
            
        Returns:
            total_loss: The loss value
            constraint_penalty: Placeholder value (0)
            noise_content_kl: Placeholder value (0)
            subject_arc_loss: Placeholder value (0)
            loss_orth: Placeholder value (0)
        """
        # Just use the simple loss for prediction
        noise_prediction_loss = self.loss(x0, noise, debug)
        
        # Return placeholder values for other metrics to maintain API compatibility
        constraint_penalty = torch.tensor(0.0, device=x0.device)
        noise_content_kl = torch.tensor(0.0, device=x0.device)
        subject_arc_loss = torch.tensor(0.0, device=x0.device)
        loss_orth = torch.tensor(0.0, device=x0.device)
        
        return noise_prediction_loss, constraint_penalty, noise_content_kl, subject_arc_loss, loss_orth

    def sample(self, shape, sample_steps=None):
        """
        Generate new samples from the diffusion model.
        
        Args:
            shape: Shape of samples to generate [batch, seq_len, channels] or [batch, channels, seq_len]
            sample_steps: Number of sampling steps (if None, uses n_steps)
            
        Returns:
            Samples generated by the diffusion process
        """
        if sample_steps is None:
            sample_steps = self.n_steps
            
        # Start from pure noise
        batch_size = shape[0]
        device = self.device
        
        # Sample x_T from standard normal distribution
        x = torch.randn(shape, device=device)
        
        # Store original shape to ensure consistent output format
        original_shape = x.shape
        
        # Progressively denoise x_t for t = T, T-1, ..., 1
        for t_step in range(sample_steps):
            # Current timestep (going backwards)
            t = self.n_steps - t_step - 1
            
            # Create a batch of timesteps
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Sample from p(x_{t-1} | x_t)
            with torch.no_grad():
                x = self.p_sample(x, t_batch)
                
                # Ensure shape consistency through each step
                if x.shape != original_shape:
                    x = x.view(original_shape)
                
        return x
    
    def sample_with_subject(self, shape, subject_ids, sample_steps=None):
        """
        Simplified to just use standard sampling without conditioning.
        
        Args:
            shape: Shape of samples to generate [batch, seq_len, channels]
            subject_ids: Subject IDs to condition on [batch] (ignored in simplified version)
            sample_steps: Number of sampling steps (if None, uses n_steps)
            
        Returns:
            Samples generated by the diffusion process
        """
        # Just call the regular sample method without subject conditioning
        return self.sample(shape, sample_steps)

class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer.
    Inspired by "Attention is All You Need" https://arxiv.org/abs/1706.03762.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="gelu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention block
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        # Cross-attention block with encoder memory
        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        # Feed-forward network block
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt

class Model(nn.Module):
    """
    Model class with a decoder integrated into the pretraining branch.
    Supports supervised, finetuning, and pretraining tasks.
    For pretraining tasks, the model now computes an encoder representation,
    projects it via a projection head, and also reconstructs the input via a decoder.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        # Embedding configuration
        if configs.no_temporal_block and configs.no_channel_block:
            raise ValueError("At least one of the two blocks should be True")
        if configs.no_temporal_block:
            patch_len_list = []
        else:
            patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        if configs.no_channel_block:
            up_dim_list = []
        else:
            up_dim_list = list(map(int, configs.up_dim_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")  # , "jitter", "mask", "channel", "drop"
        # Set default augmentations if needed for contrastive pretraining
        if augmentations == ["none"] and "pretrain" in self.task_name:
            augmentations = ["flip", "frequency", "jitter", "mask", "channel", "drop"]

        # Encoder embedding
        self.enc_embedding = TokenChannelEmbedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            patch_len_list,
            up_dim_list,
            stride_list,
            configs.dropout,
            augmentations,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ADformerLayer(
                        len(patch_len_list),
                        len(up_dim_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                        configs.no_inter_attn,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)

        # Branches for different tasks
        if self.task_name in ["supervised", "finetune"]:
            self.classifier = nn.Linear(
                configs.d_model * len(patch_num_list) + configs.d_model * len(up_dim_list),
                configs.num_class,
            )
        elif self.task_name in ["pretrain_lead", "pretrain_moco", "diffusion"]:
            input_dim = configs.d_model * len(patch_num_list) + configs.d_model * len(up_dim_list)
            self.projection_head = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(input_dim * 2, configs.d_model)
            )
            # Add decoder components from the removed diffusion branch
            self.dec_embedding = TokenChannelEmbedding(
                configs.enc_in,
                configs.seq_len,
                configs.d_model,
                patch_len_list,
                up_dim_list,
                stride_list,
                configs.dropout,
                augmentations,
            )
            d_layers = getattr(configs, "d_layers", 1)
            self.decoder_layers = nn.ModuleList([
                DecoderLayer(configs.d_model, configs.n_heads, configs.d_ff, dropout=configs.dropout, activation=configs.activation)
                for _ in range(d_layers)
            ])
            self.decoder_projection = nn.Linear(configs.d_model, configs.enc_in)
            if self.task_name == "pretrain_moco":
                K = configs.K  # queue size
                feat_dim = configs.d_model
                self.register_buffer("queue", torch.randn(feat_dim, K))
                self.queue = F.normalize(self.queue, dim=0)
                self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        else:
            raise ValueError("Task name not recognized or not implemented in the model")

        # Add timestep embedding for diffusion
        if configs.task_name == "diffusion":
            self.timestep_embed = nn.Embedding(configs.n_steps, configs.d_model)
            
            # Add components needed for the simplified diffusion forward path
            # These will be used in the diffusion_forward method
            self.diffusion_mlp = nn.Sequential(
                nn.Linear(configs.seq_len * configs.enc_in + configs.d_model, configs.d_model * 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model * 2, configs.seq_len * configs.enc_in)
            )
            
            self.noise_pred = nn.Sequential(
                nn.Linear(configs.d_model * (len(patch_num_list) + len(up_dim_list)), 
                        configs.enc_in * configs.seq_len),
                nn.Unflatten(1, (configs.enc_in, configs.seq_len))
            )

    def supervised(self, x_enc, x_mark_enc):
        # Encoder branch for supervised tasks
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
        enc_out_t, enc_out_c, _, _ = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
        if enc_out_t is None:
            enc_out = enc_out_c
        elif enc_out_c is None:
            enc_out = enc_out_t
        else:
            enc_out = torch.cat((enc_out_t, enc_out_c), dim=1)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.classifier(output)
        return output

    def pretrain(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Encoder: get latent representation
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
        enc_out_t, enc_out_c, _, _ = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
        if enc_out_t is None:
            memory = enc_out_c
        elif enc_out_c is None:
            memory = enc_out_t
        else:
            memory = torch.cat((enc_out_t, enc_out_c), dim=1)
        # Projection head branch
        encoded = self.act(memory)
        encoded = self.dropout(encoded)
        encoded_flat = encoded.reshape(encoded.shape[0], -1)
        repr_out = self.projection_head(encoded_flat)

        # Decoder branch: reconstruct the original input
        dec_out_t, dec_out_c = self.dec_embedding(x_dec)
        # Convert lists to tensors if needed
        if isinstance(dec_out_t, list):
            dec_out_t = torch.cat(dec_out_t, dim=1)
        if isinstance(dec_out_c, list):
            dec_out_c = torch.cat(dec_out_c, dim=1)
        if dec_out_t is None:
            dec_out = dec_out_c
        elif dec_out_c is None:
            dec_out = dec_out_t
        else:
            dec_out = torch.cat((dec_out_t, dec_out_c), dim=1)
        
        # Permute for nn.MultiheadAttention (seq_len, batch, d_model)
        dec_out = dec_out.transpose(0, 1)
        memory_dec = memory.transpose(0, 1)
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, memory_dec)
        dec_out = dec_out.transpose(0, 1)
        rec_out = self.decoder_projection(dec_out)
        return repr_out, rec_out
    
    def diffusion_forward(self, xt, t=None):
        """
        Forward pass for the diffusion model noise prediction.
        Args:
            xt: Noisy input at time step t - shape [batch, seq_len, channels] or [batch, channels, seq_len]
            t: Time step indices, can be None during initialization
        Returns:
            Predicted noise - same shape as input
        """
        # Store original shape for output reshaping
        original_shape = xt.shape
        batch_size = xt.shape[0]
        
        # Fix input shape inconsistency
        # The model expects [batch, seq_len, channels] but might get [batch, channels, seq_len]
        # Check dimensions and transpose if needed
        if xt.dim() == 3:
            # If channels (enc_in) is much smaller than seq_len, we can detect the right format
            # In your case, channels=19, seq_len=128
            if xt.shape[1] == self.enc_in and xt.shape[2] == self.seq_len:
                # Input is [batch, channels, seq_len], transpose to [batch, seq_len, channels]
                print(f"Transposing input from shape {xt.shape} to match expected format")
                xt = xt.transpose(1, 2)
            # elif xt.shape[1] == self.seq_len and xt.shape[2] == self.enc_in:
            #     # Already in correct [batch, seq_len, channels] format
            #     print(f"Input shape {xt.shape} already in expected format")
            # else:
            #     # If dimensions don't clearly match expected values, log a warning
            #     print(f"WARNING: Unexpected input shape {xt.shape}. Expected [{batch_size}, {self.seq_len}, {self.enc_in}] or [{batch_size}, {self.enc_in}, {self.seq_len}]")
        
        # Check if t is None and create a default t if needed
        if t is None:
            # Use timestep 0 as default
            t = torch.zeros(batch_size, dtype=torch.long, device=xt.device)
        
        # Get timestep embeddings
        t_emb = self.timestep_embed(t)  # [batch, d_model]
        
        # Process through encoder
        enc_out_t, enc_out_c = self.enc_embedding(xt)
        
        # Inject timestep embedding into encoder outputs
        if isinstance(enc_out_t, list):
            for i in range(len(enc_out_t)):
                enc_out_t[i] = enc_out_t[i] + t_emb.unsqueeze(1)
        else:
            enc_out_t = enc_out_t + t_emb.unsqueeze(1)
            
        if isinstance(enc_out_c, list):
            for i in range(len(enc_out_c)):
                enc_out_c[i] = enc_out_c[i] + t_emb.unsqueeze(1)
        else:
            enc_out_c = enc_out_c + t_emb.unsqueeze(1)
        
        # Process through encoder
        enc_out_t, enc_out_c, _, _ = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
        
        # Combine encoder outputs
        if enc_out_t is None:
            enc_out = enc_out_c
        elif enc_out_c is None:
            enc_out = enc_out_t
        else:
            enc_out = torch.cat((enc_out_t, enc_out_c), dim=1)
        
        # Apply activation and dropout
        output = self.act(enc_out)
        output = self.dropout(output)
        
        # Flatten and predict noise
        output_flat = output.reshape(batch_size, -1)
        noise_pred_flat = self.noise_pred(output_flat)
        
        # Reshape to original input shape
        # The noise_pred outputs [batch, channels, seq_len] or [batch, seq_len, channels]
        noise_pred = noise_pred_flat
        
        # Ensure the output has the same shape as the input
        if noise_pred.shape != original_shape:
            # print(f"Reshaping output from {noise_pred.shape} to match input shape {original_shape}")
            # Try direct reshaping
            try:
                noise_pred = noise_pred.view(original_shape)
            except:
                # If direct reshape fails, try transposing dimensions
                try:
                    if noise_pred.dim() == 3 and original_shape[1] == noise_pred.shape[2] and original_shape[2] == noise_pred.shape[1]:
                        noise_pred = noise_pred.transpose(1, 2)
                    else:
                        # Last resort: just reshape to match original shape's sizes
                        noise_pred = noise_pred.reshape(batch_size, -1).reshape(original_shape)
                except:
                    print(f"ERROR: Failed to reshape output to match input shape. Output shape: {noise_pred.shape}, Input shape: {original_shape}")
        
        return noise_pred
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, t=None):
        """
        Forward pass for all model tasks.
        For diffusion task, t contains timestep indices.
        """
        if self.task_name == "diffusion":
            return self.diffusion_forward(x_enc, t)
        elif self.task_name in ["supervised", "finetune"]:
            return self.supervised(x_enc, x_mark_enc)
        elif self.task_name in ["pretrain_lead", "pretrain_moco"]:
            return self.pretrain(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            raise ValueError("Task name not recognized or not implemented in the model")