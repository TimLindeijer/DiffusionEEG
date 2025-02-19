import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.ADformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ADformerLayer
from layers.Embed import TokenChannelEmbedding
import numpy as np

class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer for diffusion.
    Inspired by "Attention is All You Need" [1](https://arxiv.org/abs/1706.03762).
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
    Model class with an added decoder for diffusion.
    This version still supports supervised and pretraining tasks while also 
    adding a branch (task_name == "diffusion") to allow a diffusion-style decoder.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in

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
        augmentations = configs.augmentations.split(",")
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
                for l in range(configs.e_layers)
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
        elif self.task_name in ["pretrain_lead", "pretrain_moco"]:
            input_dim = configs.d_model * len(patch_num_list) + configs.d_model * len(up_dim_list)
            self.projection_head = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(input_dim * 2, configs.d_model)
            )
            if self.task_name == "pretrain_moco":
                K = configs.K  # queue size
                feat_dim = configs.d_model
                self.register_buffer("queue", torch.randn(feat_dim, K))
                self.queue = F.normalize(self.queue, dim=0)
                self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        elif self.task_name == "diffusion":
            # Decoder embedding (separate from the encoder)
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
            # Create decoder layers (default to one layer if "d_layers" is not specified)
            d_layers = getattr(configs, "d_layers", 1)
            self.decoder_layers = nn.ModuleList([
                DecoderLayer(configs.d_model, configs.n_heads, configs.d_ff, dropout=configs.dropout, activation=configs.activation)
                for _ in range(d_layers)
            ])
            # Final projection layer to reconstruct the original input dimension
            self.decoder_projection = nn.Linear(configs.d_model, configs.enc_in)
        else:
            raise ValueError("Task name not recognized or not implemented in the model")

    def supervised(self, x_enc, x_mark_enc):
        # Encoder branch for supervised tasks
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
        enc_out_t, enc_out_c, attns_t, attns_c = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
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

    def pretrain(self, x_enc, x_mark_enc):
        # Encoder branch for pretraining tasks
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
        enc_out_t, enc_out_c, attns_t, attns_c = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
        if enc_out_t is None:
            enc_out = enc_out_c
        elif enc_out_c is None:
            enc_out = enc_out_t
        else:
            enc_out = torch.cat((enc_out_t, enc_out_c), dim=1)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        repr_out = self.projection_head(output)
        return output, repr_out

    def diffusion(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Encoder: get latent representation
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
        enc_out_t, enc_out_c, attns_t, attns_c = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
        if enc_out_t is None:
            memory = enc_out_c
        elif enc_out_c is None:
            memory = enc_out_t
        else:
            memory = torch.cat((enc_out_t, enc_out_c), dim=1)
        # Decoder: embed the decoder input
        dec_out_t, dec_out_c = self.dec_embedding(x_dec)
        if dec_out_t is None:
            dec_out = dec_out_c
        elif dec_out_c is None:
            dec_out = dec_out_t
        else:
            dec_out = torch.cat((dec_out_t, dec_out_c), dim=1)
        # Permute for nn.MultiheadAttention (seq_len, batch, d_model)
        dec_out = dec_out.transpose(0, 1)
        memory = memory.transpose(0, 1)
        # Pass through decoder layers with cross-attention to encoder memory
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, memory)
        dec_out = dec_out.transpose(0, 1)
        # Project decoder output to reconstruct original input dimensions
        output = self.decoder_projection(dec_out)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["supervised", "finetune"]:
            return self.supervised(x_enc, x_mark_enc)
        elif self.task_name in ["pretrain_lead", "pretrain_moco"]:
            return self.pretrain(x_enc, x_mark_enc)
        elif self.task_name == "diffusion":
            return self.diffusion(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            raise ValueError("Task name not recognized or not implemented in the model")
