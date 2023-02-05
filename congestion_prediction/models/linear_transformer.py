import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch

import math

# Create a configuration for position encodings (including SignNet)
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN
from posenc_config import set_cfg_posenc

# Precompute the statistics for position encodings
from posenc_stats import compute_posenc_stats

# Use Laplace position encoder
from laplace_pos_encoder import LapPENodeEncoder

# Use SignNet encoder
from signnet_pos_encoder import SignNetNodeEncoder

# Use kernel position encoder
from kernel_pos_encoder import KernelPENodeEncoder

# Installation: pip install linear-attention-transformer
# Source: https://github.com/lucidrains/linear-attention-transformer
from linear_attention_transformer import LinearAttentionTransformer

class Linear_Transformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_seq_len = 512, heads = 8, depth = 1, n_local_attn_heads = 1, pe_type = None, pe_dim = 16, device = 'cuda'):
        super(Linear_Transformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.heads = heads
        self.depth = depth
        self.n_local_attn_heads = n_local_attn_heads
        self.pe_type = pe_type
        self.pe_dim = pe_dim
        self.device = device

        # Configuration
        self.pe_config = CN()
        self.pe_config = set_cfg_posenc(self.pe_config)

        if self.pe_type == 'LapPE':
            self.pe_config.posenc_LapPE.model = 'DeepSet'
            self.pe_config.posenc_LapPE.post_layers = 2
            self.pe_config.posenc_LapPE.dim_pe = self.pe_dim
        elif self.pe_type == 'SignNet':
            self.pe_config.posenc_SignNet.model = 'DeepSet'
            self.pe_config.posenc_SignNet.post_layers = 2
            self.pe_config.posenc_SignNet.dim_pe = self.pe_dim
        elif self.pe_type == 'RWSE':
            self.pe_config.posenc_RWSE.kernel.times = [1, 2, 3, 4, 5]
            self.pe_config.posenc_RWSE.model = 'mlp'
            self.pe_config.posenc_RWSE.post_layers = 2
            self.pe_config.posenc_RWSE.dim_pe = self.pe_dim
        else:
            assert self.pe_type is None

        # Position Encoder
        if self.pe_type == 'LapPE':
            self.pe_model = LapPENodeEncoder(cfg = self.pe_config, dim_in = input_dim, dim_emb = hidden_dim, expand_x = True)
        elif self.pe_type == 'SignNet':
            self.pe_model = SignNetNodeEncoder(cfg = self.pe_config, dim_in = input_dim, dim_emb = hidden_dim, expand_x = True)
        elif self.pe_type == 'RWSE':
            self.pe_model = KernelPENodeEncoder(kernel_type = self.pe_type, cfg = self.pe_config, dim_in = input_dim, dim_emb = hidden_dim, expand_x = True)

        # Linear Transformer
        self.transformer_model = LinearAttentionTransformer(
            dim = self.hidden_dim,
            heads = self.heads,
            depth = self.depth,
            max_seq_len = self.max_seq_len,
            n_local_attn_heads = self.n_local_attn_heads
        )

        # Top layer
        self.fc1 = nn.Linear(self.hidden_dim, 512)
        self.fc2 = nn.Linear(512, self.output_dim)

    def forward(self, batched_data):
        # Pre-computation for PE
        batched_data = compute_posenc_stats(batched_data, [self.pe_type], True, self.pe_config)
        batched_data.x = batched_data.x.to(device = self.device)

        if self.pe_type == 'LapPE':
            batched_data.EigVals = batched_data.EigVals.to(device = self.device)
            batched_data.EigVecs = batched_data.EigVecs.to(device = self.device)
        elif self.pe_type == 'SignNet':
            batched_data.eigvecs_sn = batched_data.eigvecs_sn.to(device = self.device)

        # Position Encoding
        pe = self.pe_model(batched_data)
        inputs, mask = to_dense_batch(pe.x, pe.batch)
        
        # Pad with zeros for enough tokens
        if inputs.size(1) < self.max_seq_len:
            zeros = torch.zeros(inputs.size(0), self.max_seq_len - inputs.size(1), inputs.size(2)).to(device = self.device)
            inputs = torch.cat([inputs, zeros], dim = 1)

        # Linear Transformer
        hidden = self.transformer_model(inputs)

        # Global readout
        latent = torch.mean(hidden, dim = 1)

        # Top layer
        latent = torch.tanh(self.fc1(latent))
        return self.fc2(latent)
