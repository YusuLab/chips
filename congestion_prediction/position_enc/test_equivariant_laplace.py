# Create a PyG data
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype = torch.long)
x = torch.tensor([[-1], [0], [1]], dtype = torch.float)
data = Data(x = x, edge_index = edge_index)

print(data)

# Create a configuration for position encodings (including SignNet)
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN
from posenc_config import set_cfg_posenc

config = CN()
config = set_cfg_posenc(config)
config.posenc_EquivStableLapPE.max_freqs = 5
config.posenc_EquivStableLapPE.raw_norm_type = 'batchnorm'

print(config)

# Precompute the statistics for position encodings
from posenc_stats import compute_posenc_stats

data = compute_posenc_stats(data, ['EquivStableLapPE'], True, config)

# Use equivariant and stable Laplace position encoder
from equivstable_laplace_pos_encoder import EquivStableLapPENodeEncoder

module = EquivStableLapPENodeEncoder(cfg = config, dim_emb = 20)

data = module.apply(data)

print(data)
print('Done')

