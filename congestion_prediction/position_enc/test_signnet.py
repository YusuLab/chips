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
config.posenc_SignNet.model = 'DeepSet'
config.posenc_SignNet.post_layers = 2

print(config)

# Precompute the statistics for position encodings
from posenc_stats import compute_posenc_stats

data = compute_posenc_stats(data, ['SignNet'], True, config)

# Use SignNet encoder
from signnet_pos_encoder import SignNetNodeEncoder

module = SignNetNodeEncoder(cfg = config, dim_in = 1, dim_emb = 20, expand_x = True)

data = module.apply(data)

print(data)
print('Done')
