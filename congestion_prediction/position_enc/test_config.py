from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

from posenc_config import set_cfg_posenc 

config = CN()
config = set_cfg_posenc(config)

print(config)
print('Done')
