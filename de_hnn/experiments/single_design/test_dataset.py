import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim

import numpy as np
import os
import time
import argparse
import scipy
import pickle

from torch_geometric.loader import DataLoader

from pyg_dataset import pyg_dataset
from pyg_dataset_sparse import pyg_dataset_sparse

import sys

graph_index = sys.argv[1]


# Create train dataset
dataset = pyg_dataset('../../data/2023-03-06_data/', graph_index = graph_index , target = 'demand', load_pe = False, num_eigen = 0, load_global_info = False, load_pd = True, concat = False)
# Data loaders
batch_size = 1
dataloader = DataLoader(dataset, batch_size, shuffle = True)

for batch_idx, data in enumerate(dataloader):
    print(batch_idx)
    print(data)
    node_dim = data.x.size(1)
    edge_dim = 1#data.edge_attr.size(1)
    num_outputs = data.y.size(1)
    break

print('Number of node features:', node_dim)
print('Number of edge features:', edge_dim)
print('Number of outputs:', num_outputs)
print(dataset[0])
print('Done')
