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

# Create train dataset
train_dataset = pyg_dataset('../../data/synthetic_data/', fold_index = 4, split = 'train')
test_dataset = pyg_dataset('../../data/synthetic_data/', fold_index = 4, split = 'test')

# Data loaders
batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False)

print('Number of training examples:', len(train_dataset))
print('Number of testing examples:', len(test_dataset))

for batch_idx, data in enumerate(train_dataloader):
    print(batch_idx)
    print(data)
    node_dim = data.x.size(1)
    edge_dim = data.edge_attr.size(1)
    num_outputs = data.y.size(1)
    break

for batch_idx, data in enumerate(test_dataloader):
    print(batch_idx)
    print(data)
    assert node_dim == data.x.size(1)
    assert edge_dim == data.edge_attr.size(1)
    assert num_outputs == data.y.size(1)
    break

print('Number of node features:', node_dim)
print('Number of edge features:', edge_dim)
print('Number of outputs:', num_outputs)

print('Done')
