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

# Our data loader
from torch_geometric.loader import DataLoader
from pyg_dataset import pyg_dataset

# NetlistGNN data loader
from load_netlistgnn_data import load_data

# Deep Graph Library (DGL)
import dgl

# To convert from our data to DGL
from convert_to_dgl import *

# Create train dataset
# train_dataset = pyg_dataset(data_dir = '../../data/2023-03-06_data/', fold_index = 0, split = 'train', target = 'demand')
test_dataset = pyg_dataset(data_dir = '../../data/2023-03-06_data/', fold_index = 0, split = 'test', target = 'demand', load_pe = True, num_eigen = 10)

# Data loaders
batch_size = 1
# train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False)

# print('Number of training examples:', len(train_dataset))
print('Number of testing examples:', len(test_dataset))

# Conversion to DGL format
for batch_idx, data in enumerate(test_dataloader):
    print(batch_idx)
    print(data)
    
    dgl_data = convert_to_dgl(data)
    break

print('Done')
