import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import os
import time
import argparse
import pickle

# Dataset
from RosettaStone_Dataset_PyG import RosettaStone_Dataset_PyG

# Data directory
data_dir = '../data/RosettaStone-GraphData-2023-01-21/'

# Create PyG datasets
train_dataset = RosettaStone_Dataset_PyG(data_dir = data_dir, split = 'train')
valid_dataset = RosettaStone_Dataset_PyG(data_dir = data_dir, split = 'valid')
test_dataset = RosettaStone_Dataset_PyG(data_dir = data_dir, split = 'test')

# Create PyG data loaders
batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False)

print('Done')
