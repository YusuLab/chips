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

# Functionality to convert from our data to DGL format
def convert_to_dgl(pyg_data):
    # Number of instances
    num_instances = data.y.size(0)

    # Number of nets
    num_nets = data.x.size(0) - num_instances

    # Instance indices
    instance_idx = data.edge_index[0, :]

    # Net indices
    net_idx = data.edge_index[1, :]

    # Make net indices to be starting from 0 again
    net_idx = net_idx - num_instances

    # Number of edges
    num_edges = net_idx.numel()

    # Checking the quality
    assert(torch.min(instance_idx).item() == 0)
    assert(torch.max(instance_idx).item() == num_instances - 1)
    assert(torch.min(net_idx).item() == 0)
    assert(torch.max(net_idx).item() == num_nets - 1)

    # Compute the near relationship
    near_idx_1 = torch.tensor(np.array([idx for idx in range(num_instances)]))
    near_idx_2 = torch.tensor(np.array([idx for idx in range(num_instances)]))
    near_features = torch.zeros(num_instances)

    # Compute the net's degree
    net_degree = np.zeros(num_nets)

    '''
    net_degree = np.zeros(num_nets)
    for idx in range(num_edges):
        net_degree[net_idx[idx]] += 1
    '''
    
    net_degree = torch.tensor(net_degree)

    # Net label
    net_label = torch.zeros(num_nets)

    # Deep Graph Library
    dgl_data = None

    hetero_graph = dgl.heterograph(
        {
            ('node', 'near', 'node'): (near_idx_1, near_idx_2),
            ('node', 'pins', 'net'): (instance_idx, net_idx),
            ('net', 'pinned', 'node'): (net_idx, instance_idx),
        }, 
        num_nodes_dict = {
            'node': num_instances, 
            'net': num_nets
        }
    )

    # Node & Edge features
    hetero_graph.nodes['node'].data['hv'] = pyg_data.x[ : num_instances, :]
    hetero_graph.nodes['node'].data['pos_code'] = pyg_data.evects[ : num_instances, :]
    hetero_graph.nodes['net'].data['hv'] = pyg_data.x[num_instances :, :]
    hetero_graph.nodes['net'].data['degree'] = net_degree
    hetero_graph.nodes['net'].data['label'] = net_label
    hetero_graph.edges['pins'].data['he'] = pyg_data.edge_attr
    hetero_graph.edges['pinned'].data['he'] = pyg_data.edge_attr
    hetero_graph.edges['near'].data['he'] = near_features

    return dgl_data

