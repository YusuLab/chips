import numpy as np
import os
import time
import scipy
import pickle
import networkx as nx

import sys

from torch_geometric.loader import DataLoader

from tqdm import tqdm

import matplotlib.pyplot as plt

abs_dir = '/data/son/Research/chips/congestion_prediction/data/'
raw_data_dir = abs_dir + 'RosettaStone-GraphData-2023-03-06/'
graph_rep = "star"

designs_list = [
    'superblue1',
    'superblue2',
    'superblue3',
    'superblue4',
    'superblue18',
    'superblue19'
]
num_designs = len(designs_list)
num_variants_list = [
    5,
    5,
    6,
    5,
    5,
    6
]
assert num_designs == len(num_variants_list)

# Generate all names
sample_names = []
corresponding_design = []
corresponding_variant = []
for idx in range(num_designs):
    for variant in range(num_variants_list[idx]):
        sample_name = raw_data_dir + designs_list[idx] + '/' + str(variant + 1) + '/'
        sample_names.append(sample_name)
        corresponding_design.append(designs_list[idx])
        corresponding_variant.append(variant + 1)

# Synthetic data
N = len(sample_names)
data_dir = '../../data/2023-03-06_data'

for sample in range(N):
    # Connection data 
    graph_rep = 'star'
    fn = data_dir + '/' + str(sample) + f'.{graph_rep}.pkl'
    f = open(fn, "rb")
    dictionary = pickle.load(f)
    f.close()
    print('Read file', fn)
    
    with open(f"../../data/2023-03-06_data/{sample}.node_features.pkl", "rb") as f:
        d = pickle.load(f)

    edge_index = dictionary['edge_index']
    instance_idx = dictionary['instance_idx']
    num_nodes = d['num_instances']
    
    edge_index = edge_index.tolist()
    G = nx.DiGraph(edge_index)
    nodelist = list(G.nodes)
    
    in_degree = [0 for i in range(num_nodes)]
    out_degree = [0 for i in range(num_nodes)]

    for node in tqdm(nodelist):
        in_degree[node] = G.in_degree[node]
        out_degree[node] = G.out_degree[node]
    
    
    instance_features = np.concatenate([d['instance_features'], np.array(in_degree).reshape(-1, 1), np.array(out_degree).reshape(-1, 1)], axis=1)
    d['instance_features'] = instance_features

    with open(f"../../data/2023-03-06_data/{sample}.node_features.pkl", "wb") as f:
        pickle.dump(d, f)