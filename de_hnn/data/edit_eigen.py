import numpy as np
import gzip
import json
from scipy.stats import binned_statistic_2d
import time
import pickle

import torch
import torch.nn.functional as F
from numpy.linalg import eigvals
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix, to_undirected, to_dense_adj)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

# Scipy eigendecomposition
from scipy.sparse.linalg import eigsh

abs_dir = '/data/zluo/'
raw_data_dir = abs_dir + 'new_data/'

designs_list = [
    'superblue1',
    'superblue2',
    'superblue3',
    'superblue4',
    'superblue5',
    'superblue6',
    'superblue7',
    'superblue18',
    'superblue19',
    'superblue9',
    'superblue11',
    'superblue14',
    'superblue16'
]
num_designs = len(designs_list)
num_variants_list = [
    5,
    5,
    6,
    5,
    6,
    6,
    6,
    5,
    6,
    6,
    6,
    6,
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
data_dir = '2023-03-06_data/'

# Eigendecomposition
k = 10
for idx in [0, 5, 10, 16, 21, 27, 33, 39, 40, 44, 50, 56, 62, 68]:
    fn = data_dir + str(idx) + '.eigen.' + str(k) + '.pkl'
    f = open(fn, "rb")
    dictionary = pickle.load(f)
    f.close()
    
    file_name = data_dir + str(idx) + '.node_features.pkl'
    f = open(file_name, 'rb')
    d = pickle.load(f)
    f.close()

    num_instances = d['num_instances']
    num_nets = d['num_nets']
    
    evects = dictionary['evects']
    
    dictionary['evects'] = np.concatenate([evects[:num_instances], np.zeros_like(evects[num_instances:])])
    
    fn = data_dir + str(idx) + '.eigen.' + str(k) + '.pkl'
    f = open(fn, "wb")
    pickle.dump(dictionary, f)
    f.close()
    print('Save file', fn)