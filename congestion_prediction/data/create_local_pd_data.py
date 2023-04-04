import numpy as np
import os
import time
import scipy
import pickle
import networkx as nx

import sys

from pd_util import *

from tqdm import tqdm

from persim import PersImage
from persim import PersistenceImager

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
data_dir = '2023-03-06_data/'

for sample in range(N):
    # Connection data 
    fn = data_dir + '/' + str(sample) + f'.{graph_rep}.pkl'
    f = open(fn, "rb")
    dictionary = pickle.load(f)
    f.close()
    print('Read file', fn)
    
    edge_index = dictionary['edge_index'].T
    G_out = nx.DiGraph(list(edge_index))
    G_in = G_out.reverse()
    pd_dictionary = {}
    
    
    for idx in range(2):
        if idx == 0:
            G = G_out
        else:
            G = G_in 
            
        nodelist = list(G.nodes)
        num_nodes = max(nodelist) + 1
        all_zero_dgms = [[0.0] for i in range(num_nodes)]
        all_one_dgms = [[0.0 for idx in range(4*4)] for i in range(num_nodes)]
        all_neigh = [[0 for idx in range(4)] for i in range(num_nodes)]
        
        pimgr_0 = PersistenceImager(pixel_size=4, birth_range=(0, 4), pers_range=(0, 4))
        pimgr_1 = PersistenceImager(pixel_size=1, birth_range=(0, 4), pers_range=(0, 4))
        
        
        for node in tqdm(nodelist):
            if G.degree[node] >= 100:
                zero_dgm, one_dgm, neigh_lst = gen_local_pd(G, node, 4)
                L_0 = pimgr_0.transform(zero_dgm)
                L_1 = pimgr_1.transform(one_dgm)
                all_zero_dgms[node] = L_0[0].tolist()
                all_one_dgms[node] = L_1.flatten().tolist()
                all_neigh[node] = neigh_lst

                G.remove_node(node)
                
        nodelist = list(G.nodes)
        for node in tqdm(nodelist):
            zero_dgm, one_dgm, neigh_lst = gen_local_pd(G, node, 4)
            if neigh_lst[0] == 0:
                continue
            L_0 = pimgr_0.transform(zero_dgm)
            L_1 = pimgr_1.transform(one_dgm)
            all_zero_dgms[node] = L_0.flatten().tolist()
            all_one_dgms[node] = L_1.flatten().tolist()
            all_neigh[node] = neigh_lst
                    
        pd_dictionary[f"all_zero_dgms_{idx}"] = np.array(all_zero_dgms)
        pd_dictionary[f"all_one_dgms_{idx}"] = np.array(all_one_dgms)
        pd_dictionary[f"all_neigh_{idx}"] = np.array(all_neigh)

    fn = data_dir + '/' + str(sample) + '.local_pd.pkl'
    f = open(fn, "wb")
    pickle.dump(pd_dictionary, f)
    f.close()
    print('Save file', fn)