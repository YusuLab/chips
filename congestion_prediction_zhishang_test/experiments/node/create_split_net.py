import numpy as np
import math
import random
import pickle
import sys
from tqdm import tqdm

# Fix random seed
random.seed(123456789)

# Data directory
data_dir = '../../data/2023-03-06_data/'

# Index of the graph
graph_index = sys.argv[1]

# Read features
f = open(data_dir + '/' + str(graph_index) + '.node_features.pkl', 'rb')
dictionary = pickle.load(f)
f.close()
design_name = dictionary['design']
instance_features = dictionary['instance_features']

f = open(data_dir + '/' + str(graph_index) + '.net_demand_capacity.pkl', 'rb')
dictionary = pickle.load(f)
f.close()
instance_features = np.array(dictionary['demand'])

X = instance_features

print('Graph index:', graph_index)
print('Design name:', design_name)
print(X.shape)

num_samples = X.shape[0]
perm = np.random.permutation(num_samples)

train_percent = 60
valid_percent = 20
test_percent = 20

num_train = num_samples * train_percent // 100
num_valid = num_samples * valid_percent // 100
num_test = num_samples - num_train - num_valid

train_indices = perm[:num_train]
valid_indices = perm[num_train:num_train+num_valid]
test_indices = perm[num_train+num_valid:]

assert train_indices.shape[0] == num_train
assert valid_indices.shape[0] == num_valid
assert test_indices.shape[0] == num_test

print('Number of training samples:', num_train)
print('Number of validation samples:', num_valid)
print('Number of testing samples:', num_test)

file_name = data_dir + '/' + str(graph_index) + '.bipartite.pkl'
f = open(file_name, 'rb')
dictionary = pickle.load(f)
f.close()
row = dictionary['instance_idx']
col = dictionary['net_idx']

valid_indices_inst = []
test_indices_inst = []

for idx in tqdm(range(len(row))):
    net_idx = col[idx]
    if net_idx in valid_indices:
        valid_indices_inst.append(row[idx])
    elif net_idx in test_indices:
        test_indices_inst.append(row[idx])

valid_indices_inst = np.unique(valid_indices_inst)
test_indices_inst = np.unique(test_indices_inst)

dictionary = {
    'train_indices': train_indices,
    'valid_indices': valid_indices,
    'test_indices': test_indices,
    'valid_indices_inst': valid_indices_inst,
    'test_indices_inst': test_indices_inst
}
f = open(str(graph_index) + '.split_net.pkl', 'wb')
pickle.dump(dictionary, f)
f.close()

print('Done')
