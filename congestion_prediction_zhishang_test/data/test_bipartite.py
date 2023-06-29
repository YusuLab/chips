import numpy as np
import pickle

data_dir = '/data/zluo/chips/congestion_prediction/data/2023-03-06_data/'
sample = 0

# Read connection
file_name = data_dir + '/' + str(sample) + '.bipartite.pkl'
f = open(file_name, 'rb')
dictionary = pickle.load(f)
f.close()

instance_idx = dictionary['instance_idx']
net_idx = dictionary['net_idx']
edge_attr = dictionary['edge_attr']
edge_dir = dictionary['edge_dir']

print(instance_idx.shape)
print(net_idx.shape)
print(edge_attr.shape)
print(edge_dir.shape)

print(instance_idx)
print(net_idx)
print(edge_dir)

print('Done')
