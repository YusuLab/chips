import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data

import numpy as np
import pickle

class pyg_dataset(Dataset):
    def __init__(self, data_dir, fold_index, split, total_samples = 32):
        super().__init__()
        self.data_dir = data_dir
        self.fold_index = fold_index
        self.split = split

        # Read cross-validation
        file_name = data_dir + '/6_fold_cross_validation.pkl'
        f = open(file_name, 'rb')
        dictionary = pickle.load(f)
        f.close()
        folds = dictionary['folds']

        # Take the sample indices
        test_indices = folds[self.fold_index]
        train_indices = [idx for idx in range(total_samples) if idx not in test_indices]

        if self.split == 'train':
            self.sample_indices = train_indices
        else:
            self.sample_indices = test_indices
        
        self.num_samples = len(self.sample_indices)
        print('Number of samples:', self.num_samples)

        # Read data
        self.data = []

        for sample in self.sample_indices:
            # Read node features
            file_name = data_dir + '/' + str(sample) + '.node_features.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()

            num_instances = dictionary['num_instances']
            num_nets = dictionary['num_nets']
            instance_features = torch.Tensor(dictionary['instance_features'])
            net_features = torch.zeros(num_nets, instance_features.size(1))
            x = torch.cat([instance_features, net_features], dim = 0)

            # Read learning targets
            file_name = data_dir + '/' + str(sample) + '.targets.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()

            demand = torch.Tensor(dictionary['demand'])
            y = torch.sum(demand, dim = 1).unsqueeze(dim = 1)

            # Read connection
            file_name = data_dir + '/' + str(sample) + '.bipartite.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()

            instance_idx = torch.Tensor(dictionary['instance_idx']).unsqueeze(dim = 1).long()
            net_idx = torch.Tensor(dictionary['net_idx']) + num_instances
            net_idx = net_idx.unsqueeze(dim = 1).long()

            edge_attr = torch.Tensor(dictionary['edge_attr']).unsqueeze(dim = 1).long()
            edge_index = torch.cat((instance_idx, net_idx), dim = 1)
            
            # PyG data
            example = Data()
            example.__num_nodes__ = x.size(0)
            example.x = x
            example.y = y
            example.edge_index = edge_index
            example.edge_attr = edge_attr

            self.data.append(example)

        print('Done reading data')

    def len(self):
        return self.num_samples

    def get(self, idx):
        return self.data[idx]

