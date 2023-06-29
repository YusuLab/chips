import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data

import numpy as np
import pickle

from tqdm import tqdm

class pyg_dataset(Dataset):
    def __init__(self, data_dir, fold_index, split, target, load_pe = False, num_eigen = 5, load_global_info = True, load_pd = False, total_samples = 32, graph_rep = 'bipartite', vn=False):
        super().__init__()
        self.data_dir = data_dir
        self.fold_index = fold_index
        self.split = split
        self.target = target
        assert target == 'demand' or target == 'capacity' or target == 'congestion'
        print('Learning target:', self.target)

        # Position encoding
        self.load_pe = load_pe
        self.num_eigen = num_eigen
        self.load_global_info = load_global_info
        self.load_pd = load_pd

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

        for sample in tqdm(self.sample_indices):
            print(sample)
            # Read node features
            file_name = data_dir + '/' + str(sample) + '.node_features.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()

            self.design_name = dictionary['design']

            num_instances = dictionary['num_instances']
            num_nets = dictionary['num_nets']
            instance_features = torch.Tensor(dictionary['instance_features'])
            instance_features = instance_features[:, 2:]

            # Read learning targets
            file_name = data_dir + '/' + str(sample) + '.net_demand_capacity.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()

            demand = torch.Tensor(dictionary['demand'])
            capacity = torch.Tensor(dictionary['capacity'])

            if self.target == 'demand':
                y = demand.unsqueeze(dim = 1)
            elif self.target == 'capacity':
                y = capacity.unsqueeze(dim = 1)
            elif self.target == 'congestion':
                congestion = demand - capacity
                y = congestion.unsqueeze(dim = 1)
            else:
                print('Unknown learning target')
                assert False

            # Read connection
            file_name = data_dir + '/' + str(sample) + '.bipartite.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()

            row = dictionary['instance_idx']#[select_indices]
            col = dictionary['net_idx']#[select_indices]
            edge_dir = dictionary['edge_dir']#[select_indices]
            data = torch.ones(len(row))

            i = np.array([row, col])
        
            net_inst_adj = torch.sparse_coo_tensor(i, data).t()
            
            v_drive_idx = [idx for idx in range(len(row)) if edge_dir[idx] == 1]
            v_sink_idx = [idx for idx in range(len(row)) if edge_dir[idx] == 0] 
            
            inst_net_adj_v_drive = torch.sparse_coo_tensor(i.T[v_drive_idx].T, data[v_drive_idx])
            
            inst_net_adj_v_sink = torch.sparse_coo_tensor(i.T[v_sink_idx].T, data[v_sink_idx])
            
            x = instance_features

            # PyG data
            
            example = Data()
            example.__num_nodes__ = x.size(0)
            example.x = x
            example.y = y
            example.net_inst_adj = net_inst_adj
            example.inst_net_adj_v_drive = inst_net_adj_v_drive
            example.inst_net_adj_v_sink = inst_net_adj_v_sink

            if vn:
                file_name = data_dir + '/' + str(sample) + '.single_star_part_dict.pkl'
                f = open(file_name, 'rb')

                part_dict = pickle.load(f)
                f.close()


                file_name = data_dir + '/' + str(sample) + '.star_top_part_dict.pkl'
                f = open(file_name, 'rb')

                part_id_to_top = pickle.load(f)
                f.close()

                edge_index_local_vn = []
                edge_index_vn_top = []
                part_id_lst = list(part_dict.values())

                for local_idx in range(len(example.x)):
                    vn_idx = part_dict[local_idx]
                    edge_index_local_vn.append([local_idx, vn_idx])
                

                top_part_id_lst = []
                for part_idx, top_idx in part_id_to_top.items():
                    top_part_id_lst.append(top_idx)
                    
                    edge_index_vn_top.append([part_idx, top_idx])
            
                example.num_vn = len(np.unique(part_id_lst))
                example.num_top_vn = len(np.unique(top_part_id_lst))
                
                print(example.num_vn, example.num_top_vn)
                
                example.part_id = torch.Tensor(edge_index_local_vn).long().t()[1]
                #example.edge_index_local_vn = torch.Tensor(edge_index_local_vn).long().t()
                #example.edge_index_vn_top = torch.Tensor(edge_index_vn_top).long().t()
            
            capacity = capacity.unsqueeze(dim = 1)
            norm_cap = (capacity - torch.min(capacity)) / (torch.max(capacity) - torch.min(capacity))
            capacity_features = torch.cat([capacity, torch.sqrt(capacity), norm_cap, torch.sqrt(norm_cap), torch.square(norm_cap), torch.sin(norm_cap), torch.cos(norm_cap)], dim = 1)
            
            # Load positional encoding
            if self.load_pe == True:
                file_name = data_dir + '/' + str(sample) + '.eigen.' + str(self.num_eigen) + '.pkl'
                f = open(file_name, 'rb')
                dictionary = pickle.load(f)
                f.close()

                example.evects = torch.Tensor(dictionary['evects'])
                example.evals = torch.Tensor(dictionary['evals'])

            # Load global information
            if self.load_global_info == True:
                file_name = data_dir + '/' + str(sample) + '.global_information.pkl'
                f = open(file_name, 'rb')
                dictionary = pickle.load(f)
                f.close()

                core_util = dictionary['core_utilization']
                global_info = torch.Tensor(np.array([core_util, np.sqrt(core_util), core_util ** 2, np.cos(core_util), np.sin(core_util)]))
                num_nodes = example.x.size(0)
                global_info = torch.cat([global_info.unsqueeze(dim = 0) for i in range(num_nodes)], dim = 0)

                example.x = torch.cat([example.x, global_info], dim = 1)

            # Load persistence diagram and neighbor list
            if self.load_pd == True:
                file_name = data_dir + '/' + str(sample) + '.node_neighbor_features.pkl'
                f = open(file_name, 'rb')
                dictionary = pickle.load(f)
                f.close()

                pd = torch.Tensor(dictionary['pd'])
                neighbor_list = torch.Tensor(dictionary['neighbor'])

                assert pd.size(0) == num_instances
                assert neighbor_list.size(0) == num_instances

                example.x = torch.cat([example.x, pd, neighbor_list], dim = 1)
            
            example.x_net = capacity_features
            
            self.data.append(example)

        print('Done reading data')

    def len(self):
        return self.num_samples

    def get(self, idx):
        return self.data[idx]

