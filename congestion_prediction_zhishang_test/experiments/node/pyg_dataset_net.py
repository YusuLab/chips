import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data

import numpy as np
import pickle

class pyg_dataset(Dataset):
    def __init__(self, data_dir, graph_index, target, load_pe = False, num_eigen = 5, load_global_info = True, load_pd = False, vn = False, concat=False):
        super().__init__()
        self.data_dir = data_dir
        self.graph_index = graph_index
        self.target = target
        assert target == 'demand' or target == 'capacity' or target == 'congestion'
        print('Learning target:', self.target)

        # Position encoding
        self.load_pe = load_pe
        self.num_eigen = num_eigen
        self.load_global_info = load_global_info
        self.load_pd = load_pd

        # Split
        file_name = str(graph_index) + '.split_net.pkl'
        f = open(file_name, 'rb')
        dictionary = pickle.load(f)
        f.close()

        self.train_indices = dictionary['train_indices']
        self.valid_indices = dictionary['valid_indices_inst']
        self.test_indices = dictionary['test_indices_inst']

        # Read node features
        file_name = data_dir + '/' + str(graph_index) + '.node_features.pkl'
        f = open(file_name, 'rb')
        dictionary = pickle.load(f)
        f.close()

        self.design_name = dictionary['design']

        num_instances = dictionary['num_instances']
        num_nets = dictionary['num_nets']
        instance_features = torch.Tensor(dictionary['instance_features'])
        instance_features = instance_features[:, 2:]
        net_features = torch.zeros(num_nets, instance_features.size(1))
        
        # Read learning targets
        file_name = data_dir + '/' + str(graph_index) + '.net_demand_capacity.pkl'
        f = open(file_name, 'rb')
        dictionary = pickle.load(f)
        f.close()

        demand = torch.Tensor(dictionary['demand'])
        capacity = torch.Tensor(dictionary['capacity'])

        if self.target == 'demand':
            y = demand.unsqueeze(dim = 1)#torch.sum(demand, dim = 1).unsqueeze(dim = 1)
        elif self.target == 'capacity':
            y = capacity.unsqueeze(dim = 1)#torch.sum(capacity, dim = 1).unsqueeze(dim = 1)
        elif self.target == 'congestion':
            congestion = demand - capacity
            y = congestion.unsqueeze(dim = 1)#torch.sum(congestion, dim = 1).unsqueeze(dim = 1)
        else:
            print('Unknown learning target')
            assert False

        # Read connection
        file_name = data_dir + '/' + str(graph_index) + '.bipartite.pkl'
        f = open(file_name, 'rb')
        dictionary = pickle.load(f)
        f.close()

        instance_idx = torch.Tensor(dictionary['instance_idx']).unsqueeze(dim = 1).long()
        net_idx = torch.Tensor(dictionary['net_idx']) + num_instances
        net_idx = net_idx.unsqueeze(dim = 1).long()

        edge_attr = torch.Tensor(dictionary['edge_attr']).float().unsqueeze(dim = 1).float()
        edge_index = torch.cat((instance_idx, net_idx), dim = 1)
        edge_index = torch.transpose(edge_index, 0, 1)
        x = instance_features

        # PyG data
        
        example = Data()
        example.__num_nodes__ = x.size(0)
        example.x = x

        capacity = capacity.unsqueeze(dim = 1)
        norm_cap = (capacity - torch.min(capacity)) / (torch.max(capacity) - torch.min(capacity))
        capacity_features = torch.cat([capacity, torch.sqrt(capacity), norm_cap, torch.sqrt(norm_cap), torch.square(norm_cap), torch.sin(norm_cap), torch.cos(norm_cap)], dim = 1)

        example.x_net = capacity_features
        example.num_instances = num_instances

        example.y = y
        example.edge_index_node_net = edge_index
        example.edge_index_net_node = edge_index.flip([0])

        with open(f"../../data/2023-03-06_data/{graph_index}.nn_conn.pkl", "rb") as f:
                conn_dict = pickle.load(f)

        node_node_conn = conn_dict["nn_edge_index"]
        example.edge_index_node_node = node_node_conn
        example.edge_attr = edge_attr[:2]

        if vn:
            file_name = data_dir + '/' + str(graph_index) + '.star_part_dict.pkl'
            f = open(file_name, 'rb')
            part_dict = pickle.load(f)
            f.close()

            part_id_lst = []

            for idx in range(len(example.x)):
                part_id_lst.append(part_dict[idx])

            part_id = torch.LongTensor(part_id_lst)

            example.part_id = part_id

            example.num_vn = len(torch.unique(part_id))

        # Load positional encoding
        if self.load_pe == True:
            file_name = data_dir + '/' + str(graph_index) + '.eigen.' + str(self.num_eigen) + '.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()

            example.evects = torch.Tensor(dictionary['evects'])
            example.evals = torch.Tensor(dictionary['evals'])

        # Load global information
        if self.load_global_info == True:
            file_name = data_dir + '/' + str(graph_index) + '.global_information.pkl'
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
            file_name = data_dir + '/' + str(graph_index) + '.node_neighbor_features.pkl'
            f = open(file_name, 'rb')
            dictionary = pickle.load(f)
            f.close()

            pd = torch.Tensor(dictionary['pd'])
            neighbor_list = torch.Tensor(dictionary['neighbor'])

            assert pd.size(0) == num_instances
            assert neighbor_list.size(0) == num_instances

            example.x = torch.cat([example.x, pd, neighbor_list], dim = 1)

        
        
        if concat:
            node_feat = example.x
            net_feat = example.x_net
            fill_node_feat = torch.cat([node_feat, torch.zeros(node_feat.size(0), net_feat.size(1))], dim=1)
            fill_net_feat = torch.cat([torch.zeros(net_feat.size(0), node_feat.size(1)), net_feat], dim=1)
            node_feat = torch.cat([fill_node_feat, fill_net_feat], dim=0)
            example.x = node_feat
            example.x_net = None
          
        self.example = example

        print(example, graph_index)

    def len(self):
        return 1

    def get(self, idx):
        return self.example

