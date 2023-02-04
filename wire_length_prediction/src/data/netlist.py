import torch
import numpy as np
from torch_geometric.data import Data
import os
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from torch_geometric.data import Data, InMemoryDataset, download_url
import networkx as nx
import pandas as pd
import pickle
from persim import PersImage
from persim import PersistenceImager

class netlist(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['netlist.npy']

    @property
    def processed_file_names(self):
        return ['netlist.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        data_list= []
        design = "aes"
        
        X_global = np.load("data/X_global.npy")
        Y_global = np.load("data/Y_global.npy")
        df_full = pd.read_csv(f"data/full_data.csv")
        print(len(df_full))
        for index in range(len(df_full)):
            print(index)
            path = df_full['Variant'][index][:-3] + ".npy"
            file = path.split(".")[0]
            edge_lst = np.load(f"/home/zluo/data/graphs/{design}/{path}")
            graph = nx.DiGraph(edge_lst.T.tolist())
            nodelist = list(graph.nodes())
            nodedict = {nodelist[idx]:idx for idx in range(len(nodelist))}
            
            with open(f'/home/zluo/data/output/{design}/type_dict_node_{file}inst.pkl', 'rb') as f:
                type_dict_node = pickle.load(f)
            
            for node in nodelist:
                try:
                    t = type_dict_node[node]
                    if t == 'comb':
                        t = 1
                    else:
                        t = 0
                except:
                    t = 0
                
                type_dict_node[node] = t
                
            X = torch.tensor([[graph.in_degree()[node], graph.out_degree()[node], type_dict_node[node]] for node in nodelist], dtype=torch.float)
            y = Y_global[index]
            edge_index = torch.tensor([[nodedict[tp[0]], nodedict[tp[1]]] for tp in list(graph.edges())], dtype=torch.long).T
            gp_data = Data(x=X, y=y, edge_index=edge_index, stats=torch.tensor(X_global[index], dtype=torch.float))
            data_list.append(gp_data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class netlist_pd(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['netlist.npy']

    @property
    def processed_file_names(self):
        return ['netlist.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        data_list= []
        design = "aes"
        
        X_global = np.load("data/X_global.npy")
        Y_global = np.load("data/Y_global.npy")
        df_full = pd.read_csv(f"data/full_data.csv")
        
        
        with open("data/node_local_pd_dict.pkl", "rb") as f:
            node_local_pd_dict = pickle.load(f)
        
        file_lst = os.listdir(f"/home/zluo/data/graphs/{design}")
    
        print(len(df_full))
        for index in range(len(df_full)):
            print(index)
            path = df_full['Variant'][index][:-3] + ".npy"
            file = path.split(".")[0]
            edge_lst = np.load(f"/home/zluo/data/graphs/{design}/{path}")
            graph = nx.DiGraph(edge_lst.T.tolist())
            e = []
            for node in graph.nodes:
                if graph.degree(node) >= 10:
                    e.append(node)

            graph.remove_nodes_from(e)
            nodelist = list(graph.nodes())
            nodedict = {nodelist[idx]:idx for idx in range(len(nodelist))}
            
            
            with open(f'/home/zluo/data/output/{design}/type_dict_node_{file}inst.pkl', 'rb') as f:
                type_dict_node = pickle.load(f)
                
            L_0 = np.load(f"/home/zluo/data/processed/aes/local_pd/L_0_{file}.npy")
            L_1 = np.load(f"/home/zluo/data/processed/aes/local_pd/L_1_{file}.npy")
            X = []
            
            for node in nodelist:
                x = []
                try:
                    t = type_dict_node[node]
                    
                    if t == 'comb':
                        t = 0
                    else:
                        t = 1
                        
                except:
                    t = 0
                    
                try:
                    pd_0 = L_0[node_local_pd_dict[f"{file}_{node}"]]
                    pd_1 = L_1[node_local_pd_dict[f"{file}_{node}"]]
                except:
                    pd_0 = np.zeros(L_0.shape[-1])
                    pd_1 = np.zeros(L_1.shape[-1])
                
                x = [graph.in_degree()[node], graph.out_degree()[node], t] + list(pd_0) + list(pd_1)
                X.append(x)
           
            X = torch.tensor(X, dtype=torch.float)
            print(X.shape)
            print(X_global[index].shape)
            y = Y_global[index]
            edge_index = torch.tensor([[nodedict[tp[0]], nodedict[tp[1]]] for tp in list(graph.edges())], dtype=torch.long).T
            gp_data = Data(x=X, y=y, edge_index=edge_index, stats=torch.tensor(X_global[index], dtype=torch.float))
            data_list.append(gp_data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class netlist_gb(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['netlist.npy']

    @property
    def processed_file_names(self):
        return ['netlist.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        data_list= []
        design = "aes"
        
        X_global = np.load("data/X_global.npy")
        Y_global = np.load("data/Y_global.npy")
        df_full = pd.read_csv(f"data/full_data.csv")
        print(len(df_full))
        for index in range(len(df_full)):
            print(index)
            path = df_full['Variant'][index][:-3] + ".npy"
            file = path.split(".")[0]
            edge_lst = np.load(f"/home/zluo/data/graphs/{design}/{path}")
            graph = nx.DiGraph(edge_lst.T.tolist())
            nodelist = list(graph.nodes())
            nodedict = {nodelist[idx]:idx for idx in range(len(nodelist))}
            
            with open(f'/home/zluo/data/output/{design}/type_dict_node_{file}inst.pkl', 'rb') as f:
                type_dict_node = pickle.load(f)
            
            for node in nodelist:
                try:
                    t = type_dict_node[node]
                    if t == 'comb':
                        t = 1
                    else:
                        t = 0
                except:
                    t = 0
                
                type_dict_node[node] = t
                
            X = torch.tensor([[graph.in_degree()[node], graph.out_degree()[node], type_dict_node[node]] + list(X_global[index]) for node in nodelist], dtype=torch.float)
            y = Y_global[index]
            edge_index = torch.tensor([[nodedict[tp[0]], nodedict[tp[1]]] for tp in list(graph.edges())], dtype=torch.long).T
            gp_data = Data(x=X, y=y, edge_index=edge_index)
            data_list.append(gp_data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
