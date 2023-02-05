import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from graph_conv import GNN_node, GNN_node_Virtualnode, GINConvWithoutBondEncoder

from torch_scatter import scatter_mean
# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean",
                    
                    aggregators = ['mean', 'min', 'max', 'std'], # For PNA
                    scalers = ['identity', 'amplification', 'attenuation'], # For PNA
                    deg = None, # For PNA
                    edge_dim = None, # For PNA

                    use_signnet = False, # For SignNet position encoding
                    node_dim = None, # For SignNet position encoding
                    cfg_posenc = None, # For SignNet position encoding
                
                    device = 'cuda'
                ):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, 
                    aggregators = aggregators, scalers = scalers, deg = deg, edge_dim = edge_dim, 
                    use_signnet = use_signnet, node_dim = node_dim, cfg_posenc = cfg_posenc,
                    device = device).to(device = device)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type,
                    aggregators = aggregators, scalers = scalers, deg = deg, edge_dim = edge_dim, 
                    use_signnet = use_signnet, node_dim = node_dim, cfg_posenc = cfg_posenc,
                    device = device).to(device = device)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), 
                                        torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)

'''
class GNN_TopKPooling(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio):
        super().__init__()
        self.gnns =  nn.ModuleList()
        self.norms = nn.ModuleList()
        self.pools = nn.ModuleList()

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        self.num_layer = num_layer
        for i in range(self.num_layer):
            self.gnns.append(GINConvWithoutBondEncoder(emb_dim))
            if i < self.num_layer -1:
                self.norms.append(nn.BatchNorm1d(emb_dim))
                self.pools.append(pyg_nn.TopKPooling(emb_dim))

        self.drop_ratio = drop_ratio
        self.fc = nn.Linear(emb_dim, 1)
    

    def forward(self, batch_data):
        x, edge_index, edge_attr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch 

        h = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        for i in range(self.num_layer):
            h = self.gnns[i](h, edge_index, edge_attr)
            if i < self.num_layer - 1:
                h, edge_index, edge_attr, batch, _, _ = self.pools[i](h, edge_index, edge_attr, batch)
                #print(h.shape)
                #print(edge_index.shape)
                #print(edge_attr.shape)
                h = self.norms[i](h)
                h = h.relu()
                h = F.dropout(h, p =self.drop_ratio, training = self.training)
            else:
                h = F.dropout(h, p = self.drop_ratio, training = self.training)

        h_g = pyg_nn.global_mean_pool(h, batch)
        return self.fc(h_g)
'''


if __name__ == '__main__':
    GNN(num_tasks = 10)
