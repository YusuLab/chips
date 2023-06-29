import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
from torch_geometric.nn.conv.pna_conv import PNAConv
from torch.nn import Sequential as Seq, Linear, ReLU

import math

# Precompute the statistics for position encodings
from posenc_stats import compute_posenc_stats

# Use SignNet encoder
from signnet_pos_encoder import SignNetNodeEncoder

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim, edge_dim):
        '''
            emb_dim (int): node embedding dimensionality
            edge_dim (int): input edge dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), 
                                        torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, emb_dim), nn.ReLU())

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, edge_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        #self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, emb_dim), nn.ReLU())

    def forward(self, x, edge_index):
        x = self.linear(x)
        #edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out

class HyperConv(MessagePassing):
    def __init__(self, out_channels, edge_dim):
        super(HyperConv, self).__init__(aggr='add')

        self.phi = Seq(Linear(out_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

        self.psi = Seq(Linear(out_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))
                
        self.mlp = Seq(Linear(out_channels * 3, out_channels * 3),
                       ReLU(),
                       Linear(out_channels * 3, out_channels))
        
    def forward(self, x, x_net, net_inst_adj, inst_net_adj_v_drive, inst_net_adj_v_sink):

        net_agg = self.phi(x_net)#self.net_mlp(torch.concat([torch.mm(net_inst_adj, h), x_net], dim=1))

        h_drive = torch.mm(inst_net_adj_v_drive, net_agg)

        h_sink = self.psi(torch.mm(inst_net_adj_v_sink, net_agg))

        h = self.mlp(torch.concat([x, h_drive, h_sink], dim=1))

        net_agg = torch.mm(net_inst_adj, h)

        return h, net_agg 
    

    
### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, JK = "concat", residual = True, gnn_type = 'gin', norm_type = "layer",
                        aggregators = ['mean', 'min', 'max', 'std'], # For PNA
                        scalers = ['identity', 'amplification', 'attenuation'], # For PNA
                        deg = None, # For PNA
                        edge_dim = None, # For PNA

                        use_signnet = False, # For SignNet position encoding
                        node_dim = None, # For SignNet position encoding
                        net_dim = None, 
                        cfg_posenc = None, # For SignNet position encoding

                        num_nodes = None, # Number of nodes

                        device = 'cuda'
                    ):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.device = device

        self.num_layer = num_layer
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        self.node_dim = node_dim
        self.net_dim = net_dim
        self.edge_dim = edge_dim
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
    
        self.gnn_type = gnn_type
    
        self.use_signnet = use_signnet
        self.cfg_posenc = cfg_posenc
        if use_signnet == False:
            self.node_encoder = nn.Sequential(
                    nn.Linear(node_dim, 2 * emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1),
                    nn.Linear(2 * emb_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1)
            )
            
            self.node_encoder_net = nn.Sequential(
                    nn.Linear(net_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1),
                    nn.Linear(emb_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1)
                )
            if self.gnn_type == 'hyper':
                self.node_encoder = nn.Sequential(
                    nn.Linear(node_dim, 2 * emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1),
                    nn.Linear(2 * emb_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1)
                )

                self.node_encoder_net = nn.Sequential(
                    nn.Linear(net_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1),
                    nn.Linear(emb_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1)
                )
        else:
            self.node_encoder = SignNetNodeEncoder(cfg = cfg_posenc, dim_in = node_dim, dim_emb = emb_dim, expand_x = True).to(device = device)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
                
        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        #self.batch_norms = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        #self.net_norms = torch.nn.ModuleList()
        #used to combine MP from both direction
        self.mlps = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, edge_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, edge_dim))
            elif gnn_type == 'pna':
                self.convs.append(PNAConv(in_channels = emb_dim, out_channels = emb_dim, aggregators = aggregators, scalers = scalers, deg = deg, edge_dim = edge_dim))
            elif gnn_type == 'hyper':
                self.convs.append(HyperConv(emb_dim, edge_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))
                    
            #self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            if norm_type == "batch":
                self.norms.append(torch.nn.BatchNorm1d(emb_dim))
                #self.net_norms.append(torch.nn.BatchNorm1d(emb_dim))
            elif norm_type == "layer":
                self.norms.append(torch.nn.LayerNorm(emb_dim))
                #self.net_norms.append(torch.nn.LayerNorm(emb_dim))
            else:
                raise NotImplemented
                
        if gnn_type != 'hyper':
            # Reverse, from target to source
            self.re_convs = torch.nn.ModuleList()

            for layer in range(num_layer):
                if gnn_type == 'gin':
                    self.re_convs.append(GINConv(emb_dim, edge_dim))
                elif gnn_type == 'gcn':
                    self.re_convs.append(GCNConv(emb_dim, edge_dim))
                elif gnn_type == 'pna':
                    self.re_convs.append(PNAConv(in_channels = emb_dim, out_channels = emb_dim, aggregators = aggregators, scalers = scalers, deg = deg, edge_dim = edge_dim))
                else:
                    raise ValueError('Undefined GNN type called {}'.format(gnn_type))

    def forward(self, batched_data):
        
        if self.gnn_type == 'hyper':
            x, x_net, net_inst_adj, inst_net_adj_v_drive, inst_net_adj_v_sink, batch = batched_data.x, batched_data.x_net, batched_data.net_inst_adj, batched_data.inst_net_adj_v_drive, batched_data.inst_net_adj_v_sink, batched_data.batch

        else:
            x, x_net, edge_index_node_net, edge_index_net_node, num_instances, batch = batched_data.x, batched_data.x_net, batched_data.edge_index_node_net, batched_data.edge_index_net_node, batched_data.num_instances,  batched_data.batch 
        #print(edge_index_v_drive.shape, edge_index_v_sink.shape)
        ### computing input node embedding
        if self.use_signnet == False:
            if self.gnn_type == 'hyper':
                x = self.node_encoder(x)
                h_list = [x]
                x_net = self.node_encoder_net(x_net)
                h_net_list = [x_net]

            else:
                x_inst = self.node_encoder(x)
                x_net = self.node_encoder_net(x_net)
                x = torch.cat([x_inst, x_net], dim=0)
                h_list = [x]

        else:
            batched_data = compute_posenc_stats(batched_data, ['SignNet'], True, self.cfg_posenc)
            batched_data.x = batched_data.x.to(device = self.device)
            batched_data.eigvecs_sn = batched_data.eigvecs_sn.to(device = self.device)
            h_list = [self.node_encoder(batched_data).x]

        for layer in range(self.num_layer):
            if self.gnn_type == 'hyper':
                h, h_net = self.convs[layer](h_list[0], h_net_list[layer], net_inst_adj, inst_net_adj_v_drive, inst_net_adj_v_sink)
                h_net = self.norms[layer](h_net)
                h_net = F.leaky_relu(h_net, negative_slope = 0.1)
                
                if self.residual:
                    h_net += h_net_list[layer]
                
            else:
                h = self.convs[layer](h_list[layer], edge_index_node_net)
                h_re = self.re_convs[layer](h, edge_index_net_node)
            
                h = h_re
                                     
            if self.residual:
                h += h_list[0]
                
            h = self.norms[layer](h)
            h = F.leaky_relu(h, negative_slope = 0.1)

            if self.gnn_type == 'hyper':
                h_list = [h]
                h_net_list.append(h_net)    
            
            else:
                h_list.append(h)
        
        ### Different implementations of Jk-concat
        if self.JK == "last":
            if self.gnn_type == 'hyper':
                node_representation = h_net_list[-1]
            else:
                node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                if self.gnn_type == 'hyper':
                    node_representation += h_net_list[layer]
                else:
                    node_representation += h_list[layer]
        elif self.JK == "concat":
            if self.gnn_type == 'hyper':
                node_representation = torch.cat(h_net_list, dim = 1)
            else:
                node_representation = torch.cat(h_list, dim = 1)
        
        if self.gnn_type == 'hyper':
            return node_representation
        else:
            return node_representation[num_instances:]


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, JK = "concat", residual = True, gnn_type = 'gin', norm_type = "layer",
                        aggregators = ['mean', 'min', 'max', 'std'], # For PNA
                        scalers = ['identity', 'amplification', 'attenuation'], # For PNA
                        deg = None, # For PNA
                        edge_dim = None, # For PNA

                        use_signnet = False, # For SignNet position encoding
                        node_dim = None, # For SignNet position encoding
                        cfg_posenc = None, # For SignNet position encoding

                        num_nodes = None, # Number of nodes
                 
                        net_dim = None,

                        device = 'cuda'
                    ):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.device = device

        self.gnn_type = gnn_type
        self.num_layer = num_layer
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim

        self.use_signnet = use_signnet
        self.cfg_posenc = cfg_posenc
        
        if use_signnet == False:
            self.node_encoder = nn.Sequential(
                    nn.Linear(node_dim, 2 * emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1),
                    nn.Linear(2 * emb_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1)
            )
            
            self.node_encoder_net = nn.Sequential(
                    nn.Linear(net_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1),
                    nn.Linear(emb_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1)
                )
            
            if self.gnn_type == 'hyper':
                self.node_encoder = nn.Sequential(
                    nn.Linear(node_dim, 2 * emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1),
                    nn.Linear(2 * emb_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1)
                )

                self.node_encoder_net = nn.Sequential(
                    nn.Linear(net_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1),
                    nn.Linear(emb_dim, emb_dim),
                    nn.LeakyReLU(negative_slope = 0.1)
                )

        else:
            self.node_encoder = SignNetNodeEncoder(cfg = cfg_posenc, dim_in = node_dim, dim_emb = emb_dim, expand_x = True).to(device = device)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        #self.batch_norms = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.mlps = torch.nn.ModuleList()
        
        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, edge_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, edge_dim))
                self.mlps.append(torch.nn.Linear(emb_dim * 2, emb_dim))
            elif gnn_type == 'pna':
                self.convs.append(PNAConv(in_channels = emb_dim, out_channels = emb_dim, aggregators = aggregators, scalers = scalers, deg = deg, edge_dim = edge_dim))
            elif gnn_type == 'hyper':
                self.convs.append(HyperConv(emb_dim, edge_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            #self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            if norm_type == "batch":
                self.norms.append(torch.nn.BatchNorm1d(emb_dim))
            elif norm_type == "layer":
                self.norms.append(torch.nn.LayerNorm(emb_dim))
            else:
                raise NotImplemented
                
                
            if gnn_type != 'hyper':
                # Reverse, from target to source
                self.re_convs = torch.nn.ModuleList()

                for layer in range(num_layer):
                    if gnn_type == 'gin':
                        self.re_convs.append(GINConv(emb_dim, edge_dim))
                    elif gnn_type == 'gcn':
                        self.re_convs.append(GCNConv(emb_dim, edge_dim))
                    elif gnn_type == 'pna':
                        self.re_convs.append(PNAConv(in_channels = emb_dim, out_channels = emb_dim, aggregators = aggregators, scalers = scalers, deg = deg, edge_dim = edge_dim))
                    else:
                        raise ValueError('Undefined GNN type called {}'.format(gnn_type))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(emb_dim, 2 * emb_dim), 
                        torch.nn.LeakyReLU(negative_slope = 0.1),
                        torch.nn.Linear(2 * emb_dim, emb_dim), 
                        torch.nn.LeakyReLU(negative_slope = 0.1)
                    )
            )


    def forward(self, batched_data):
        
        x, x_net, net_inst_adj, inst_net_adj_v_drive, inst_net_adj_v_sink, batch, num_vn = batched_data.x, batched_data.x_net, batched_data.net_inst_adj, batched_data.inst_net_adj_v_drive, batched_data.inst_net_adj_v_sink, batched_data.part_id, batched_data.num_vn
        
        #batch = edge_index_local_vn[1]
        #top_batch = edge_index_vn_top[1]

        ### virtual node embeddings 
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(num_vn).to(batch.dtype).to(batch.device))
    
        ###########################
        
        if self.use_signnet == False:
            if self.gnn_type == 'hyper':
                x = self.node_encoder(x)
                x_net = self.node_encoder_net(x_net)
                h_list = [x]
                h_net_list = [x_net]

            else:
                x_inst = self.node_encoder(x[:num_instances])
                x_net = self.node_encoder_net(x[num_instances:])
                x = torch.cat([x_inst, x_net], dim=0)
                h_list = [x]

        else:
            batched_data = compute_posenc_stats(batched_data, ['SignNet'], True, self.cfg_posenc)
            batched_data.x = batched_data.x.to(device = self.device)
            batched_data.eigvecs_sn = batched_data.eigvecs_sn.to(device = self.device)
            h_list = [self.node_encoder(batched_data).x]
        
        
        
        ##################################
        
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            if self.gnn_type == 'hyper':
                h, h_net = self.convs[layer](h_list[layer], h_net_list[layer], net_inst_adj, inst_net_adj_v_drive, inst_net_adj_v_sink)
                h_net = self.norms[layer](h_net)
                h_net = F.leaky_relu(h_net, negative_slope = 0.1)
                
                if self.residual:
                    h_net += h_net_list[layer]
                
            else:
                h = self.convs[layer](h_list[layer], edge_index)
                h_re = self.re_convs[layer](h_list[layer], edge_index.flip([0]))
            
                h = self.mlps[layer](torch.concat([h, h_re], dim = 1))
                                     
                
            h = self.norms[layer](h)
            h = F.leaky_relu(h, negative_slope = 0.1)
            
            if self.residual:
                h += h_list[layer]

            if self.gnn_type == 'hyper':
                h_list.append(h)
                h_net_list.append(h_net)    
            
            else:
                h_list.append(h)
                
                
            ## update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_mean_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + self.mlp_virtualnode_list[layer](virtualnode_embedding_temp)
                else:
                    virtualnode_embedding = self.mlp_virtualnode_list[layer](virtualnode_embedding_temp)
        
        ### Different implementations of Jk-concat
        if self.JK == "last":
            if self.gnn_type == 'hyper':
                node_representation = h_net_list[-1]
            else:
                node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                if self.gnn_type == 'hyper':
                    node_representation += h_net_list[layer]
                else:
                    node_representation += h_list[layer]
        elif self.JK == "concat":
            if self.gnn_type == 'hyper':
                node_representation = torch.cat(h_net_list, dim = 1)
            else:
                node_representation = torch.cat(h_list, dim = 1)
        
        
        if self.gnn_type == 'hyper':
            return node_representation
        else:
            return node_representation[num_instances:]


if __name__ == "__main__":
    pass
