import torch
from torch import nn
import os
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_add_pool
from torch.nn import BatchNorm1d, ReLU, Linear, Sequential


class MLP(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(MLP, self).__init__()
        self.fc1 = Linear(input_channels, hidden_channels[0])
        self.fc2 = Linear(hidden_channels[0], hidden_channels[1])
        self.fc3 = Linear(hidden_channels[1], hidden_channels[2])
        self.fc4 = Linear(hidden_channels[2], hidden_channels[3])
        self.fc5 = Linear(hidden_channels[3], output_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x



class GAT(torch.nn.Module):
    """
    GAT model for Graph-level tasks
    """
    def __init__(self, num_global_features, num_node_features, hid, in_head, out_head):
        super(GAT, self).__init__()
        self.hid = hid
        self.in_head = in_head
        self.out_head = out_head
        self.num_global_features = num_global_features
        
        self.conv1 = GATConv(num_node_features, self.hid, heads=self.in_head)
        self.conv2 = GATConv(self.hid*self.in_head, self.hid, concat=False,
                             heads=self.in_head)
        self.conv3 = GATConv(self.hid, self.hid, concat=False, heads=self.out_head)
        self.fc1 = nn.Linear(self.hid+self.num_global_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x, edge_index, batch, stats):
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = torch.cat((x, stats.reshape(x.shape[0], self.num_global_features)), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    
class GCN(torch.nn.Module):
    """
    GCN model for Graph-level tasks
    """
    def __init__(self, hidden_channels, num_global_features, num_node_features):
        super(GCN, self).__init__()
        self.num_global_features = num_global_features
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels+self.num_global_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x, edge_index, batch, stats):


        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = torch.cat((x, stats.reshape(x.shape[0], self.num_global_features)), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    
    


    
class GCN_G(torch.nn.Module):
    """
    GCN model for Graph-level tasks
    """
    def __init__(self, hidden_channels, num_node_features):
        super(GCN_G, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x, edge_index, batch):


        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x    
    
    
    
    
class GAT_G(torch.nn.Module):
    """
    GAT model for Graph-level tasks
    """
    def __init__(self, num_node_features, hid, in_head, out_head):
        super(GAT_G, self).__init__()
        self.hid = hid
        self.in_head = in_head
        self.out_head = out_head
        
        self.conv1 = GATConv(num_node_features, self.hid, heads=self.in_head, concat=False)
        self.conv2 = GATConv(self.hid*self.in_head, self.hid, concat=False,
                             heads=self.in_head)
        self.conv3 = GATConv(self.hid, self.hid, concat=False, heads=self.out_head)

        self.fc1 = nn.Linear(self.hid, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x, edge_index, batch):
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    
    
    

    
