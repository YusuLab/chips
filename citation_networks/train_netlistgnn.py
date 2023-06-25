import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

# Deep Graph Library (DGL)
import dgl

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PyTorch Dataset
dataset = 'Cora'
transform = T.Compose([
    T.RandomNodeSplit(num_val = 100, num_test = 100),
    T.TargetIndegree(),
])
dataset = Planetoid(root = './', name = 'Cora', transform = transform)
data = dataset[0]

print(data)

# DGL Dataset
num_instances = data.x.size(0)
num_nets = num_instances

instance_idx = np.array([idx for idx in range(num_instances)])
num_edges = num_instances

dgl_data = dgl.heterograph(
    {
        ('node', 'near', 'node'): (data.edge_index[0, :], data.edge_index[1, :]),
        ('node', 'pins', 'net'): (instance_idx, instance_idx),
        ('net', 'pinned', 'node'): (instance_idx, instance_idx),
    },
    num_nodes_dict = {
        'node': num_instances,
        'net': num_nets
    }
)

dgl_data.nodes['node'].data['hv'] = data.x
dgl_data.nodes['node'].data['pos_code'] = torch.ones(num_instances, 1)
dgl_data.nodes['net'].data['hv'] = torch.ones(num_nets, 1)
dgl_data.nodes['net'].data['degree'] = torch.ones(num_nets, 1)
dgl_data.nodes['net'].data['label'] = torch.ones(num_nets, 1)
dgl_data.edges['pins'].data['he'] = torch.ones(num_edges, 1)
dgl_data.edges['pinned'].data['he'] = torch.ones(num_edges, 1)
dgl_data.edges['near'].data['he'] = data.edge_attr

# Model
import sys
sys.path.insert(1, '../congestion_prediction/models/')
from NetlistGNN import * # NetlistGNN from PKU

# Model creation
config = {
    'N_LAYER': 3,
    'NODE_FEATS': 32,
    'NET_FEATS': 4,
    'PIN_FEATS': 4,
    'EDGE_FEATS': 4,
}

model = NetlistGNN(
    in_node_feats = dataset.num_features,
    in_net_feats = 1,
    in_pin_feats = 1,
    in_edge_feats = 1,
    n_target = dataset.num_classes,
    activation = 'relu',
    config = config,
    recurrent = False,
    topo_conv_type = 'CFCNN',
    geom_conv_type = 'SAGE'
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) #, weight_decay = 5e-3)

data = data.to(device)

def train():
    optimizer.zero_grad()

    # NetlistGNN PKU
    prediction, _ = model.forward(
        in_node_feat = dgl_data.nodes['node'].data['hv'].to(device),
        in_net_feat = dgl_data.nodes['net'].data['hv'].to(device),
        in_pin_feat = dgl_data.edges['pinned'].data['he'].to(device),
        in_edge_feat = dgl_data.edges['near'].data['he'].to(device),
        node_net_graph = dgl_data.to(device),
    )

    # Apply softmax
    prediction = torch.softmax(prediction, dim = 1)

    prediction = prediction[data.train_mask]
    target = data.y[data.train_mask]
    
    loss = F.nll_loss(prediction, target)
    print(loss)
    loss.backward()
    
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    
    # NetlistGNN PKU
    prediction, _ = model.forward(
        in_node_feat = dgl_data.nodes['node'].data['hv'].to(device),
        in_net_feat = dgl_data.nodes['net'].data['hv'].to(device),
        in_pin_feat = dgl_data.edges['pinned'].data['he'].to(device),
        in_edge_feat = dgl_data.edges['near'].data['he'].to(device),
        node_net_graph = dgl_data.to(device),
    )

    # Apply softmax
    log_probs = torch.softmax(prediction, dim = 1)

    accs = []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

print('Done')
