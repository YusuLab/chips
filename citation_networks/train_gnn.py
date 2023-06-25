import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
dataset = 'Cora'
transform = T.Compose([
    T.RandomNodeSplit(num_val = 100, num_test = 100),
    T.TargetIndegree(),
])
dataset = Planetoid(root = './', name = 'Cora', transform = transform)
data = dataset[0]
data = data.to(device)

print(data)

# Model
import sys
sys.path.insert(1, '../congestion_prediction/models/')
from gnn_hetero import GNN # Heterogeneous GNN

# Model creation
model = GNN(
    gnn_type = "gcn", 
    num_tasks = dataset.num_classes, 
    virtual_node = False, 
    num_layer = 3, 
    emb_dim = 32,
    use_signnet = False, 
    node_dim = dataset.num_features, 
    edge_dim = 1, 
    cfg_posenc = None,
    device = device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) #, weight_decay = 5e-3)

def train():
    optimizer.zero_grad()

    prediction = torch.softmax(model(data), dim = 1)

    prediction = prediction[data.train_mask]
    target = data.y[data.train_mask]
    
    loss = F.nll_loss(prediction, target)
    print(loss)
    loss.backward()
    
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    log_probs, accs = model(data), []
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
