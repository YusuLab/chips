import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch_geometric.utils import degree
import pickle
from torch import optim

# PyTorch data loader
# from torch.utils.data import DataLoader

# PyTorch geometric data loader
from torch_geometric.loader import DataLoader
from pyg_dataset import pyg_dataset

import numpy as np
import os
import time
import argparse
import scipy
from tqdm import tqdm

# Model
import sys
sys.path.insert(1, '../../models/')
from gnn_hetero import GNN # Heterogeneous GNN

# Create a configuration for position encodings (including SignNet)
from yacs.config import CfgNode as CN
from posenc_config import set_cfg_posenc

# For Laplacian position encoding
from scipy.sparse import csgraph

# Fix number of threads
torch.set_num_threads(4)

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Supervised learning')
    parser.add_argument('--dir', '-dir', type = str, default = '.', help = 'Directory to save results')
    parser.add_argument('--data_dir', '-data_dir', type = str, default = '.', help = 'Directory that contains all the raw datasets')
    parser.add_argument('--name', '-name', type = str, default = 'NAME', help = 'Name')
    parser.add_argument('--num_epoch', '-num_epoch', type = int, default = 2048, help = 'Number of epochs')
    parser.add_argument('--batch_size', '-batch_size', type = int, default = 20, help = 'Batch size')
    parser.add_argument('--learning_rate', '-learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
    parser.add_argument('--seed', '-s', type = int, default = 123456789, help = 'Random seed')
    parser.add_argument('--n_layers', '-n_layers', type = int, default = 3, help = 'Number of layers of message passing')
    parser.add_argument('--hidden_dim', '-hidden_dim', type = int, default = 32, help = 'Hidden dimension')
    parser.add_argument('--test_mode', '-test_mode', type = int, default = 0, help = 'Test mode')
    parser.add_argument('--pe', '-pe', type = str, default = 'none', help = 'Position encoding')
    parser.add_argument('--pos_dim', '-pos_dim', type = int, default = 0, help = 'Dimension of position encoding')
    parser.add_argument('--virtual_node', '-virtual_node', type = int, default = 0, help = 'Virtual node')
    parser.add_argument('--gnn_type', '-gnn_type', type = str, default = 'gin', help = 'GNN type')
    parser.add_argument('--fold', '-fold', type = int, default = 0, help = 'Fold index in cross-validation')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
    args = parser.parse_args()
    return args

args = _parse_args()
if args.test_mode == 0:
    log_name = args.dir + "/" + args.name + ".log"
else:
    print("Test mode")
    log_name = args.dir + "/" + args.name + ".test_mode.log"
model_name = args.dir + "/" + args.name + ".model"
LOG = open(log_name, "w")

# Fix CPU torch random seed
torch.manual_seed(args.seed)

# Fix GPU torch random seed
torch.cuda.manual_seed(args.seed)

# Fix the Numpy random seed
np.random.seed(args.seed)

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = args.device
print(device)

# Dataset
print('Create data loaders')
pe = args.pe
pos_dim = args.pos_dim

# For SignNet position encoding
config = None
use_signnet = False
if pe == 'signnet':
    use_signnet = True
    config = CN()
    config = set_cfg_posenc(config)
    config.posenc_SignNet.model = 'DeepSet'
    config.posenc_SignNet.post_layers = 2
    config.posenc_SignNet.dim_pe = pos_dim

# Dataset
print(args.data_dir)
train_dataset = pyg_dataset(data_dir = args.data_dir, fold_index = args.fold, split = 'train')
test_dataset = pyg_dataset(data_dir = args.data_dir, fold_index = args.fold, split = 'test')

# Data loaders
batch_size = args.batch_size
train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
valid_dataloader = DataLoader(test_dataset, batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle = False)

print('Number of training examples:', len(train_dataset))
print('Number of testing examples:', len(test_dataset))

for batch_idx, data in enumerate(train_dataloader):
    print(batch_idx)
    print(data)
    node_dim = data.x.size(1)
    edge_dim = data.edge_attr.size(1)
    num_outputs = data.y.size(1)
    break

for batch_idx, data in enumerate(test_dataloader):
    print(batch_idx)
    print(data)
    assert node_dim == data.x.size(1)
    assert edge_dim == data.edge_attr.size(1)
    assert num_outputs == data.y.size(1)
    break

print('Number of node features:', node_dim)
print('Number of edge features:', edge_dim)
print('Number of outputs:', num_outputs)

# Statistics
y = []
for batch_idx, data in enumerate(train_dataloader):
    y.append(data.y.detach().numpy())
y = np.concatenate(y)

y_min = np.min(y)
y_max = np.max(y)
y_mean = np.mean(y)
y_std = np.std(y)

print('y min:', y_min)
print('y max:', y_max)
print('y mean:', y_mean)
print('y std:', y_std)

# Init model and optimizer
if args.virtual_node == 1:
    virtual_node = True
else:
    virtual_node = False
gnn_type = args.gnn_type

print('GNN type:', gnn_type)
print('Virtual node:', virtual_node)

if gnn_type == 'pna':
    aggregators = ['mean', 'min', 'max', 'std']
    scalers = ['identity', 'amplification', 'attenuation']

    print('Computing the in-degree histogram')
    deg = torch.zeros(10, dtype = torch.long)
    for batch_idx, data in enumerate(train_dataloader):
        d = degree(data.edge_index[1], num_nodes = data.num_nodes, dtype = torch.long)
        deg += torch.bincount(d, minlength = deg.numel())
    print('Done computing the in-degree histogram')

    model = GNN(gnn_type = gnn_type, num_tasks = num_outputs, virtual_node = virtual_node, num_layer = args.n_layers, emb_dim = args.hidden_dim,
            aggregators = aggregators, scalers = scalers, deg = deg, edge_dim = edge_dim, 
            use_signnet = use_signnet, node_dim = node_dim, cfg_posenc = config,
            device = device).to(device)
else:
    model = GNN(gnn_type = gnn_type, num_tasks = num_outputs, virtual_node = virtual_node, num_layer = args.n_layers, emb_dim = args.hidden_dim,
            use_signnet = use_signnet, node_dim = node_dim, edge_dim = edge_dim, cfg_posenc = config,
            device = device).to(device)

optimizer = Adagrad(model.parameters(), lr = args.learning_rate)

num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
print('Number of learnable parameters:', num_parameters)
LOG.write('Number of learnable parameters: ' + str(num_parameters) + '\n')
print('Done with model creation')

# Test mode
if args.test_mode == 1:
    print("Skip the training")
    num_epoch = 0
else:
    num_epoch = args.num_epoch

# Train model
best_mae = 1e9
for epoch in range(num_epoch):
    print('--------------------------------------')
    print('Epoch', epoch)
    LOG.write('--------------------------------------\n')
    LOG.write('Epoch ' + str(epoch) + '\n')

    # Training
    t = time.time()
    total_loss = 0.0
    nBatch = 0
    sum_error = 0.0
    num_samples = 0

    for batch_idx, data in enumerate(train_dataloader):
        data = data.to(device = device)
        target = (data.y - y_mean) / y_std

        if use_signnet == True:
            data.x = data.x.type(torch.FloatTensor).to(device = device)

        if gnn_type == 'pna':
            data.edge_attr = data.edge_attr.type(torch.FloatTensor).to(device = device)

        predict = model(data)
        predict = predict[: target.size(0), :]

        optimizer.zero_grad()

        # Mean squared error loss
        loss = F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')
 
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        nBatch += 1

        sum_error += torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
        num_samples += predict.size(0)

        if batch_idx % 10 == 0:
            print('Batch', batch_idx, '/', len(train_dataloader),': Loss =', loss.item())
            LOG.write('Batch ' + str(batch_idx) + '/' + str(len(train_dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

    train_mae = sum_error / (num_samples * num_outputs)
    avg_loss = total_loss / nBatch

    print('Train average loss:', avg_loss)
    LOG.write('Train average loss: ' + str(avg_loss) + '\n')
    print('Train MAE:', train_mae)
    LOG.write('Train MAE: ' + str(train_mae) + '\n')
    print('Train MAE (original scale):', train_mae * y_std)
    LOG.write('Train MAE (original scale): ' + str(train_mae * y_std) + '\n')
    print("Train time =", "{:.5f}".format(time.time() - t))
    LOG.write("Train time = " + "{:.5f}".format(time.time() - t) + "\n")

    # Validation
    t = time.time()
    model.eval()
    total_loss = 0.0
    nBatch = 0

    with torch.no_grad():
        sum_error = 0.0
        num_samples = 0
        for batch_idx, data in enumerate(valid_dataloader):
            data = data.to(device = device)
            target = (data.y - y_mean) / y_std

            if use_signnet == True:
                data.x = data.x.type(torch.FloatTensor).to(device = device)

            if gnn_type == 'pna':
                data.edge_attr = data.edge_attr.type(torch.FloatTensor).to(device = device)

            predict = model(data)
            predict = predict[: target.size(0), :]

            # Mean squared error loss
            loss = F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')

            total_loss += loss.item()
            nBatch += 1

            sum_error += torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
            num_samples += predict.size(0)
             
            if batch_idx % 10 == 0:
                print('Valid Batch', batch_idx, '/', len(valid_dataloader),': Loss =', loss.item())
                LOG.write('Valid Batch ' + str(batch_idx) + '/' + str(len(valid_dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

    valid_mae = sum_error / (num_samples * num_outputs)
    avg_loss = total_loss / nBatch

    print('Valid average loss:', avg_loss)
    LOG.write('Valid average loss: ' + str(avg_loss) + '\n')
    print('Valid MAE:', valid_mae)
    LOG.write('Valid MAE: ' + str(valid_mae) + '\n')
    print('Valid MAE (original scale):', valid_mae * y_std)
    LOG.write('Valid MAE (original scale): ' + str(valid_mae * y_std) + '\n')
    print("Valid time =", "{:.5f}".format(time.time() - t))
    LOG.write("Valid time = " + "{:.5f}".format(time.time() - t) + "\n")
    
    if valid_mae < best_mae:
        best_mae = valid_mae
        print('Current best MAE updated:', best_mae)
        LOG.write('Current best MAE updated: ' + str(best_mae) + '\n')
        print('Current best MAE (original scale) updated:', best_mae * y_std)
        LOG.write('Current best MAE (original scale) updated: ' + str(best_mae * y_std) + '\n')
        
        torch.save(model.state_dict(), model_name)
        print("Save the best model to " + model_name)
        LOG.write("Save the best model to " + model_name + "\n")
    else:
        # Early stopping
        # break
        pass

if args.test_mode == 0:
    print('--------------------------------------')
    LOG.write('--------------------------------------\n')
    print('Best valid MAE:', best_mae)
    LOG.write('Best valid MAE: ' + str(best_mae) + '\n')
    print('Best valid MAE (original scale):', best_mae * y_std)
    LOG.write('Best valid MAE (original scale): ' + str(best_mae * y_std) + '\n')

# Load the model with the best validation
print("Load the trained model at", model_name)
model.load_state_dict(torch.load(model_name))

# Testing
t = time.time()
model.eval()
total_loss = 0.0
nBatch = 0

with torch.no_grad():
    sum_error = 0.0
    num_samples = 0
    for batch_idx, data in enumerate(test_dataloader):
        data = data.to(device = device)
        target = (data.y - y_mean) / y_std

        if use_signnet == True:
            data.x = data.x.type(torch.FloatTensor).to(device = device)

        if gnn_type == 'pna':
            data.edge_attr = data.edge_attr.type(torch.FloatTensor).to(device = device)

        predict = model(data)
        predict = predict[: target.size(0), :]
        
        # Mean squared error loss
        loss = F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')

        total_loss += loss.item()
        nBatch += 1

        sum_error += torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
        num_samples += predict.size(0)

        if batch_idx % 10 == 0:
            print('Test Batch', batch_idx, '/', len(test_dataloader),': Loss =', loss.item())
            LOG.write('Test Batch ' + str(batch_idx) + '/' + str(len(test_dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

test_mae = sum_error / (num_samples * num_outputs)
avg_loss = total_loss / nBatch

print('--------------------------------------')
LOG.write('--------------------------------------\n')
print('Test average loss:', avg_loss)
LOG.write('Test average loss: ' + str(avg_loss) + '\n')
print('Test MAE:', test_mae)
LOG.write('Test MAE: ' + str(test_mae) + '\n')
print('Test MAE (original scale):', test_mae * y_std)
LOG.write('Test MAE (original scale): ' + str(test_mae * y_std) + '\n')
print("Test time =", "{:.5f}".format(time.time() - t))
LOG.write("Test time = " + "{:.5f}".format(time.time() - t) + "\n")

LOG.close()
