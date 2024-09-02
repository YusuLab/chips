import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch_geometric.utils import degree
import pickle
from torch import optim

# For visualization
from utils import *

# PyTorch data loader
# from torch.utils.data import DataLoader

# PyTorch geometric data loader
from torch_geometric.loader import DataLoader
from pyg_dataset_net import pyg_dataset
import numpy as np
import os
import time
import argparse
import scipy
from tqdm import tqdm

# Model
import sys
sys.path.insert(1, '../../models_net/')
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
    parser.add_argument('--target', '-target', type = str, default = 'none', help = 'Learning target')
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
    parser.add_argument('--load_global_info', '-load_global_info', type = int, default = 0, help = 'Global information')
    parser.add_argument('--load_pd', '-load_pd', type = int, default = 0, help = 'Persistence diagram & Neighbor list')
    parser.add_argument('--graph_index', '-graph_index', type = int, default = 0, help = 'Index of the graph')
    parser.add_argument('--device', '-device', type = str, default = 'cpu', help = 'cuda/cpu')
    parser.add_argument('--split', '-split', type = int, default = 1, help = 'Index of the split')
    parser.add_argument('--pl', '-pl', type = int, default = 0, help = 'use placement info or not')
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

if args.gnn_type not in ['gcn', 'gat']:
    sparse = True
else:
    sparse = False

if sparse:
    from pyg_dataset_sparse import *

print("Sparse", sparse, args.gnn_type)
# Dataset
print('Create data loaders')
pe = args.pe
pos_dim = args.pos_dim

#loss_type = args.loss_type
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

load_global_info = False
if args.load_global_info == 1:
    load_global_info = True

load_pd = False
if args.load_pd == 1:
    load_pd = True

if args.virtual_node >= 1:
    virtual_node = True
    single = False
    
    if args.virtual_node == 2:
        single = True
else:
    virtual_node = False
    single = False
    
print("single:", single)

if pe == 'lap':
    dataset = pyg_dataset(data_dir = args.data_dir, graph_index = args.graph_index, target = args.target, load_pe = True, num_eigen = pos_dim, load_global_info = load_global_info, load_pd = load_pd, vn = virtual_node, net=True, split = args.split, pl = args.pl)
else:
    if sparse:
        dataset = pyg_dataset(data_dir = args.data_dir, graph_index = args.graph_index, target = args.target, load_global_info = load_global_info, load_pd = load_pd, vn = virtual_node, net=True, split = args.split)
    else:
        dataset = pyg_dataset(data_dir = args.data_dir, graph_index = args.graph_index, target = args.target, load_global_info = load_global_info, load_pd = load_pd, vn = virtual_node, net=True, split = args.split)

# Data loaders
batch_size = args.batch_size
dataloader = DataLoader(dataset, batch_size, shuffle = False)

print('Number of nodes in the training set:', dataset.train_indices.shape[0])
print('Number of nodes in the validation set:', dataset.valid_indices.shape[0])
print('Number of nodes in the testing set:', dataset.test_indices.shape[0])

for batch_idx, data in enumerate(dataloader):
    print(batch_idx)
    print(data)
    node_dim = data.x.size(1)
    if sparse:
        edge_dim = 1
        net_dim = data.x_net.size(1)
    else:
        edge_dim = data.edge_attr.size(1)
        net_dim = data.x_net.size(1)
    num_outputs = data.y.size(1)

print('Number of node features:', node_dim)
print('Number of edge features:', edge_dim)
print('Number of outputs:', num_outputs)

if pe == 'lap':
    node_dim += pos_dim
    net_dim += pos_dim

    print('Number of eigenvectors:', pos_dim)
    print('Number of node features + position encoding:', node_dim)

# Statistics
y = []
for batch_idx, data in enumerate(dataloader):
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
gnn_type = args.gnn_type

print('GNN type:', gnn_type)
print('Virtual node:', virtual_node)

# with open(f"{args.data_dir}/{args.graph_index}.net_dict.pkl", "rb") as f:
#     net_dict = pickle.load(f)

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
            use_signnet = use_signnet, node_dim = node_dim, net_dim = net_dim,  cfg_posenc = config,
            device = device).to(device)
else:
    model = GNN(gnn_type = gnn_type, num_tasks = num_outputs, virtual_node = virtual_node, num_layer = args.n_layers, emb_dim = args.hidden_dim,
            use_signnet = use_signnet, node_dim = node_dim, net_dim = net_dim, edge_dim = edge_dim, cfg_posenc = config,
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

#if sparse:
    #print("Load the trained model at", model_name)
    #try:
        #model.load_state_dict(torch.load(model_name))
    #except:
        #print("start new model")
        #num_epoch = 2000
# Train model
print(num_epoch)

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

    for batch_idx, data in enumerate(dataloader):
        data = data.to(device = device)
        target = (data.y - y_mean) / y_std
        #weights = data.weights.to(device = device)

        if pe == 'lap':
            data.x = torch.cat([data.x, data.evects[:data.x.shape[0]]], dim = 1)
            data.x_net = torch.cat([data.x_net, data.evects[data.x.shape[0]:]], dim = 1)

        if use_signnet == True:
            data.x = data.x.type(torch.FloatTensor).to(device = device)

        if gnn_type == 'pna':
            data.edge_attr = data.edge_attr.type(torch.FloatTensor).to(device = device)

        # Model
        predict = model(data)

        # Train indices
        predict = predict[dataset.train_indices, :]
        target = target[dataset.train_indices, :]
        #weights = weights[dataset.train_indices]

        optimizer.zero_grad()

        # Mean squared error loss
            
        #mse_loss = weighted_mse(predict.view(-1), target.view(-1), weights)
        loss = F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')
        mse_loss = loss

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        nBatch += 1

        sum_error += torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
        num_samples += predict.size(0)

        if batch_idx % 10 == 0:
            print('Batch', batch_idx, '/', len(dataloader),': Loss =', loss.item())
            LOG.write('Valid Batch ' + str(batch_idx) + '/' + str(len(dataloader)) + ': Loss = ' + str(loss.item()) + ' MSE loss = ' + str(mse_loss.item()) + '\n')

    train_mae = sum_error / (num_samples * num_outputs)
    avg_loss = total_loss / nBatch

    print('Train average loss:', train_mae)
    LOG.write('Train average loss: ' + str(train_mae) + '\n')
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
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device = device)
            target = (data.y - y_mean) / y_std
            #weights = data.weights.to(device = device)

            if pe == 'lap':
                data.x = torch.cat([data.x, data.evects[:data.x.shape[0]]], dim = 1)
                data.x_net = torch.cat([data.x_net, data.evects[data.x.shape[0]:]], dim = 1)

            if use_signnet == True:
                data.x = data.x.type(torch.FloatTensor).to(device = device)

            if gnn_type == 'pna':
                data.edge_attr = data.edge_attr.type(torch.FloatTensor).to(device = device)

            predict = model(data)
            
            # Validation indices
            predict = predict[dataset.valid_indices, :]
            target = target[dataset.valid_indices, :]

            # Mean squared error loss
            #mse_loss = weighted_mse(predict.view(-1), target.view(-1), weights)
            loss = F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')
            mse_loss = loss

            total_loss += loss.item()
            nBatch += 1

            sum_error += torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
            num_samples += predict.size(0)
             
            if batch_idx % 10 == 0:
                print('Valid Batch', batch_idx, '/', len(dataloader),': Loss =', loss.item())
                LOG.write('Valid Batch ' + str(batch_idx) + '/' + str(len(dataloader)) + ': Loss = ' + str(loss.item()) + ' MSE loss = ' + str(mse_loss.item()) + '\n')

    valid_mae = sum_error / (num_samples * num_outputs)
    avg_loss = total_loss / nBatch

    print('Valid average loss:', valid_mae)
    LOG.write('Valid average loss: ' + str(valid_mae) + '\n')
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

y_test = []
y_hat = []

with torch.no_grad():
    sum_error = 0.0
    num_samples = 0
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device = device)
        target = (data.y - y_mean) / y_std

        if pe == 'lap':
            data.x = torch.cat([data.x, data.evects[:data.x.shape[0]]], dim = 1)
            data.x_net = torch.cat([data.x_net, data.evects[data.x.shape[0]:]], dim = 1)

        if use_signnet == True:
            data.x = data.x.type(torch.FloatTensor).to(device = device)

        if gnn_type == 'pna':
            data.edge_attr = data.edge_attr.type(torch.FloatTensor).to(device = device)
        
        print(data)
        # Model
        predict = model(data)
        
        # Test indices
        predict = predict[dataset.test_indices, :]
        target = target[dataset.test_indices, :]
        
        # Mean squared error loss
        loss = F.mse_loss(predict.view(-1), target.view(-1), reduction = 'mean')

        y_test.append(target.view(-1))
        y_hat.append(predict.view(-1))

        total_loss += loss.item()
        nBatch += 1

        sum_error += torch.sum(torch.abs(predict.view(-1) - target.view(-1))).detach().cpu().numpy()
        num_samples += predict.size(0)

        if batch_idx % 10 == 0:
            print('Test Batch', batch_idx, '/', len(dataloader),': Loss =', loss.item())
            LOG.write('Test Batch ' + str(batch_idx) + '/' + str(len(dataloader)) + ': Loss = ' + str(loss.item()) + '\n')

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

# Visualiation
truth = torch.cat(y_test, dim = 0).cpu().detach().numpy() * y_std + y_mean
predict = torch.cat(y_hat, dim = 0).cpu().detach().numpy() * y_std + y_mean

with open(args.dir + "/" + args.name + ".truth.npy", 'wb') as f:
    np.save(f, truth)

with open(args.dir + "/" + args.name + ".predict.npy", 'wb') as f:
    np.save(f, predict)

#method_name = args.gnn_type
#design_name = dataset.design_name
#output_name = args.dir + "/" + args.name + ".png"
#plot_figure(truth, predict, method_name, design_name, output_name)
print('Done')

LOG.close()
