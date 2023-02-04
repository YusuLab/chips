import torch
from torch import nn
import os
import numpy as np
import torch.nn.functional as F


def experiment(n, train_dataset, val_dataset, model, criterion, optimizer, scheduler, device, eval_crs, gb=False):
    loss_train_all = []
    loss_val_all = []
    
    def train(): 
        model.train()

        loss_train_tp = []
        for data in train_dataset:
            data = data.to(device)
            optimizer.zero_grad()
            if gb:
                output = model(data.x, data.edge_index, data.batch)[0]
            else:
                output = model(data.x, data.edge_index, data.batch, data.stats)[0]
            loss = criterion(output, data.y)
            tp = []
            for cri in eval_crs:
                loss_tr = cri(output, data.y).item()
                tp.append(loss_tr)
            
            loss_train_tp.append(tp)
            loss.backward()
            optimizer.step()
        
        loss_train_tp = np.mean(np.array(loss_train_tp), axis=0)
        
        return loss_train_tp
    
    def test(): 
        model.eval()
        
        loss_val_tp = []
        for data in val_dataset:
            data = data.to(device)
            if gb:
                output = model(data.x, data.edge_index, data.batch)[0]
            else:
                output = model(data.x, data.edge_index, data.batch, data.stats)[0]

            tp = []
            for criterion in eval_crs:
                loss_t = criterion(output, data.y).item()
                tp.append(loss_t)
           
            loss_val_tp.append(tp)
            
        loss_val_tp = np.mean(np.array(loss_val_tp), axis=0)
            
        return loss_val_tp
    
    
    for epoch in range(n):
        loss_train_tp = train()
        loss_val_tp = test()

        #with open("train_log", "w", buffering=1) as f:
        print(f"{epoch}, {loss_train_tp}, {loss_val_tp}")
        
        loss_train_all.append(loss_train_tp)
        loss_val_all.append(loss_val_tp)
        
        #scheduler.step()
        
    return loss_train_all, loss_val_all
