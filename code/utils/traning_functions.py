import os.path as osp
import numpy as np
import pandas as pd
from scipy.stats import zscore

import torch
import torch.nn as nn
import torch_geometric



def train_process(model, train_loader, optimizer, criterion, device):
    model.train()
    for data in train_loader: 
         data = data.to(device)
         optimizer.zero_grad()
         y_logits,_ = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         y_logits = y_logits.to(device)
         loss = criterion(y_logits.float(), torch.unsqueeze(data.y.float().to(device),1))  # Compute the loss.
         loss.backward()
         optimizer.step()
        

def test_process(model, loader, device):
     model.eval()
     correct = 0
     for data in loader:  
         data = data.to(device)
         y_logits,_ = model(data.x, data.edge_index, data.batch)
         y_probs = torch.sigmoid(y_logits)
         y_pred = (y_probs > 0.5).float()
         y_pred = y_pred.to(device)
         correct += int((y_pred.squeeze() == data.y).sum())
     return correct / len(loader.dataset)




    

    
     
