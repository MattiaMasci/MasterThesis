import random
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import datasets, transforms

train_dataloader = torch.load('../data/dataloaders/train_dataloader.pth')
test_dataloader = torch.load('../data/dataloaders/test_dataloader.pth')

somma = 0
for batch, (X, y) in enumerate(train_dataloader):
    somma += len(X)

print(somma)

"""print(len(train_dataloader))
print(len(test_dataloader.dataset))"""

"""
tensor = torch.load('../plot/LSTM/comparisons/time/LRS/time_0.pt')
print(tensor.shape)
print(tensor)

tensor = torch.load('../plot/LSTM/comparisons/time/LES/time_1.pt')
print(tensor.shape)
print(tensor)

tensor = torch.load('../plot/LSTM/comparisons/time/LES/time_2.pt')
print(tensor.shape)
print(tensor)

tensor = torch.load('../plot/LSTM/comparisons/time/LES/time_3.pt')
print(tensor.shape)
print(tensor)

tensor = torch.load('../plot/LSTM/comparisons/time/LES/time_4.pt')
print(tensor.shape)
print(tensor)
"""

"""
tensor = torch.tensor([])

torch.save(tensor, '../plot/LSTM/comparisons/time/time.pt')

torch.save(tensor, '../plot/LSTM/comparisons/time/LES/time_0.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/LES/time_1.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/LES/time_2.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/LES/time_3.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/LES/time_4.pt')

torch.save(tensor, '../plot/LSTM/comparisons/time/LRS/time_0.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/LRS/time_1.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/LRS/time_2.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/LRS/time_3.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/LRS/time_4.pt')

torch.save(tensor, '../plot/LSTM/comparisons/time/SFS/time_0.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/SFS/time_1.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/SFS/time_2.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/SFS/time_3.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/SFS/time_4.pt')

torch.save(tensor, '../plot/LSTM/comparisons/time/RSFS/time_0.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/RSFS/time_1.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/RSFS/time_2.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/RSFS/time_3.pt')
torch.save(tensor, '../plot/LSTM/comparisons/time/RSFS/time_4.pt')
"""

"""
# SAVE INITIALIZATIONS LSTM 4
net = LSTMModel()

for i in range(11,101):
    for name, param in net.lstm1.named_parameters():
        if 'weight_ih' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)
    for name, param in net.lstm2.named_parameters():
        if 'weight_ih' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)
    for name, param in net.lstm3.named_parameters():
        if 'weight_ih' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)
    for name, param in net.lstm4.named_parameters():
        if 'weight_ih' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)

    torch.nn.init.orthogonal_(net.fc1.weight)
    net.fc1.bias.data.fill_(0)

    torch.nn.init.orthogonal_(net.fc2.weight)
    net.fc2.bias.data.fill_(0)

    torch.nn.init.orthogonal_(net.fc3.weight)
    net.fc3.bias.data.fill_(0)

    torch.save(net.state_dict(), f'../models/LSTM/initializations/init{i}.pt')
    """