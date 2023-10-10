import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch import nn
import matplotlib.pyplot as plt
import os
from net_definition import Net, Conv2Net
from training_loops import train_loop, test_loop
#from training_loops import train_loop, test_loop
from freezing_methods import normalizedGradientDifferenceFreezingProcedure, gradientNormChangeFreezingProcedure,\
layerInfluenceAnalysis

# Dataset loading
training_data = torch.load('../../data/reduced_training_set.pt')
test_data = torch.load('../../data/reduced_testing_set.pt')

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# Parameters setting
learning_rate = 1e-3
batch_size = 64
loss_fn = nn.CrossEntropyLoss()

# Model loading
net = Conv2Net()

checkpoint = torch.load('../../models/CIFAR-net.pt')
del checkpoint['model_weights']['conv3.weight'], checkpoint['model_weights']['conv3.bias'], \
    checkpoint['model_weights']['conv4.weight'], checkpoint['model_weights']['conv4.bias'], \
    checkpoint['model_weights']['conv5.weight'], checkpoint['model_weights']['conv5.bias'], \
    checkpoint['model_weights']['fc1.weight'], checkpoint['model_weights']['fc1.bias'], \
    checkpoint['model_weights']['fc2.weight'], checkpoint['model_weights']['fc2.bias']

fc1_weights = torch.zeros([100, 2048])
torch.nn.init.xavier_uniform_(fc1_weights)

fc1_bias = torch.zeros(100)
torch.nn.init.normal_(fc1_bias)

fc2_weights = torch.zeros([10, 100])
torch.nn.init.xavier_uniform_(fc2_weights)

fc2_bias = torch.zeros(10)
torch.nn.init.normal_(fc2_bias)

checkpoint['model_weights']['fc1.weight'] = fc1_weights
checkpoint['model_weights']['fc1.bias'] = fc1_bias
checkpoint['model_weights']['fc2.weight'] = fc2_weights
checkpoint['model_weights']['fc2.bias'] = fc2_bias

net.load_state_dict(checkpoint['model_weights'])

# Parameters setting 
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
epochs = 50

# For plot
net_acc_values = torch.zeros([epochs])
net_loss_values = torch.zeros([epochs])
count = 0

# Array for influence analysis
accuracy_analysis_array = torch.zeros([epochs,4])
loss_analysis_array = torch.zeros([epochs,4])

# Training loop
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")

    train_loop(train_dataloader, net, loss_fn, optimizer)
    net_acc_values[count], net_loss_values[count] = test_loop(test_dataloader, net, loss_fn)

    accuracy_temp, loss_temp = layerInfluenceAnalysis(net)
    accuracy_temp[3] = net_acc_values[count]
    loss_temp[3] = net_loss_values[count]
    accuracy_analysis_array[t] = accuracy_temp
    loss_analysis_array[t] = loss_temp

    count = count+1

print("Done!")

# influence Analysis
torch.save(accuracy_analysis_array, '../../plot/data/influenceAnalysis/conv2Net/accuracy50.pt')
torch.save(loss_analysis_array, '../../plot/data/influenceAnalysis/conv2Net/loss50.pt')
