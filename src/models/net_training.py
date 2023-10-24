import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import StepLR
from torch import nn
import matplotlib.pyplot as plt
import os
from net_definition import Net
#from training_loops_additional import train_loop, test_loop
from training_loops import train_loop, test_loop
from freezing_methods import normalizedGradientDifferenceFreezingProcedure, gradientNormChangeFreezingProcedure,\
layerInfluenceAnalysis, layerInfluenceAnalysis2

# Dataset loading
training_data = torch.load('../../data/reduced_training_set.pt')
test_data = torch.load('../../data/reduced_testing_set.pt')

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

"""
training_data = datasets.CIFAR10(
  root="../../data",
  train=True,
  download=True,
  transform = ToTensor()
)

test_data = datasets.CIFAR10(
  root="../../data",
  train=False,
  download=True,
  transform=ToTensor()
)
"""

# Parameters setting
learning_rate = 1e-3
batch_size = 64
loss_fn = nn.CrossEntropyLoss()

# Model loading
net = Net()

checkpoint = torch.load('../../models/CIFAR-net.pt')
net.load_state_dict(checkpoint['model_weights'])

# Parameters setting 
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
epochs = 5

"""
# Learning rate decay
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
"""

# For plot
net_acc_values = torch.zeros([epochs])
net_loss_values = torch.zeros([epochs])
count = 0

"""
# normalizedGradientDifferenceFreezingProcedure
freezing_rate_values = torch.zeros([epochs,7])
freeze = False

# Influence analysis
accuracy_analysis_array = torch.zeros([epochs,7])
loss_analysis_array = torch.zeros([epochs,7])

# gradientNormChangeFreezingProcedure
step = 1
gradient_difference_norm_change_array = torch.zeros([epochs,7,((389//step)+1)])
gradient_norm_difference_change_array = torch.zeros([epochs,7,((389//step)+1)])
"""

# Training loop
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")

    """
    print('PARAMETERS THAT REQUIRE GRADIENT:')
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
    print()
    """

    train_loop(train_dataloader, net, loss_fn, optimizer)
    net_acc_values[count], net_loss_values[count] = test_loop(test_dataloader, net, loss_fn)

    """
    # Learning rate decay
    scheduler.step()
    """

    # influence Analysis2
    layerInfluenceAnalysis2(net)

    """
    # influence Analysis
    accuracy_temp, loss_temp = layerInfluenceAnalysis(net)
    accuracy_temp[6] = net_acc_values[count]
    loss_temp[6] = net_loss_values[count]
    accuracy_analysis_array[t] = accuracy_temp
    loss_analysis_array[t] = loss_temp
    
    # normalizedGradientDifferenceFreezingProcedure
    freezing_rate_values[count] = normalizedGradientDifferenceFreezingProcedure(t+1,epochs,net,1,grad_dict)
    """
    
    count = count+1

    """ 
    if freeze == True:
        for param in net.parameters():
            param.requires_grad = True
            freeze = False

    if n!= None and n!=6: # Total number of layers (*modifica salta epoca)
        freeze = True
        # Layers freezing
        index = 0
        for param in net.parameters():
            param.requires_grad = False
            if index == ((n*2)+1):
                break
            index = index+1
    
    # gradientNormChangeFreezingProcedure
    gradient_difference_norm_change_array[t], gradient_norm_difference_change_array[t] = \
    gradientNormChangeFreezingProcedure(t+1,epochs,net,1,step,grad_dict)

    if n!= None: # Total number of layers
        if n==6:
            break
        # Layers freezing
        index = 0
        for param in net.parameters():
            param.requires_grad = False
            if index == ((n*2)+1):
                break
            index = index+1
    """ 

print("Done!")

"""
# normalizedGradientDifferenceFreezingProcedure
torch.save(freezing_rate_values, '../../plot/basicModel/freezingRateProcedure/freezing_rate50.pt')

# gradientNormChangeFreezingProcedure
torch.save(gradient_difference_norm_change_array, '../../plot/basicModel/gradientNormChanges/decay/gradient_difference_norm_change50.pt')
torch.save(gradient_norm_difference_change_array, '../../plot/basicModel/gradientNormChanges/decay/gradient_norm_difference_change50.pt')

# influence Analysis
torch.save(accuracy_analysis_array, '../../plot/basicModel/influenceAnalysis/accuracy50.pt')
torch.save(loss_analysis_array, '../../plot/basicModel/influenceAnalysis/loss50.pt')
"""