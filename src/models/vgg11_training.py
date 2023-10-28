import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.models import vgg11
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch import nn
import matplotlib.pyplot as plt
import os
from training_loops import train_loop, test_loop
#from training_loops_additional import train_loop, test_loop
from net_definition import WrapperNet
from freezing_methods import normalizedGradientDifferenceFreezingProcedure, gradientNormChangeFreezingProcedure,\
layerInfluenceAnalysis, netComposition
import copy

# Resize the images in the dataset
transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize( 
       (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) 
    )
])

# Dataset loading
training_data = datasets.CIFAR10(
  root="../../data",
  train=True,
  download=True,
  transform=transform
)

test_data = datasets.CIFAR10(
  root="../../data",
  train=False,
  download=True,
  transform=transform
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Parameters setting
learning_rate = 1e-3
batch_size = 64
loss_fn = nn.CrossEntropyLoss()

# Model loading
vgg11 = vgg11(weights=None)

sequence = nn.Sequential()
count = 0

for children in vgg11.children():
    if isinstance(children,nn.Sequential):
        for sub_children in children:
            sequence.add_module(str(count),sub_children)
            count = count+1
            if count == 22:
                sequence.add_module(str(count),nn.Flatten())
                count = count+1
    else:
        sequence.add_module(str(count),children)
        count = count+1
        if count == 22:
                sequence.add_module(str(count),nn.Flatten())
                count = count+1

net = WrapperNet(copy.deepcopy(sequence),nn.Linear(in_features=1000, out_features=10,bias=True))

# Parameters setting 
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
epochs = 50

# For plot
net_acc_values = torch.zeros([epochs])
net_loss_values = torch.zeros([epochs])
count = 0

"""
# normalizedGradientDifferenceFreezingProcedure
freezing_rate_values = torch.zeros([epochs,12])
freeze = False
"""

# Array for influence analysis
accuracy_analysis_array = torch.zeros([epochs,12])
loss_analysis_array = torch.zeros([epochs,12])


# Training loop
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")

    train_loop(train_dataloader, net, loss_fn, optimizer)
    net_acc_values[count], net_loss_values[count] = test_loop(test_dataloader, net, loss_fn)

    # influence Analysis
    accuracy_temp, loss_temp = layerInfluenceAnalysis(net, 10, batch_size, 3, 224, 224, 1)
    accuracy_temp[11] = net_acc_values[count]
    loss_temp[11] = net_loss_values[count]
    accuracy_analysis_array[t] = accuracy_temp
    loss_analysis_array[t] = loss_temp

    """
    # normalizedGradientDifferenceFreezingProcedure
    freezing_rate_values[count] = normalizedGradientDifferenceFreezingProcedure(t+1,epochs,net,1,grad_dict,grad_dict_abs)
    """

    count = count+1

print("Done!")

"""
# normalizedGradientDifferenceFreezingProcedure
torch.save(freezing_rate_values, '../../plot/VGG11/freezingRateProcedure/freezing_rate50_true.pt')
"""

# influence Analysis
torch.save(accuracy_analysis_array, '../../plot/VGG11/influenceAnalysis/accuracy50.pt')
torch.save(loss_analysis_array, '../../plot/VGG11/influenceAnalysis/loss50.pt')