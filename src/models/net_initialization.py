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
from net_definition import Net

# Weights initialization
firstConvolutionalLayer_weights = torch.zeros([16, 3, 5, 5])
torch.nn.init.xavier_uniform_(firstConvolutionalLayer_weights)

secondConvolutionalLayer_weights = torch.zeros([32, 16, 5, 5])
torch.nn.init.xavier_uniform_(secondConvolutionalLayer_weights)

thirdConvolutionalLayer_weights = torch.zeros([32, 32, 3, 3])
torch.nn.init.xavier_uniform_(thirdConvolutionalLayer_weights)

fourthConvolutionalLayer_weights = torch.zeros([64, 32, 3, 3])
torch.nn.init.xavier_uniform_(fourthConvolutionalLayer_weights)

fifthConvolutionalLayer_weights = torch.zeros([128, 64, 3, 3])
torch.nn.init.xavier_uniform_(fifthConvolutionalLayer_weights)

firstLinearLayer_weights = torch.zeros([100, 512])
torch.nn.init.xavier_uniform_(firstLinearLayer_weights)

secondLinearLayer_weights = torch.zeros([10, 100])
torch.nn.init.xavier_uniform_(secondLinearLayer_weights)

# Model creation
net = Net()
net.weights_init(firstConvolutionalLayer_weights, secondConvolutionalLayer_weights, thirdConvolutionalLayer_weights,\
                  fourthConvolutionalLayer_weights, fifthConvolutionalLayer_weights, \
                    firstLinearLayer_weights, secondLinearLayer_weights)

# Save the initialized model
torch.save({'model_weights': net.state_dict()}, '../../models/CIFAR-net.pt')