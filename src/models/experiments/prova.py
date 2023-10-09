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

firstLinearLayer_weights = torch.zeros([10, 10, 2048])
torch.nn.init.xavier_uniform_(firstLinearLayer_weights[0])

print(firstLinearLayer_weights)

def funzione():
    global secondLinearLayer_weights
    secondLinearLayer_weights = secondLinearLayer_weights +1
    print(secondLinearLayer_weights)

#funzione()
