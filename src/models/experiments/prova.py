import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, models
from torchvision.models import efficientnet_b0, vgg19, vgg11, VGG19_Weights
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch import nn
import matplotlib.pyplot as plt
import os

net = vgg11(weights=None)

net.classifier.add_module('7', nn.Linear(in_features=1000, out_features=10,bias=True))

for i in net.children():
    if isinstance(i, nn.Sequential):
        for m in i:
            if isinstance(m, nn.Conv2d):
                print(m)
            print(m)
    else:
        print(i)

"""
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
"""