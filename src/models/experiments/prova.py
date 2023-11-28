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
import sys
import copy
sys.path.append('../')

"""
checkpoint = torch.load('../../../data/VGG11/new/weight0')
print(checkpoint.keys())
"""

for i in range(51):
    checkpoint = torch.load('../../../data/VGG11/weight' + str(i))
    checkpoint['0.weight'] = checkpoint['seq.0.weight']
    checkpoint['0.bias'] = checkpoint['seq.0.bias']
    checkpoint['3.weight'] = checkpoint['seq.3.weight']
    checkpoint['3.bias'] = checkpoint['seq.3.bias']
    checkpoint['6.weight'] = checkpoint['seq.6.weight']
    checkpoint['6.bias'] = checkpoint['seq.6.bias']
    checkpoint['8.weight'] = checkpoint['seq.8.weight']
    checkpoint['8.bias'] = checkpoint['seq.8.bias']
    checkpoint['11.weight'] = checkpoint['seq.11.weight']
    checkpoint['11.bias'] = checkpoint['seq.11.bias']
    checkpoint['13.weight'] = checkpoint['seq.13.weight']
    checkpoint['13.bias'] = checkpoint['seq.13.bias']
    checkpoint['16.weight'] = checkpoint['seq.16.weight']
    checkpoint['16.bias'] = checkpoint['seq.16.bias']
    checkpoint['18.weight'] = checkpoint['seq.18.weight']
    checkpoint['18.bias'] = checkpoint['seq.18.bias']
    checkpoint['23.weight'] = checkpoint['seq.23.weight']
    checkpoint['23.bias'] = checkpoint['seq.23.bias']
    checkpoint['26.weight'] = checkpoint['seq.26.weight']
    checkpoint['26.bias'] = checkpoint['seq.26.bias']
    checkpoint['29.weight'] = checkpoint['seq.29.weight']
    checkpoint['29.bias'] = checkpoint['seq.29.bias']
    checkpoint['30.weight'] = checkpoint['leaf.weight']
    checkpoint['30.bias'] = checkpoint['leaf.bias']

    del checkpoint['seq.0.weight'], checkpoint['seq.0.bias'], checkpoint['seq.3.weight'], checkpoint['seq.3.bias'],\
    checkpoint['seq.6.weight'], checkpoint['seq.6.bias'],checkpoint['seq.8.weight'], checkpoint['seq.8.bias'],\
    checkpoint['seq.11.weight'], checkpoint['seq.11.bias'],checkpoint['seq.13.weight'], checkpoint['seq.13.bias'],\
    checkpoint['seq.16.weight'], checkpoint['seq.16.bias'],checkpoint['seq.18.weight'], checkpoint['seq.18.bias'],\
    checkpoint['seq.23.weight'], checkpoint['seq.23.bias'],checkpoint['seq.26.weight'], checkpoint['seq.26.bias'],\
    checkpoint['seq.29.weight'], checkpoint['seq.29.bias'],checkpoint['leaf.weight'], checkpoint['leaf.bias'],

    torch.save(checkpoint, '../../../data/VGG11/weight' + str(i))