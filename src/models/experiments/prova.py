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
 
mod = nn.Linear(10,1)
mod1 = nn.Linear(10,1)

seq1 = nn.Sequential()
seq2 = nn.Sequential()

seq1.add_module('0', mod)

seq2 = copy.copy(seq1)

seq1 = nn.Sequential()

seq1.add_module('0',mod1)

print(seq1[0].weight)
print(seq2[0].weight)

with torch.no_grad():
    seq1[0].weight[0][0] = 18

print(seq1[0].weight)
print(seq2[0].weight)

#print(seq1 == seq2)
#print(seq1.get_submodule('0') == seq2.get_submodule('0'))