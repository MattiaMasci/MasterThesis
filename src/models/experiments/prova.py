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

values1 = torch.load('../freezing_rate.pt')
values2 = torch.load('../freezing_rate50.pt')

print(torch.equal(values1,values2))
