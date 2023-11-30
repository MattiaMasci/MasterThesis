import torch
from torch import nn

influence_dataloaders = {'train': 3, 'test': 4}
dataloaders = {'train': 1, 'test': 2, 'influence': influence_dataloaders}

print(dataloaders)
print(dataloaders['influence'])