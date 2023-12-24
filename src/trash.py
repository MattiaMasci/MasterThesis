import torch
from torch import nn
from torchvision.models import vgg16
import numpy as np
import random

model = vgg16(weights=None)
print(model)
model.classifier[6] = nn.Linear(4096, 10)
print(model)