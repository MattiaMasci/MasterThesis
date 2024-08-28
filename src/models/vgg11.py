import torch
from torch import nn
from torchvision.models import vgg11
import logging
import time
import copy

logger = logging.getLogger('Main Logger')

class VGG11(nn.Module):
    def __init__(self, num_classes, device):
        super(VGG11, self).__init__()
        layer_list = ('conv','linear')

        net = vgg11(weights=None)
        #net.classifier[6] = nn.Linear(4096, 10)

        sequence = nn.Sequential()
        count = 0
        num_layers = 0

        for children in net.children():
            if isinstance(children,nn.Sequential):
                for sub_children in children:
                    if any(substring.lower() in str(sub_children).lower() for substring in layer_list):
                        num_layers = num_layers+1
                    sequence.add_module(str(count),sub_children)
                    count = count+1
                    if count == 22:
                        sequence.add_module(str(count),nn.Flatten())
                        count = count+1
            else:
                if any(substring.lower() in str(children).lower() for substring in layer_list):
                    num_layers = num_layers+1
                sequence.add_module(str(count),children)
                count = count+1
                if count == 22:
                    sequence.add_module(str(count),nn.Flatten())
                    count = count+1

        sequence.add_module(str(count),nn.Linear(in_features=1000, out_features=num_classes,bias=True))
        num_layers = num_layers+1
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.net = sequence
        self.device = device
        self.net.to(device)   
        
    def forward(self, x):
        x = self.net(x)
        return x

    def initialize(self, init):
        self.init = init
        path = f'../models/VGG11/init/{self.init}'
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint)
        logger.info(f'Initialization in use: {path}')
    
    def reset(self):
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()