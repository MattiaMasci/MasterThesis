import torch
from torch import nn
from torchvision.models import vgg11

layer_list = ('conv','linear')

net = vgg11(weights=None)

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

sequence.add_module(str(count),nn.Linear(in_features=1000, out_features=10,bias=True)) 

checkpoint = torch.load('../models/VGG11/init/init3')
sequence.load_state_dict(checkpoint)

net = vgg11(weights=None)

sequence1 = nn.Sequential()
count = 0
num_layers = 0

for children in net.children():
    if isinstance(children,nn.Sequential):
        for sub_children in children:
            if any(substring.lower() in str(sub_children).lower() for substring in layer_list):
                num_layers = num_layers+1
            sequence1.add_module(str(count),sub_children)
            count = count+1
            if count == 22:
                sequence1.add_module(str(count),nn.Flatten())
                count = count+1
    else:
        if any(substring.lower() in str(children).lower() for substring in layer_list):
            num_layers = num_layers+1
        sequence1.add_module(str(count),children)
        count = count+1
        if count == 22:
            sequence1.add_module(str(count),nn.Flatten())
            count = count+1

sequence1.add_module(str(count),nn.Linear(in_features=1000, out_features=10,bias=True)) 

checkpoint = torch.load('../models/VGG11/init/init3')
sequence.load_state_dict(checkpoint)

checkpoint1 = torch.load('../models/VGG11/init/init2')
sequence1.load_state_dict(checkpoint)

print(torch.equal(checkpoint['0.weight'], checkpoint1['0.weight']))