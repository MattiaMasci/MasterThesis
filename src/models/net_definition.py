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

# Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )

        """
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.flatten = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)
        """
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
        """
        
    def weights_init(self, first, second, third, fourth, fifth, sixth, seventh):
        # Initialization of first convolutional layer weights
        self.features[0].weight = nn.Parameter(first, requires_grad=True)
        
        # Initialization of second convolutional layer weights
        self.features[3].weight = nn.Parameter(second, requires_grad=True)
        
        # Initialization of third convolutional layer weights
        self.features[6].weight = nn.Parameter(third, requires_grad=True)
        
        # Initialization of fourth convolutional layer weights
        self.features[8].weight = nn.Parameter(fourth, requires_grad=True)
        
        # Initialization of fifth convolutional layer weights
        self.features[11].weight = nn.Parameter(fifth, requires_grad=True)
        
        # Initialization of first linear layer weights
        self.classifier[0].weight = nn.Parameter(sixth, requires_grad=True)
        
        # Initialization of second linear layer weights
        self.classifier[2].weight = nn.Parameter(seventh, requires_grad=True)

# Model definition
class FirstLayerNet(nn.Module):
    def __init__(self):
        super(FirstLayerNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4096, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x
        
    def weights_init(self, first, second):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=False)
        
        # Initialization of second linear layer weights
        self.fc1.weight = nn.Parameter(second, requires_grad=True)

    def bias_init(self, first):
        # Initialization of first convolutional layer bias
        self.conv1.bias = nn.Parameter(first, requires_grad=False)

# Model definition
class SecondLayerNet(nn.Module):
    def __init__(self):
        super(SecondLayerNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2048, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x
        
    def weights_init(self, first, second, third):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=False)
        
        # Initialization of second convolutional layer weights
        self.conv2.weight = nn.Parameter(second, requires_grad=False)
        
        # Initialization of first linear layer weights
        self.fc1.weight = nn.Parameter(third, requires_grad=True)

    def bias_init(self, first, second):
        # Initialization of first convolutional layer bias
        self.conv1.bias = nn.Parameter(first, requires_grad=False)

        # Initialization of second convolutional layer bias
        self.conv2.bias = nn.Parameter(second, requires_grad=False)

# Model definition
class ThirdLayerNet(nn.Module):
    def __init__(self):
        super(ThirdLayerNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2048, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return x
        
    def weights_init(self, first, second, third, fourth):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=False)
        
        # Initialization of second convolutional layer weights
        self.conv2.weight = nn.Parameter(second, requires_grad=False)
        
        # Initialization of third convolutional layer weights
        self.conv3.weight = nn.Parameter(third, requires_grad=False)
        
        # Initialization of first linear layer weights
        self.fc1.weight = nn.Parameter(fourth, requires_grad=True)

    def bias_init(self, first, second, third):
        # Initialization of first convolutional layer bias
        self.conv1.bias = nn.Parameter(first, requires_grad=False)

        # Initialization of second convolutional layer bias
        self.conv2.bias = nn.Parameter(second, requires_grad=False)

        # Initialization of third convolutional layer bias
        self.conv3.bias = nn.Parameter(third, requires_grad=False)

# Model definition
class FourthLayerNet(nn.Module):
    def __init__(self):
        super(FourthLayerNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x
        
    def weights_init(self, first, second, third, fourth, fifth):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=False)
        
        # Initialization of second convolutional layer weights
        self.conv2.weight = nn.Parameter(second, requires_grad=False)
        
        # Initialization of third convolutional layer weights
        self.conv3.weight = nn.Parameter(third, requires_grad=False)
        
        # Initialization of fourth convolutional layer weights
        self.conv4.weight = nn.Parameter(fourth, requires_grad=False)
        
        # Initialization of first linear layer weights
        self.fc1.weight = nn.Parameter(fifth, requires_grad=True)

    def bias_init(self, first, second, third, fourth):
        # Initialization of first convolutional layer bias
        self.conv1.bias = nn.Parameter(first, requires_grad=False)

        # Initialization of second convolutional layer bias
        self.conv2.bias = nn.Parameter(second, requires_grad=False)

        # Initialization of third convolutional layer bias
        self.conv3.bias = nn.Parameter(third, requires_grad=False)

        # Initialization of fourth convolutional layer bias
        self.conv4.bias = nn.Parameter(fourth, requires_grad=False)

# Model definition
class FifthLayerNet(nn.Module):
    def __init__(self):
        super(FifthLayerNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x
        
    def weights_init(self, first, second, third, fourth, fifth, sixth):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=False)
        
        # Initialization of second convolutional layer weights
        self.conv2.weight = nn.Parameter(second, requires_grad=False)
        
        # Initialization of third convolutional layer weights
        self.conv3.weight = nn.Parameter(third, requires_grad=False)
        
        # Initialization of fourth convolutional layer weights
        self.conv4.weight = nn.Parameter(fourth, requires_grad=False)
        
        # Initialization of fifth convolutional layer weights
        self.conv5.weight = nn.Parameter(fifth, requires_grad=False)
        
        # Initialization of first linear layer weights
        self.fc1.weight = nn.Parameter(sixth, requires_grad=True)

    def bias_init(self, first, second, third, fourth, fifth):
        # Initialization of first convolutional layer bias
        self.conv1.bias = nn.Parameter(first, requires_grad=False)

        # Initialization of second convolutional layer bias
        self.conv2.bias = nn.Parameter(second, requires_grad=False)

        # Initialization of third convolutional layer bias
        self.conv3.bias = nn.Parameter(third, requires_grad=False)

        # Initialization of fourth convolutional layer bias
        self.conv4.bias = nn.Parameter(fourth, requires_grad=False)

        # Initialization of fifth convolutional layer bias
        self.conv5.bias = nn.Parameter(fifth, requires_grad=False)

# Model definition
class SixthLayerNet(nn.Module):
    def __init__(self):
        super(SixthLayerNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def weights_init(self, first, second, third, fourth, fifth, sixth, seventh):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=False)
        
        # Initialization of second convolutional layer weights
        self.conv2.weight = nn.Parameter(second, requires_grad=False)
        
        # Initialization of third convolutional layer weights
        self.conv3.weight = nn.Parameter(third, requires_grad=False)
        
        # Initialization of fourth convolutional layer weights
        self.conv4.weight = nn.Parameter(fourth, requires_grad=False)
        
        # Initialization of fifth convolutional layer weights
        self.conv5.weight = nn.Parameter(fifth, requires_grad=False)
        
        # Initialization of first linear layer weights
        self.fc1.weight = nn.Parameter(sixth, requires_grad=False)
        
        # Initialization of second linear layer weights
        self.fc2.weight = nn.Parameter(seventh, requires_grad=True)

    def bias_init(self, first, second, third, fourth, fifth, sixth):
        # Initialization of first convolutional layer bias
        self.conv1.bias = nn.Parameter(first, requires_grad=False)

        # Initialization of second convolutional layer bias
        self.conv2.bias = nn.Parameter(second, requires_grad=False)

        # Initialization of third convolutional layer bias
        self.conv3.bias = nn.Parameter(third, requires_grad=False)

        # Initialization of fourth convolutional layer bias
        self.conv4.bias = nn.Parameter(fourth, requires_grad=False)

        # Initialization of fifth convolutional layer bias
        self.conv5.bias = nn.Parameter(fifth, requires_grad=False)

        # Initialization of first linear layer bias
        self.fc1.bias = nn.Parameter(sixth, requires_grad=False)

# Model definition
class Conv2Net(nn.Module):
    def __init__(self):
        super(Conv2Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2048, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.flatten = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
        
    def weights_init(self, first, second, third, fourth):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=True)
        
        # Initialization of second convolutional layer weights
        self.conv2.weight = nn.Parameter(second, requires_grad=True)
        
        # Initialization of first linear layer weights
        self.fc1.weight = nn.Parameter(third, requires_grad=True)
        
        # Initialization of second linear layer weights
        self.fc2.weight = nn.Parameter(fourth, requires_grad=True)

# Model definition
class ThirdLayerConv2Net(nn.Module):
    def __init__(self):
        super(ThirdLayerConv2Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2048, 100)
        self.fc2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def weights_init(self, first, second, third, fourth):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=False)
        
        # Initialization of second convolutional layer weights
        self.conv2.weight = nn.Parameter(second, requires_grad=False)
        
        # Initialization of fourth convolutional layer weights
        self.fc1.weight = nn.Parameter(third, requires_grad=False)
        
        # Initialization of first linear layer weights
        self.fc2.weight = nn.Parameter(fourth, requires_grad=True)

    def bias_init(self, first, second, third):
        # Initialization of first convolutional layer bias
        self.conv1.bias = nn.Parameter(first, requires_grad=False)

        # Initialization of second convolutional layer bias
        self.conv2.bias = nn.Parameter(second, requires_grad=False)

        # Initialization of fourth convolutional layer bias
        self.fc1.bias = nn.Parameter(third, requires_grad=False)

# Model definition
class Conv3Net(nn.Module):
    def __init__(self):
        super(Conv3Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2048, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.flatten = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
        
    def weights_init(self, first, second, third, fourth, fifth):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=True)
        
        # Initialization of second convolutional layer weights
        self.conv2.weight = nn.Parameter(second, requires_grad=True)
        
        # Initialization of third convolutional layer weights
        self.conv3.weight = nn.Parameter(third, requires_grad=True)
        
        # Initialization of first linear layer weights
        self.fc1.weight = nn.Parameter(fourth, requires_grad=True)
        
        # Initialization of second linear layer weights
        self.fc2.weight = nn.Parameter(fifth, requires_grad=True)

# Model definition
class FourthLayerConv3Net(nn.Module):
    def __init__(self):
        super(FourthLayerConv3Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2048, 100)
        self.fc2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def weights_init(self, first, second, third, fourth, fifth):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=False)
        
        # Initialization of second convolutional layer weights
        self.conv2.weight = nn.Parameter(second, requires_grad=False)
        
        # Initialization of third convolutional layer weights
        self.conv3.weight = nn.Parameter(third, requires_grad=False)
        
        # Initialization of fourth convolutional layer weights
        self.fc1.weight = nn.Parameter(fourth, requires_grad=False)
        
        # Initialization of first linear layer weights
        self.fc2.weight = nn.Parameter(fifth, requires_grad=True)

    def bias_init(self, first, second, third, fourth):
        # Initialization of first convolutional layer bias
        self.conv1.bias = nn.Parameter(first, requires_grad=False)

        # Initialization of second convolutional layer bias
        self.conv2.bias = nn.Parameter(second, requires_grad=False)

        # Initialization of third convolutional layer bias
        self.conv3.bias = nn.Parameter(third, requires_grad=False)

        # Initialization of fourth convolutional layer bias
        self.fc1.bias = nn.Parameter(fourth, requires_grad=False)

# Model definition
class Conv4Net(nn.Module):
    def __init__(self):
        super(Conv4Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1024, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.flatten = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
        
    def weights_init(self, first, second, third, fourth, fifth, sixth):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=True)
        
        # Initialization of second convolutional layer weights
        self.conv2.weight = nn.Parameter(second, requires_grad=True)
        
        # Initialization of third convolutional layer weights
        self.conv3.weight = nn.Parameter(third, requires_grad=True)

        # Initialization of third convolutional layer weights
        self.conv4.weight = nn.Parameter(fourth, requires_grad=True)
        
        # Initialization of first linear layer weights
        self.fc1.weight = nn.Parameter(fifth, requires_grad=True)
        
        # Initialization of second linear layer weights
        self.fc2.weight = nn.Parameter(sixth, requires_grad=True)

# Model definition
class FifthLayerConv4Net(nn.Module):
    def __init__(self):
        super(FifthLayerConv4Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1024, 100)
        self.fc2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def weights_init(self, first, second, third, fourth, fifth, sixth):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=False)
        
        # Initialization of second convolutional layer weights
        self.conv2.weight = nn.Parameter(second, requires_grad=False)
        
        # Initialization of third convolutional layer weights
        self.conv3.weight = nn.Parameter(third, requires_grad=False)

        # Initialization of third convolutional layer weights
        self.conv4.weight = nn.Parameter(fourth, requires_grad=False)
        
        # Initialization of fourth convolutional layer weights
        self.fc1.weight = nn.Parameter(fifth, requires_grad=False)
        
        # Initialization of first linear layer weights
        self.fc2.weight = nn.Parameter(sixth, requires_grad=True)

    def bias_init(self, first, second, third, fourth, fifth):
        # Initialization of first convolutional layer bias
        self.conv1.bias = nn.Parameter(first, requires_grad=False)

        # Initialization of second convolutional layer bias
        self.conv2.bias = nn.Parameter(second, requires_grad=False)

        # Initialization of third convolutional layer bias
        self.conv3.bias = nn.Parameter(third, requires_grad=False)

        # Initialization of third convolutional layer bias
        self.conv4.bias = nn.Parameter(fourth, requires_grad=False)

        # Initialization of fourth convolutional layer bias
        self.fc1.bias = nn.Parameter(fifth, requires_grad=False)

# Model definition
class ModifiedFc1Net(nn.Module):
    def __init__(self):
        super(ModifiedFc1Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512, 250)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(250, 10)
        self.flatten = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
        
    def weights_init(self, first, second, third, fourth, fifth, sixth, seventh):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=True)
        
        # Initialization of second convolutional layer weights
        self.conv2.weight = nn.Parameter(second, requires_grad=True)
        
        # Initialization of third convolutional layer weights
        self.conv3.weight = nn.Parameter(third, requires_grad=True)
        
        # Initialization of fourth convolutional layer weights
        self.conv4.weight = nn.Parameter(fourth, requires_grad=True)
        
        # Initialization of fifth convolutional layer weights
        self.conv5.weight = nn.Parameter(fifth, requires_grad=True)
        
        # Initialization of first linear layer weights
        self.fc1.weight = nn.Parameter(sixth, requires_grad=True)
        
        # Initialization of second linear layer weights
        self.fc2.weight = nn.Parameter(seventh, requires_grad=True)

# Model definition
class SixthLayerModifiedFc1Net(nn.Module):
    def __init__(self):
        super(SixthLayerModifiedFc1Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512, 250)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(250, 10)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def weights_init(self, first, second, third, fourth, fifth, sixth, seventh):
        # Initialization of first convolutional layer weights
        self.conv1.weight = nn.Parameter(first, requires_grad=False)
        
        # Initialization of second convolutional layer weights
        self.conv2.weight = nn.Parameter(second, requires_grad=False)
        
        # Initialization of third convolutional layer weights
        self.conv3.weight = nn.Parameter(third, requires_grad=False)
        
        # Initialization of fourth convolutional layer weights
        self.conv4.weight = nn.Parameter(fourth, requires_grad=False)
        
        # Initialization of fifth convolutional layer weights
        self.conv5.weight = nn.Parameter(fifth, requires_grad=False)
        
        # Initialization of first linear layer weights
        self.fc1.weight = nn.Parameter(sixth, requires_grad=False)
        
        # Initialization of second linear layer weights
        self.fc2.weight = nn.Parameter(seventh, requires_grad=True)

    def bias_init(self, first, second, third, fourth, fifth, sixth):
        # Initialization of first convolutional layer bias
        self.conv1.bias = nn.Parameter(first, requires_grad=False)

        # Initialization of second convolutional layer bias
        self.conv2.bias = nn.Parameter(second, requires_grad=False)

        # Initialization of third convolutional layer bias
        self.conv3.bias = nn.Parameter(third, requires_grad=False)

        # Initialization of fourth convolutional layer bias
        self.conv4.bias = nn.Parameter(fourth, requires_grad=False)

        # Initialization of fifth convolutional layer bias
        self.conv5.bias = nn.Parameter(fifth, requires_grad=False)

        # Initialization of first linear layer bias
        self.fc1.bias = nn.Parameter(sixth, requires_grad=False)

# Model definition
class WrapperNet(nn.Module):
    def __init__(self):
        super(WrapperNet, self).__init__()
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.net(x)
        return x