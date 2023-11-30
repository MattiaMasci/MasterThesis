import torch
from torch import nn
#from training.training_loops import train_loop, test_loop
from training.training_loops_with_gradient_info import train_loop, test_loop
from analysis.freezing_methods import normalizedGradientDifferenceFreezingProcedure, gradientNormChangeFreezingProcedure
from analysis.influence import sequential2wrappers, layerInfluenceAnalysis
import logging
import time

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

    def initialize(self):
        checkpoint = torch.load('../models/CIFAR-net.pt')
        del checkpoint['model_weights']['conv5.weight'], checkpoint['model_weights']['conv5.bias'], \
            checkpoint['model_weights']['fc1.weight'], checkpoint['model_weights']['fc1.bias'], \
            checkpoint['model_weights']['fc2.weight'], checkpoint['model_weights']['fc2.bias']

        fc1_weights = torch.zeros([100, 1024])
        torch.nn.init.xavier_uniform_(fc1_weights)

        fc1_bias = torch.zeros(100)
        torch.nn.init.normal_(fc1_bias)

        fc2_weights = torch.zeros([10, 100])
        torch.nn.init.xavier_uniform_(fc2_weights)

        fc2_bias = torch.zeros(10)
        torch.nn.init.normal_(fc2_bias)

        checkpoint['model_weights']['fc1.weight'] = fc1_weights
        checkpoint['model_weights']['fc1.bias'] = fc1_bias
        checkpoint['model_weights']['fc2.weight'] = fc2_weights
        checkpoint['model_weights']['fc2.bias'] = fc2_bias

        net.load_state_dict(checkpoint['model_weights'])

    def train(self, dataloaders, learning_rate=1e-3, loss_fn=nn.functional.cross_entropy, epochs=50):
        self.optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

        # For plot
        net_acc_values = torch.zeros([epochs])
        net_loss_values = torch.zeros([epochs])
        count = 0

        # Array for influence analysis
        accuracy_analysis_array = torch.zeros([epochs,6])
        loss_analysis_array = torch.zeros([epochs,6])

        # Training loop
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")

            train_loop(train_dataloader, net, loss_fn, optimizer)
            net_acc_values[count], net_loss_values[count] = test_loop(test_dataloader, net, loss_fn)

            accuracy_temp, loss_temp = layerInfluenceAnalysis(net)
            accuracy_temp[5] = net_acc_values[count]
            loss_temp[5] = net_loss_values[count]
            accuracy_analysis_array[t] = accuracy_temp
            loss_analysis_array[t] = loss_temp

            count = count+1

        print("Done!")

        # influence Analysis
        torch.save(accuracy_analysis_array, '../../plot/basicModel/influenceAnalysis/conv4Net/accuracy50.pt')
        torch.save(loss_analysis_array, '../../plot/basicModel/influenceAnalysis/conv4Net/loss50.pt')