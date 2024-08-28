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
        self.fc1 = nn.Linear(3072, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # Input size = 3x32x32
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
        
    def weights_init(self, first, second, third):
        # Initialization of first linear layer weights
        self.fc1.weight = nn.Parameter(first, requires_grad=True)
        self.fc2.weight = nn.Parameter(second, requires_grad=True)
        self.fc3.weight = nn.Parameter(third, requires_grad=True)

firstLinearLayer_weights = torch.zeros([1000, 3072])
torch.nn.init.xavier_uniform_(firstLinearLayer_weights)

secondLinearLayer_weights = torch.zeros([500, 1000])
torch.nn.init.xavier_uniform_(secondLinearLayer_weights)

thirdLinearLayer_weights = torch.zeros([10, 500])
torch.nn.init.xavier_uniform_(thirdLinearLayer_weights)

# Model creation
net = Net()
net.weights_init(firstLinearLayer_weights, secondLinearLayer_weights, thirdLinearLayer_weights)

# Dataset loading
training_data = torch.load('../../data/reducedDataset/subset_test.pt')
test_data = torch.load('../../data/reducedDataset/subset_train.pt')

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# Parameters setting
learning_rate = 1e-3
batch_size = 64
loss_fn = nn.CrossEntropyLoss()

# Training and testing loop definition
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return [correct, test_loss]

# Parameters setting 
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
epochs = 2

fc1_weights = net.fc1.weight.clone()
fc2_weights = net.fc2.weight.clone()
fc3_weights = net.fc3.weight.clone()

fc1_bias = net.fc1.bias.clone()
fc2_bias = net.fc2.bias.clone()
fc3_bias = net.fc3.bias.clone()

print('FC1 INITIALIZATION WEIGHTS')
print(net.fc1.weight)

print('-------- TRAINING WITH FREEZING --------')

# Training loop
print(f"Epoch 1\n-------------------------------")
train_loop(train_dataloader, net, loss_fn, optimizer)
test_loop(test_dataloader, net, loss_fn)

print('COMPARISON FIRST EPOCH BIAS - INITIALIZATION BIAS')
print(False in torch.eq(fc1_bias, net.fc1.bias))
print(False in torch.eq(fc2_bias, net.fc2.bias))
print(False in torch.eq(fc3_bias, net.fc3.bias))

print('FC1 FIRST EPOCH WEIGHTS')
print(net.fc1.weight)

print('FC1 FIRST EPOCH GRADIENT')
print(net.fc1.weight.grad)

fc1_grad = net.fc1.weight.grad.clone()
fc2_grad = net.fc2.weight.grad.clone()
fc3_grad = net.fc3.weight.grad.clone()

fc1_weights_after = net.fc1.weight.clone()
fc2_weights_after = net.fc2.weight.clone()
fc3_weights_after = net.fc3.weight.clone()

fc1_bias_after = net.fc1.bias.clone()
fc2_bias_after = net.fc2.bias.clone()
fc3_bias_after = net.fc3.bias.clone()

net.fc1.weight.data = fc1_weights.clone()
net.fc2.weight.data = fc2_weights.clone()
#net.fc3.weight.data = fc3_weights.clone()

net.fc1.bias.data = fc1_bias.clone()
net.fc2.bias.data = fc2_bias.clone()
#net.fc3.bias.data = fc3_bias.clone()

print('FC1 RE-INITIALIZED WEIGHTS')
print(net.fc1.weight)

print(f"Epoch 2\n-------------------------------")
train_loop(train_dataloader, net, loss_fn, optimizer)
test_loop(test_dataloader, net, loss_fn)

print('FC1 FINAL WEIGHTS')
print(net.fc1.weight)

print('FC1 FINAL GRADIENT')
print(net.fc1.weight.grad)

print('COMPARISON FINAL GRADIENT - FIRST EPOCH GRADIENT')
print(False in torch.eq(fc1_grad, net.fc1.weight.grad))
print(False in torch.eq(fc2_grad, net.fc2.weight.grad))
print(False in torch.eq(fc3_grad, net.fc3.weight.grad))
print('COMPARISON FINAL WEIGHTS - INITIALIZATION WEIGHTS')
print(False in torch.eq(fc1_weights, net.fc1.weight))
print(False in torch.eq(fc2_weights, net.fc2.weight))
print(False in torch.eq(fc3_weights, net.fc3.weight))
print('COMPARISON FINAL WEIGHTS - FIRST EPOCH WEIGHTS')
print(False in torch.eq(fc1_weights_after, net.fc1.weight))
print(False in torch.eq(fc2_weights_after, net.fc2.weight))
print(False in torch.eq(fc3_weights_after, net.fc3.weight))
print('COMPARISON FINAL BIAS - INITIALIZATION BIAS')
print(False in torch.eq(fc1_bias, net.fc1.bias))
print(False in torch.eq(fc2_bias, net.fc2.bias))
print(False in torch.eq(fc3_bias, net.fc3.bias))
print('COMPARISON FINAL BIAS - FIRST EPOCH BIAS')
print(False in torch.eq(fc1_bias_after, net.fc1.bias))
print(False in torch.eq(fc2_bias_after, net.fc2.bias))
print(False in torch.eq(fc3_bias_after, net.fc3.bias))