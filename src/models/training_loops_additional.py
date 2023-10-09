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

# Training and testing loop definition
def train_loop(dataloader, model, loss_fn, optimizer):

    # Initialization of gradients arrays
    conv1_gradient = torch.zeros([391, 16, 3, 5, 5])
    conv2_gradient = torch.zeros([391, 32, 16, 5, 5])
    conv3_gradient = torch.zeros([391, 32, 32, 3, 3])
    conv4_gradient = torch.zeros([391, 64, 32, 3, 3])
    conv5_gradient = torch.zeros([391, 128, 64, 3, 3])
    fc1_gradient = torch.zeros([391, 100, 512])
    fc2_gradient = torch.zeros([391, 10, 100])

    size = len(dataloader.dataset)
    counter = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        # When we call the backward() method on the loss tensor, the gradients computed by PyTorch accumulate 
        # (i.e., they are added to the existing gradients) for each parameter during each iteration. 
        # This is useful when we want to accumulate gradients across multiple batches, but it can lead to 
        # incorrect gradient computations when we only want to compute the gradients for a single batch. Therefore, 
        # before calling backward() for a new minibatch, we need to zero out the gradients from the previous minibatch. 
        # Otherwise, we would be using stale gradients from previous minibatches, which could lead to incorrect parameter updates.
        
        loss.backward()
        
        # Gradients of the weights of each layer
        if model.conv1.weight.grad != None: conv1_gradient[counter] = model.conv1.weight.grad.clone()
        if model.conv2.weight.grad != None: conv2_gradient[counter] = model.conv2.weight.grad.clone()
        if model.conv3.weight.grad != None: conv3_gradient[counter] = model.conv3.weight.grad.clone()
        if model.conv4.weight.grad != None: conv4_gradient[counter] = model.conv4.weight.grad.clone()
        if model.conv5.weight.grad != None: conv5_gradient[counter] = model.conv5.weight.grad.clone()
        if model.fc1.weight.grad != None: fc1_gradient[counter] = model.fc1.weight.grad.clone()
        if model.fc2.weight.grad != None: fc2_gradient[counter] = model.fc2.weight.grad.clone()
        
        counter = counter+1
        
        optimizer.step()
        # Once the gradients have been computed, they are passed to the optimizer.step() function. 
        # The optimizer.step() function is responsible for updating the weights and biases of the 
        # neural network based on the gradients. In other words, it adjusts the weights and biases 
        # in the direction that reduces the loss function the most.

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    if not(torch.any(conv1_gradient)): conv1_gradient = None
    if not(torch.any(conv2_gradient)): conv2_gradient = None
    if not(torch.any(conv3_gradient)): conv3_gradient = None
    if not(torch.any(conv4_gradient)): conv4_gradient = None
    if not(torch.any(conv5_gradient)): conv5_gradient = None
    if not(torch.any(fc1_gradient)): fc1_gradient = None
    if not(torch.any(fc2_gradient)): fc2_gradient = None
        
    grad_dict = {0:conv1_gradient,1:conv2_gradient,2:conv3_gradient,3:conv4_gradient,4:conv5_gradient,\
                            5:fc1_gradient,6:fc2_gradient}

    return grad_dict


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