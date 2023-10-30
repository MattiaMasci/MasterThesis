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
    layer_list = ('conv','linear')
    gradient_list = []
    abs_gradient_list = []

    for children in model.children():
        if isinstance(children, nn.Sequential):
            for sub_children in children:
                if any(substring.lower() in str(sub_children).lower() for substring in layer_list):
                    array = torch.zeros(sub_children.weight.size())
                    gradient_list.append(array.clone())
                    abs_gradient_list.append(array.clone())
        else:
            if any(substring.lower() in str(children).lower() for substring in layer_list):
                array = torch.zeros(children.weight.size())
                gradient_list.append(array.clone())
                abs_gradient_list.append(array.clone())

    size = len(dataloader.dataset)
    
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
        i = 0

        for children in model.children():
            if isinstance(children, nn.Sequential):
                for sub_children in children:
                    if any(substring.lower() in str(sub_children).lower() for substring in layer_list):
                        if sub_children.weight.grad != None:
                            gradient_list[i] = torch.add(gradient_list[i], sub_children.weight.grad)
                            abs_gradient_list[i] = torch.add(abs_gradient_list[i], abs(sub_children.weight.grad))
                            i = i+1
            else:
                if any(substring.lower() in str(children).lower() for substring in layer_list):
                    if children.weight.grad != None:
                            gradient_list[i] = torch.add(gradient_list[i], children.weight.grad)
                            abs_gradient_list[i] = torch.add(abs_gradient_list[i], abs(children.weight.grad))
                            i = i+1
        
        optimizer.step()
        # Once the gradients have been computed, they are passed to the optimizer.step() function. 
        # The optimizer.step() function is responsible for updating the weights and biases of the 
        # neural network based on the gradients. In other words, it adjusts the weights and biases 
        # in the direction that reduces the loss function the most.

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    i = 0
    for children in model.children():
        if isinstance(children, nn.Sequential):
            for sub_children in children:
                if any(substring.lower() in str(sub_children).lower() for substring in layer_list):
                    if not(torch.any(gradient_list[i])):
                        gradient_list[i] = None
                        abs_gradient_list[i] = None
                    i = i+1
        else:
            if any(substring.lower() in str(children).lower() for substring in layer_list):
                if not(torch.any(gradient_list[i])):
                    gradient_list[i] = None
                    abs_gradient_list[i] = None
                i = i+1

    grad_dict = dict(zip(range(len(gradient_list)), gradient_list))
    abs_grad_dict = dict(zip(range(len(abs_gradient_list)), abs_gradient_list))

    return grad_dict, abs_grad_dict

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