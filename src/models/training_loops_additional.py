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
    conv1_gradient = torch.zeros([64, 3, 3, 3])
    conv1_gradient_abs = torch.zeros([64, 3, 3, 3])
    conv2_gradient = torch.zeros([128, 64, 3, 3])
    conv2_gradient_abs = torch.zeros([128, 64, 3, 3])
    conv3_gradient = torch.zeros([256, 128, 3, 3])
    conv3_gradient_abs = torch.zeros([256, 128, 3, 3])
    conv4_gradient = torch.zeros([256, 256, 3, 3])
    conv4_gradient_abs = torch.zeros([256, 256, 3, 3])
    conv5_gradient = torch.zeros([512, 256, 3, 3])
    conv5_gradient_abs = torch.zeros([512, 256, 3, 3])
    conv6_gradient = torch.zeros([512, 512, 3, 3])
    conv6_gradient_abs = torch.zeros([512, 512, 3, 3])
    conv7_gradient = torch.zeros([512, 512, 3, 3])
    conv7_gradient_abs = torch.zeros([512, 512, 3, 3])
    conv8_gradient = torch.zeros([512, 512, 3, 3])
    conv8_gradient_abs = torch.zeros([512, 512, 3, 3])
    fc1_gradient = torch.zeros([4096, 25088])
    fc1_gradient_abs = torch.zeros([4096, 25088])
    fc2_gradient = torch.zeros([4096, 4096])
    fc2_gradient_abs = torch.zeros([4096, 4096])
    fc3_gradient = torch.zeros([1000, 4096])
    fc3_gradient_abs = torch.zeros([1000, 4096])
    fc4_gradient = torch.zeros([10, 1000])
    fc4_gradient_abs = torch.zeros([10, 1000])

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
        if model.features[0].weight.grad != None:
            conv1_gradient = torch.add(conv1_gradient, model.features[0].weight.grad)
            conv1_gradient_abs = torch.add(conv1_gradient_abs, abs(model.features[0].weight.grad))
        if model.features[3].weight.grad != None:
            conv2_gradient = torch.add(conv2_gradient, model.features[3].weight.grad)
            conv2_gradient_abs = torch.add(conv2_gradient_abs, abs(model.features[3].weight.grad))
        if model.features[6].weight.grad != None:
            conv3_gradient = torch.add(conv3_gradient, model.features[6].weight.grad)
            conv3_gradient_abs = torch.add(conv3_gradient_abs, abs(model.features[6].weight.grad))
        if model.features[8].weight.grad != None:
            conv4_gradient = torch.add(conv4_gradient, model.features[8].weight.grad)
            conv4_gradient_abs = torch.add(conv4_gradient_abs, abs(model.features[8].weight.grad))
        if model.features[11].weight.grad != None:
            conv5_gradient = torch.add(conv5_gradient, model.features[11].weight.grad)
            conv5_gradient_abs = torch.add(conv5_gradient_abs, abs(model.features[11].weight.grad))
        if model.features[13].weight.grad != None:
            conv6_gradient = torch.add(conv6_gradient, model.features[13].weight.grad)
            conv6_gradient_abs = torch.add(conv6_gradient_abs, abs(model.features[13].weight.grad))
        if model.features[16].weight.grad != None:
            conv7_gradient = torch.add(conv7_gradient, model.features[16].weight.grad)
            conv7_gradient_abs = torch.add(conv7_gradient_abs, abs(model.features[16].weight.grad))
        if model.features[18].weight.grad != None:
            conv8_gradient = torch.add(conv8_gradient, model.features[18].weight.grad)
            conv8_gradient_abs = torch.add(conv8_gradient_abs, abs(model.features[18].weight.grad))
        if model.classifier[0].weight.grad != None:
            fc1_gradient = torch.add(fc1_gradient, model.classifier[0].weight.grad)
            fc1_gradient_abs = torch.add(fc1_gradient_abs, abs(model.classifier[0].weight.grad))
        if model.classifier[3].weight.grad != None:
            fc2_gradient = torch.add(fc2_gradient, model.classifier[3].weight.grad)
            fc2_gradient_abs = torch.add(fc2_gradient_abs, abs(model.classifier[3].weight.grad))
        if model.classifier[6].weight.grad != None:
            fc3_gradient = torch.add(fc3_gradient, model.classifier[6].weight.grad)
            fc3_gradient_abs = torch.add(fc3_gradient_abs, abs(model.classifier[6].weight.grad))
        if model.classifier[7].weight.grad != None:
            fc4_gradient = torch.add(fc4_gradient, model.classifier[7].weight.grad)
            fc4_gradient_abs = torch.add(fc4_gradient_abs, abs(model.classifier[7].weight.grad))
        
        optimizer.step()
        # Once the gradients have been computed, they are passed to the optimizer.step() function. 
        # The optimizer.step() function is responsible for updating the weights and biases of the 
        # neural network based on the gradients. In other words, it adjusts the weights and biases 
        # in the direction that reduces the loss function the most.

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    if not(torch.any(conv1_gradient)):
        conv1_gradient = None
        conv1_gradient_abs = None
    if not(torch.any(conv2_gradient)):
        conv2_gradient = None
        conv2_gradient_abs = None
    if not(torch.any(conv3_gradient)):
        conv3_gradient = None
        conv3_gradient_abs = None
    if not(torch.any(conv4_gradient)):
        conv4_gradient = None
        conv4_gradient_abs = None
    if not(torch.any(conv5_gradient)):
        conv5_gradient = None
        conv5_gradient_abs = None
    if not(torch.any(conv6_gradient)):
        conv6_gradient = None
        conv6_gradient_abs = None
    if not(torch.any(conv7_gradient)):
        conv7_gradient = None
        conv7_gradient_abs = None
    if not(torch.any(conv8_gradient)):
        conv8_gradient = None
        conv8_gradient_abs = None
    if not(torch.any(fc1_gradient)):
        fc1_gradient = None
        fc1_gradient_abs = None
    if not(torch.any(fc2_gradient)):
        fc2_gradient = None
        fc2_gradient_abs = None
    if not(torch.any(fc3_gradient)):
        fc3_gradient = None
        fc3_gradient_abs = None
    if not(torch.any(fc4_gradient)):
        fc4_gradient = None
        fc4_gradient_abs = None
        
    grad_dict = {0:conv1_gradient,1:conv2_gradient,2:conv3_gradient,3:conv4_gradient,4:conv5_gradient,\
                            5:conv6_gradient,6:conv7_gradient,7:conv8_gradient,8:fc1_gradient,\
                                9:fc2_gradient,10:fc3_gradient,11:fc4_gradient}
    grad_dict_abs = {0:conv1_gradient_abs,1:conv2_gradient_abs,2:conv3_gradient_abs,3:conv4_gradient_abs,4:conv5_gradient_abs,\
                            5:conv6_gradient_abs,6:conv7_gradient_abs,7:conv8_gradient_abs,8:fc1_gradient_abs,\
                                9:fc2_gradient_abs,10:fc3_gradient_abs,11:fc4_gradient_abs}

    return grad_dict, grad_dict_abs


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