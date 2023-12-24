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
import logging
import time

logger = logging.getLogger('Main Logger')

device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

# Training and testing loop definition
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    """
    # Time accumulators
    for_loop_time_accumulator = 0
    forward_time_accumulator = 0
    backward_time_accumulator = 0
    start_time = time.time()
    for_loop_time = start_time
    """
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        """
        for_loop_time_accumulator = for_loop_time_accumulator + (time.time()-for_loop_time)
        forward_time = time.time()
        """
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        """
        forward_time_accumulator = forward_time_accumulator + (time.time()-forward_time)
        backward_time = time.time()
        """
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #backward_time_accumulator = backward_time_accumulator + (time.time()-backward_time)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.info(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
    """
        for_loop_time = time.time()
    end_time = time.time()
    
    logger.debug(f'Train loop total time: {round(end_time-start_time,2)}s')
    logger.debug(f'For loop time percentage with respect to '
    f'total time: {round((for_loop_time_accumulator/(end_time-start_time))*100,1)}%')
    logger.debug(f'Forward time percentage with respect to '
    f'total time: {round((forward_time_accumulator/(end_time-start_time))*100,1)}%')
    logger.debug(f'Backward time percentage with respect to '
    f'total time: {round((backward_time_accumulator/(end_time-start_time))*100,1)}%')
    logger.debug(f'Backward time percentage with respect to '
    f'forward time: {round((backward_time_accumulator/(forward_time_accumulator))*100,1)}%\n')
    """

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    logger.info(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')
    
    return [correct, test_loss]