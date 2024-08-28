import torch
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

# Download CIFAR-10 dataset
training_data = datasets.CIFAR10(
  root="../../data",
  train=True,
  download=True,
  transform = ToTensor()
)

test_data = datasets.CIFAR10(
  root="../../data",
  train=False,
  download=True,
  transform = ToTensor()
)

ratio=0.5

# Define the desired subset size
subset_size = int(len(training_data)*ratio)

# Create a subset of the CIFAR10 dataset
subset_indices = torch.randperm(len(training_data))[:subset_size]
subset_train = Subset(training_data, subset_indices)

subset_size = int(len(test_data)*ratio)

# Create a subset of the CIFAR10 dataset
subset_indices = torch.randperm(len(test_data))[:subset_size]
subset_test = Subset(test_data, subset_indices)

# Save the reduced dataset
torch.save(subset_train, '../../data/reducedDataset/subset_train.pt')
torch.save(subset_test, '../../data/reducedDataset/subset_test.pt')