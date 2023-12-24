import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

# Resize the images in the dataset
train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Dataset loading
training_data = datasets.CIFAR10(
root="../../data",
train=True,
download=True,
transform=train_transform
)

test_data = datasets.CIFAR10(
root="../../data",
train=False,
download=True,
transform=test_transform
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

torch.save(train_dataloader, '../../data/dataloaders/train_dataloader.pth')
torch.save(test_dataloader, '../../data/dataloaders/test_dataloader.pth')