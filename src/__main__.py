import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from models.vgg11 import VGG11
import logging, sys

# Logging settings
logger = logging.getLogger('Main Logger')
logger.setLevel(logging.DEBUG)
std_out = logging.StreamHandler(stream=sys.stdout)
std_out.setLevel(logging.DEBUG)
#format = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s : %(message)s")
formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")
std_out.setFormatter(formatter)
logger.addHandler(std_out)

device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

def datasetReduction(training_data, test_data, ratio=0.5, batch_size=64):

    # Define the desired subset size
    subset_size = int(len(training_data)*ratio)

    # Create a subset of the CIFAR10 dataset
    subset_indices = torch.randperm(len(training_data))[:subset_size]
    subset_train = Subset(training_data, subset_indices)

    train_dataloader = DataLoader(subset_train, batch_size=batch_size, shuffle=False)

    subset_size = int(len(test_data)*ratio)

    # Create a subset of the CIFAR10 dataset
    subset_indices = torch.randperm(len(test_data))[:subset_size]
    subset_test = Subset(test_data, subset_indices)

    test_dataloader = DataLoader(subset_test, batch_size=batch_size, shuffle=False)
    dataloaders = {"train":train_dataloader,"test":test_dataloader}
    return dataloaders

def main():

    # Resize the images in the dataset
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize( 
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) 
        )
    ])

    # Dataset loading
    training_data = datasets.CIFAR10(
    root="../data",
    train=True,
    download=True,
    transform=transform
    )

    test_data = datasets.CIFAR10(
    root="../data",
    train=False,
    download=True,
    transform=transform
    )

    """
    dataloaders = datasetReduction(training_data, test_data, 0.1)

    dataloaders['influence'] = dict(dataloaders)
    """

    # Dataloaders for VGG11 training
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    dataloaders = {"train":train_dataloader,"test":test_dataloader}

    num_classes = len(training_data.classes)

    net = VGG11(num_classes, device)
    net.initialize()
    net.train(dataloaders)

if __name__ == "__main__":
    main()