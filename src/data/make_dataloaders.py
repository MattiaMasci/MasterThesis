import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

def datasetReduction(training_data, test_data, ratio):

    train_indices = torch.load('../../data/CIFAR-10_stats/indices_tensor_train.pt')
    test_indices = torch.load('../../data/CIFAR-10_stats/indices_tensor_test.pt')

    train_tensor = torch.flatten(train_indices[:,:int(train_indices.shape[1]*ratio)]).int()
    train_tensor = train_tensor[torch.randperm(train_tensor.shape[0])]

    subset_train = Subset(training_data, train_tensor)

    test_tensor = torch.flatten(test_indices[:,:int(test_indices.shape[1]*ratio)]).int()
    test_tensor = test_tensor[torch.randperm(test_tensor.shape[0])]

    subset_test = Subset(test_data, test_tensor)

    tensor = torch.zeros(10)
    for item in subset_train:
        tensor[item[1]] +=1

    #print(tensor)

    tensor = torch.zeros(10)
    for item in subset_test:
        tensor[item[1]] +=1

    #print(tensor)

    return subset_train, subset_test

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

training_data, test_data = datasetReduction(training_data, test_data, 0.9)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

torch.save(train_dataloader, '../../data/dataloaders/dataset_reduction/train_dataloader_45000.pth')
torch.save(test_dataloader, '../../data/dataloaders/dataset_reduction/test_dataloader_45000.pth')