import torch
from influence import layerInfluenceAnalysis, sequential2wrappers
from torchvision import datasets
from torchvision.models import vgg11
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from training_loops import train_loop, test_loop
#from training_loops_with_freezing_values import train_loop, test_loop
from freezing_methods import normalizedGradientDifferenceFreezingProcedure, gradientNormChangeFreezingProcedure
import logging
import time
import sys

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
  root="../../data",
  train=True,
  download=True,
  transform=transform
)

test_data = datasets.CIFAR10(
  root="../../data",
  train=False,
  download=True,
  transform=transform
)

# Dataloaders for VGG11 training
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

num_classes = len(training_data.classes)

"""
# Dataloaders for influence analysis
ratio = 0.5
batch_size = 64

# Define the desired subset size
subset_size = int(len(training_data)*ratio)

# Create a subset of the CIFAR10 dataset
subset_indices = torch.randperm(len(training_data))[:subset_size]
subset_train = Subset(training_data, subset_indices)

train_dataloader = DataLoader(subset_train, batch_size=batch_size, shuffle=True)

subset_size = int(len(test_data)*ratio)

# Create a subset of the CIFAR10 dataset
subset_indices = torch.randperm(len(test_data))[:subset_size]
subset_test = Subset(test_data, subset_indices)

test_dataloader = DataLoader(subset_test, batch_size=batch_size, shuffle=True)
dataloaders = {"train":train_dataloader,"test":test_dataloader}
"""

# Parameters setting
learning_rate = 1e-3
batch_size = 64
loss_fn = nn.CrossEntropyLoss()

# Model loading
vgg11 = vgg11(weights=None)

sequence = nn.Sequential()
count = 0

for children in vgg11.children():
    if isinstance(children,nn.Sequential):
        for sub_children in children:
            sequence.add_module(str(count),sub_children)
            count = count+1
            if count == 22:
                sequence.add_module(str(count),nn.Flatten())
                count = count+1
    else:
        sequence.add_module(str(count),children)
        count = count+1
        if count == 22:
                sequence.add_module(str(count),nn.Flatten())
                count = count+1

sequence.add_module(str(count),nn.Linear(in_features=1000, out_features=num_classes,bias=True))
net = sequence
net.to(device)
#print(net)

# Parameters setting 
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
epochs = 50

# For plot
net_acc_values = torch.zeros([epochs])
net_loss_values = torch.zeros([epochs])
count = 0

"""
# normalizedGradientDifferenceFreezingProcedure
freezing_rate_values = torch.zeros([epochs,12])
freeze = False

# Initialization of gradients arrays for normalizedGradientDifferenceFreezingProcedure
layer_list = ('conv','linear')
gradient_list = []
abs_gradient_list = []

for children in net:
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

# influenceAnalysis
accuracy_analysis_array = torch.zeros([epochs,12])
loss_analysis_array = torch.zeros([epochs,12])

input, y = next(iter(train_dataloader))
net_list = sequential2wrappers(net, num_classes, torch.unsqueeze(input[0],dim=0).to(device), device)

# Leaves initializations fixed for influenceAnalysis
index = 1
for wrp in net_list:
    checkpoint = torch.load(f'../../data/VGG11/leavesInitializations/{index}Net')
    wrp.leaf.load_state_dict(checkpoint)
    index = index+1

net_list.to(device)
"""

#i = 0

checkpoint = torch.load('../../data/VGG11/weight0')
net.load_state_dict(checkpoint)

# Time measurement
start_time = time.time()

# Training loop
logger.info('----------- VGG11 TRAINING -----------')
for t in range(epochs):
    logger.info(f'Epoch {t+1}\n-------------------------------')

    train_loop(train_dataloader, net, loss_fn, optimizer)#, gradient_list, abs_gradient_list)
    #checkpoint = torch.load('../../data/VGG11/weight' + str(t+1))
    #net.load_state_dict(checkpoint)

    net_acc_values[count], net_loss_values[count] = test_loop(test_dataloader, net, loss_fn)

    """
    accuracy_array, loss_array = layerInfluenceAnalysis(net, net_list, dataloaders)
    accuracy_analysis_array[t] = torch.cat((accuracy_array,torch.tensor([net_acc_values[count]])),0)
    loss_analysis_array[t] = torch.cat((loss_array,torch.tensor([net_loss_values[count]])),0)

    # normalizedGradientDifferenceFreezingProcedure
    freezing_rate_values[count] = normalizedGradientDifferenceFreezingProcedure(t+1,epochs,net,1,grad_dict,grad_dict_abs)
    """

    count = count+1
    #i = i+1

end_time = time.time()

print("Done!")
print('Total training time: ' + str(end_time-start_time))

"""
# normalizedGradientDifferenceFreezingProcedure
torch.save(freezing_rate_values, '../../plot/VGG11/freezingRateProcedure/freezing_rate50_true.pt')

# influence Analysis
torch.save(accuracy_analysis_array, '../../plot/VGG11/influenceAnalysis/accuracy50.pt')
torch.save(loss_analysis_array, '../../plot/VGG11/influenceAnalysis/loss50.pt')
"""