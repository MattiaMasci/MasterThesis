import torch
from influence import layerInfluenceAnalysis, sequential2wrappers
from torchvision import datasets
from torchvision.models import vgg11
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from training_loops import train_loop, test_loop
from freezing_methods import normalizedGradientDifferenceFreezingProcedure, gradientNormChangeFreezingProcedure

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

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
dataloaders = {"train":train_dataloader,"test":test_dataloader}

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

sequence.add_module(str(count),nn.Linear(in_features=1000, out_features=10,bias=True))
net = sequence
print(net)

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

# Array for influence analysis
accuracy_analysis_array = torch.zeros([epochs,12])
loss_analysis_array = torch.zeros([epochs,12])
"""

net.to(device)

i = 0

input, y = next(iter(train_dataloader)) 
net_list = sequential2wrappers(net, 10, [3, 224, 224], device)
net_list.to(device)

#torch.save(net.state_dict(), '../../data/VGG11/weight' + str(i))

# Training loop
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")

    train_loop(train_dataloader, net, loss_fn, optimizer)
    net_acc_values[count], net_loss_values[count] = test_loop(test_dataloader, net, loss_fn)

    accuracy_array, loss_array = layerInfluenceAnalysis(net, net_list, dataloaders)

    """
    # influence Analysis
    accuracy_temp, loss_temp = layerInfluenceAnalysis(net, 10, [3, 224, 224], 1)
    accuracy_temp[11] = net_acc_values[count]
    loss_temp[11] = net_loss_values[count]
    accuracy_analysis_array[t] = accuracy_temp
    loss_analysis_array[t] = loss_temp

    # normalizedGradientDifferenceFreezingProcedure
    freezing_rate_values[count] = normalizedGradientDifferenceFreezingProcedure(t+1,epochs,net,1,grad_dict,grad_dict_abs)
    """

    count = count+1
    i = i+1
    #torch.save(net.state_dict(), '../../data/VGG11/weight' + str(i))

print("Done!")

"""
# normalizedGradientDifferenceFreezingProcedure
torch.save(freezing_rate_values, '../../plot/VGG11/freezingRateProcedure/freezing_rate50_true.pt')

# influence Analysis
torch.save(accuracy_analysis_array, '../../plot/VGG11/influenceAnalysis/accuracy50.pt')
torch.save(loss_analysis_array, '../../plot/VGG11/influenceAnalysis/loss50.pt')
"""