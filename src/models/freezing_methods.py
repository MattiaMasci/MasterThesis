import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch import nn
import matplotlib.pyplot as plt
from collections import OrderedDict
from net_definition import FirstLayerNet, SecondLayerNet, ThirdLayerNet, FourthLayerNet, FifthLayerNet, SixthLayerNet, \
FourthLayerConv3Net, ThirdLayerConv2Net, FifthLayerConv4Net, SixthLayerModifiedFc1Net, WrapperNet
from training_loops import train_loop, test_loop
import os
import copy

def normalizedGradientDifferenceFreezingProcedure(current_epoch, total_epochs, model, frequence, grad_dict, grad_dict_abs):
    """ 
    Prende in input l'indice dell'epoca corrente, il numero di epoche totali, il modello in training,
    la frequenza con cui si vuole freezare e un dizionario ordinato con un tensore gradiente per ogni layer,
    e ritorna un indice relativo al layer da freezare (da 0 a n.layer-1)
    """
    if current_epoch % frequence == 0:
        # Freezing decisions part
        if current_epoch>=0: #(total_epochs // 10):
            print('--------- FREEZING PROCEDURE ---------')
            # Layer counting
            #num_layers = 0
            #for name, param in model.named_parameters():
            #    if 'weight' in name:
            #        num_layers = num_layers+1
            freezingRate_array = torch.zeros(len(grad_dict))-1
            layer_counter = -1
            for name, param in model.named_parameters():
                if 'weight' in name:
                    layer_counter = layer_counter+1
                    if grad_dict[layer_counter] != None:
                        numerator_totalSummation = torch.sum(abs(grad_dict[layer_counter]))
                        denominator_totalSummation = torch.sum(grad_dict_abs[layer_counter])
                    
                        freezingRate_array[layer_counter] = 1 - (numerator_totalSummation/denominator_totalSummation)
                
            # Array standardization
            print("Tensor before normalize:\n", freezingRate_array)

            mean, std= torch.mean(freezingRate_array[freezingRate_array!=-1]), \
                torch.std(freezingRate_array[freezingRate_array!=-1])
            print("Mean and Std before Normalize:\n", mean, std)

            freezingRate_array[freezingRate_array==-1] = float('-inf')
            standardized_freezingRate_array = freezingRate_array.clone()

            if not(torch.isnan(std)): standardized_freezingRate_array  = (standardized_freezingRate_array-mean)/std

            standardized_freezingRate_array[0] = freezingRate_array[0]
            standardized_freezingRate_array[layer_counter] = freezingRate_array[layer_counter]
            print("Tensor after Normalize:\n", standardized_freezingRate_array)
        
            # Maximum subarray sum
            cum_sum = torch.cumsum(standardized_freezingRate_array[standardized_freezingRate_array!=float('-inf')],dim=0)
            count = (standardized_freezingRate_array[standardized_freezingRate_array==float('-inf')].size(dim=0))
            n = torch.argmax(cum_sum)+count
        
            print('Cumulative sum array:')
            print(cum_sum)
            print('Calculated argmax:')
            print(n)
            return freezingRate_array #n, 

def gradientNormChangeFreezingProcedure(current_epoch, total_epochs, model, frequence, step, grad_dict):
    """ 
    Prende in input l'indice dell'epoca corrente, il numero di epoche totali, il modello in training,
    la frequenza con cui si vuole freezare, la frequenza con cui si vogliono controllare i cambiamenti nel gradiente
    e un dizionario ordinato con un tensore gradiente per ogni layer, e ritorna un indice relativo al layer 
    da freezare (da 0 a n.layer-1)    
    """
    if current_epoch % frequence == 0:
        # Freezing decisions part
        if current_epoch>=0: #(total_epochs // 10):
            print('--------- FREEZING PROCEDURE ---------')
            total_number_iterations = grad_dict[len(grad_dict)-1].size()[0]
            frozen_layer = 0
            for name, param in model.named_parameters():
                if not(param.requires_grad):
                    if 'weight' in name:
                        frozen_layer = frozen_layer+1
            # Array utilizzato per il metodo
            gradient_norm_array = torch.zeros(len(grad_dict))+float('inf')
            # Arrays utilizzati per i plot
            gradient_difference_norm_change_array = torch.zeros([7,(((total_number_iterations-2)//step)+1)])
            gradient_norm_difference_change_array = torch.zeros([7,(((total_number_iterations-2)//step)+1)])
            for i in range(frozen_layer,len(grad_dict)):
                somma = 0
                z = 0
                for j in range(1,total_number_iterations,step):
                    previous_iteration_norm = torch.norm(grad_dict[i][j-1])
                    gradient_norm_change = (abs(previous_iteration_norm-torch.norm(grad_dict[i][j])))/previous_iteration_norm
                    gradient_norm_change2 = (torch.norm(grad_dict[i][j-1]-grad_dict[i][j]))
                    gradient_difference_norm_change_array[i,z] = gradient_norm_change
                    gradient_norm_difference_change_array[i,z] = gradient_norm_change2
                    z = z+1
                    somma = somma + gradient_norm_change
                gradient_norm_array[i] = somma

            return gradient_difference_norm_change_array, gradient_norm_difference_change_array
        
            """gradient_norm_array = gradient_norm_array/(((total_number_iterations-2)//step)+1)
            print('Gradient norm change array:')
            print(gradient_norm_array)

            if (torch.argmin(gradient_norm_array) == frozen_layer):
                print('Freeze layer: '+ str(frozen_layer+1))
                print()
                return frozen_layer
            print()
            """

"""
# Additional linear layers initializations      
firstLinearLayer_weights = torch.zeros([10, 10, 4096])
for i in range(0,10):
    torch.nn.init.xavier_uniform_(firstLinearLayer_weights[i])

secondLinearLayer_weights = torch.zeros([10, 10, 2048])
for i in range(0,10):
    torch.nn.init.xavier_uniform_(secondLinearLayer_weights[i])

thirdLinearLayer_weights = torch.zeros([10, 10, 2048])
for i in range(0,10):
    torch.nn.init.xavier_uniform_(thirdLinearLayer_weights[i])

fourthLinearLayer_weights = torch.zeros([10, 10, 1024])
for i in range(0,10):
    torch.nn.init.xavier_uniform_(fourthLinearLayer_weights[i])

fifthLinearLayer_weights = torch.zeros([10, 10, 512])
for i in range(0,10):
    torch.nn.init.xavier_uniform_(fifthLinearLayer_weights[i])

sixthLinearLayer_weights = torch.zeros([10, 10, 250])
for i in range(0,10):
    torch.nn.init.xavier_uniform_(sixthLinearLayer_weights[i])

# Conv3Net analysis
fourthLinearLayer_weights = torch.zeros([10, 10, 100])
for i in range(0,10):
    torch.nn.init.xavier_uniform_(fourthLinearLayer_weights[i])
    
# Conv2Net analysis
thirdLinearLayer_weights = torch.zeros([10, 10, 100])
for i in range(0,10):
    torch.nn.init.xavier_uniform_(thirdLinearLayer_weights[i])

def layerInfluenceAnalysis(model):

    Prende in input il modello in training e ritorna in output i valori di accuracy e loss di n-1 (n = numero layer)
    modelli copia costruiti a partire dal primo aggiungendo incrementalmente i layer
    

    # Dataset loading
    training_data = torch.load('../../data/reduced_training_set.pt')
    test_data = torch.load('../../data/reduced_testing_set.pt')

    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    # Parameters setting
    learning_rate_ = 1e-3
    batch_size = 64
    loss_fn = nn.CrossEntropyLoss()    

    accuracy_array = torch.zeros([7])
    loss_array = torch.zeros([7])

    print('--------- Analysis ---------')
    print('FIRST TYPE NET TRAINING')

    first_layer_nets = nn.ModuleList()
    for i in range(0,10):
        # First net initialization
        first_layer_nets.insert(i, FirstLayerNet())
        first_layer_nets[i].weights_init(model.conv1.weight.data,firstLinearLayer_weights[i].clone())
        first_layer_nets[i].bias_init(model.conv1.bias.data)

    # Parameters setting 
    epochs = 1

    accuracy_sum = 0
    loss_sum = 0
    for i in range(0,10):
        optimizer = torch.optim.SGD(first_layer_nets[i].parameters(), lr=learning_rate_)
        # Training loop
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
        
            train_loop(train_dataloader, first_layer_nets[i], loss_fn, optimizer)
            accuracy, loss = test_loop(test_dataloader, first_layer_nets[i], loss_fn)
            accuracy_sum = accuracy_sum+accuracy
            loss_sum = loss_sum+loss
        
    accuracy_array[0] = accuracy_sum/10
    loss_array[0] = loss_sum/10

    print('SECOND TYPE NET TRAINING')

    second_layer_nets = nn.ModuleList()
    for i in range(0,10):
        # Second net initialization
        second_layer_nets.insert(i, SecondLayerNet())
        second_layer_nets[i].weights_init(model.conv1.weight.data,model.conv2.weight.data,secondLinearLayer_weights[i].clone())
        second_layer_nets[i].bias_init(model.conv1.bias.data,model.conv2.bias.data)
        
    accuracy_sum = 0
    loss_sum = 0
    for i in range(0,10):
        optimizer = torch.optim.SGD(second_layer_nets[i].parameters(), lr=learning_rate_)
        # Training loop
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
        
            train_loop(train_dataloader, second_layer_nets[i], loss_fn, optimizer)
            accuracy, loss = test_loop(test_dataloader, second_layer_nets[i], loss_fn)
            accuracy_sum = accuracy_sum+accuracy
            loss_sum = loss_sum+loss

    accuracy_array[1] = accuracy_sum/10
    loss_array[1] = loss_sum/10

    print('THIRD TYPE NET TRAINING')

    third_layer_nets = nn.ModuleList()
    for i in range(0,10):
        # Third net initialization
        third_layer_nets.insert(i, ThirdLayerNet())
        third_layer_nets[i].weights_init(model.conv1.weight.data,model.conv2.weight.data,\
                                         model.conv3.weight.data,thirdLinearLayer_weights[i].clone())
        third_layer_nets[i].bias_init(model.conv1.bias.data,model.conv2.bias.data,model.conv3.bias.data)
        
    accuracy_sum = 0
    loss_sum = 0
    for i in range(0,10):
        optimizer = torch.optim.SGD(third_layer_nets[i].parameters(), lr=learning_rate_)
        # Training loop
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
        
            train_loop(train_dataloader, third_layer_nets[i], loss_fn, optimizer)
            accuracy, loss = test_loop(test_dataloader, third_layer_nets[i], loss_fn)
            accuracy_sum = accuracy_sum+accuracy
            loss_sum = loss_sum+loss
    
    accuracy_array[2] = accuracy_sum/10
    loss_array[2] = loss_sum/10

    print('FOURTH TYPE NET TRAINING')

    fourth_layer_nets = nn.ModuleList()
    for i in range(0,10):
        # Fourth net initialization
        fourth_layer_nets.insert(i, FourthLayerNet())
        fourth_layer_nets[i].weights_init(model.conv1.weight.data,model.conv2.weight.data,\
                                         model.conv3.weight.data,model.conv4.weight.data,fourthLinearLayer_weights[i].clone())
        fourth_layer_nets[i].bias_init(model.conv1.bias.data,model.conv2.bias.data,model.conv3.bias.data,model.conv4.bias.data)
        
    accuracy_sum = 0
    loss_sum = 0
    for i in range(0,10):
        optimizer = torch.optim.SGD(fourth_layer_nets[i].parameters(), lr=learning_rate_)
        # Training loop
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
        
            train_loop(train_dataloader, fourth_layer_nets[i], loss_fn, optimizer)
            accuracy, loss = test_loop(test_dataloader, fourth_layer_nets[i], loss_fn)
            accuracy_sum = accuracy_sum+accuracy
            loss_sum = loss_sum+loss

    accuracy_array[3] = accuracy_sum/10
    loss_array[3] = loss_sum/10

    print('FIFTH TYPE NET TRAINING')

    fifth_layer_nets = nn.ModuleList()
    for i in range(0,10):
        # Fifth net initialization
        fifth_layer_nets.insert(i, FifthLayerNet())
        fifth_layer_nets[i].weights_init(model.conv1.weight.data,model.conv2.weight.data,\
                                         model.conv3.weight.data,model.conv4.weight.data,\
                                            model.conv5.weight.data,fifthLinearLayer_weights[i].clone())
        fifth_layer_nets[i].bias_init(model.conv1.bias.data,model.conv2.bias.data,model.conv3.bias.data,model.conv4.bias.data,\
                                      model.conv5.bias.data)
        
    accuracy_sum = 0
    loss_sum = 0
    for i in range(0,10):
        optimizer = torch.optim.SGD(fifth_layer_nets[i].parameters(), lr=learning_rate_)
        # Training loop
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
        
            train_loop(train_dataloader, fifth_layer_nets[i], loss_fn, optimizer)
            accuracy, loss = test_loop(test_dataloader, fifth_layer_nets[i], loss_fn)
            accuracy_sum = accuracy_sum+accuracy
            loss_sum = loss_sum+loss

    accuracy_array[4] = accuracy_sum/10
    loss_array[4] = loss_sum/10

    print('SIXTH TYPE NET TRAINING')

    sixth_layer_nets = nn.ModuleList()
    for i in range(0,10):
        # Sixth net initialization
        sixth_layer_nets.insert(i, SixthLayerModifiedFc1Net())
        sixth_layer_nets[i].weights_init(model.conv1.weight.data,model.conv2.weight.data,\
                                         model.conv3.weight.data,model.conv4.weight.data,\
                                            model.conv5.weight.data,model.fc1.weight.data,sixthLinearLayer_weights[i].clone())
        sixth_layer_nets[i].bias_init(model.conv1.bias.data,model.conv2.bias.data,model.conv3.bias.data,model.conv4.bias.data,\
                                      model.conv5.bias.data,model.fc1.bias.data)
        
    accuracy_sum = 0
    loss_sum = 0
    for i in range(0,10):
        optimizer = torch.optim.SGD(sixth_layer_nets[i].parameters(), lr=learning_rate_)
        # Training loop
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
        
            train_loop(train_dataloader, sixth_layer_nets[i], loss_fn, optimizer)
            accuracy, loss = test_loop(test_dataloader, sixth_layer_nets[i], loss_fn)
            accuracy_sum = accuracy_sum+accuracy
            loss_sum = loss_sum+loss

    accuracy_array[5] = accuracy_sum/10
    loss_array[5] = loss_sum/10

    # Conv3Net analysis
    print('FOURTH TYPE NET TRAINING')
    
    fourth_layer_nets = nn.ModuleList()
    for i in range(0,10):
        # Third net initialization
        fourth_layer_nets.insert(i, FourthLayerConv3Net())
        fourth_layer_nets[i].weights_init(model.conv1.weight.data,model.conv2.weight.data,\
                                         model.conv3.weight.data,model.fc1.weight.data,fourthLinearLayer_weights[i].clone())
        fourth_layer_nets[i].bias_init(model.conv1.bias.data,model.conv2.bias.data,model.conv3.bias.data,model.fc1.bias.data)
        
    accuracy_sum = 0
    loss_sum = 0
    for i in range(0,10):
        optimizer = torch.optim.SGD(fourth_layer_nets[i].parameters(), lr=learning_rate_)
        # Training loop
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
        
            train_loop(train_dataloader, fourth_layer_nets[i], loss_fn, optimizer)
            accuracy, loss = test_loop(test_dataloader, fourth_layer_nets[i], loss_fn)
            accuracy_sum = accuracy_sum+accuracy
            loss_sum = loss_sum+loss
    
    accuracy_array[3] = accuracy_sum/10
    loss_array[3] = loss_sum/10

    # Conv2Net analysis
    print('THIRD TYPE NET TRAINING')
    
    third_layer_nets = nn.ModuleList()
    for i in range(0,10):
        # Third net initialization
        third_layer_nets.insert(i, ThirdLayerConv2Net())
        third_layer_nets[i].weights_init(model.conv1.weight.data,model.conv2.weight.data,\
                                         model.fc1.weight.data,thirdLinearLayer_weights[i].clone())
        third_layer_nets[i].bias_init(model.conv1.bias.data,model.conv2.bias.data,model.fc1.bias.data)
        
    accuracy_sum = 0
    loss_sum = 0
    for i in range(0,10):
        optimizer = torch.optim.SGD(third_layer_nets[i].parameters(), lr=learning_rate_)
        # Training loop
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
        
            train_loop(train_dataloader, third_layer_nets[i], loss_fn, optimizer)
            accuracy, loss = test_loop(test_dataloader, third_layer_nets[i], loss_fn)
            accuracy_sum = accuracy_sum+accuracy
            loss_sum = loss_sum+loss
    
    accuracy_array[2] = accuracy_sum/10
    loss_array[2] = loss_sum/10

    # Conv4Net analysis
    print('FIFTH TYPE NET TRAINING')
    
    fifth_layer_nets = nn.ModuleList()
    for i in range(0,10):
        # Fifth net initialization
        fifth_layer_nets.insert(i, FifthLayerConv4Net())
        fifth_layer_nets[i].weights_init(model.conv1.weight.data,model.conv2.weight.data,\
                                         model.conv3.weight.data,model.conv4.weight.data,model.fc1.weight.data,\
                                            fifthLinearLayer_weights[i].clone())
        fifth_layer_nets[i].bias_init(model.conv1.bias.data,model.conv2.bias.data,model.conv3.bias.data,model.conv4.bias.data,\
                                      model.fc1.bias.data)
        
    accuracy_sum = 0
    loss_sum = 0
    for i in range(0,10):
        optimizer = torch.optim.SGD(fifth_layer_nets[i].parameters(), lr=learning_rate_)
        # Training loop
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
        
            train_loop(train_dataloader, fifth_layer_nets[i], loss_fn, optimizer)
            accuracy, loss = test_loop(test_dataloader, fifth_layer_nets[i], loss_fn)
            accuracy_sum = accuracy_sum+accuracy
            loss_sum = loss_sum+loss
    
    accuracy_array[4] = accuracy_sum/10
    loss_array[4] = loss_sum/10

    return accuracy_array, loss_array
    """

def layerInfluenceAnalysis(model, num_classes, batch_size, in_channels, in_height, in_width, iterations):
    """ 
    Prende in input il modello in training e ritorna in output i valori di accuracy e loss di n-1 (n = numero layer)
    modelli copia costruiti a partire dal primo aggiungendo incrementalmente i layer
    
    
    # Dataset loading
    training_data = torch.load('../../data/reduced_training_set.pt')
    test_data = torch.load('../../data/reduced_testing_set.pt')
    """

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
    transform= transform
    )

    test_data = datasets.CIFAR10(
    root="../../data",
    train=False,
    download=True,
    transform= transform
    )
    
    train_dataloader = DataLoader(training_data, batch_size)
    test_dataloader = DataLoader(test_data, batch_size)

    # Parameters setting
    learning_rate = 1e-3
    loss_fn = nn.CrossEntropyLoss()    
    epochs = 1

    print()
    print('----------- Analysis -----------')
    print()

     # Counting the number of layers in the network
    num_layers = 0
    layer_list = ('conv','linear')

    for children in model.children():
        if isinstance(children, nn.Sequential):
            for sub_children in children:
                if any(substring.lower() in str(sub_children).lower() for substring in layer_list):
                    num_layers = num_layers+1
        else:
            if any(substring.lower() in str(children).lower() for substring in layer_list):
                num_layers = num_layers+1

    # Freezing of the net
    for param in model.parameters():
        param.requires_grad = False

    net_list = netComposition(model)

    accuracy_array = torch.zeros([num_layers])
    loss_array = torch.zeros([num_layers])

    # Training of the nets
    index = 0
    for net in net_list:
        accuracy_sum = 0
        loss_sum = 0

        print('TRAINING OF ' + str(index+1) + ' TYPE OF NET')
        print()
        for i in range(0,iterations):
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

            # Training loop
            for t in range(epochs):
                print(f"Epoch {t+1}\n-------------------------------")
        
                train_loop(train_dataloader, net, loss_fn, optimizer)
                accuracy, loss = test_loop(test_dataloader, net, loss_fn)
                accuracy_sum = accuracy_sum+accuracy
                loss_sum = loss_sum+loss
            
            net.reset()
        print('--------------------')

        accuracy_array[index] = accuracy_sum/iterations
        loss_array[index] = loss_sum/iterations

        index = index+1

    for param in model.parameters():
        param.requires_grad = True

    return accuracy_array, loss_array

def netComposition(model):
    """ 
    
    """

    layer_list = ('conv','linear')

    # Composition of network
    count = 0

    net_list = nn.ModuleList()
    sequence = nn.Sequential()
    flattening = nn.Flatten()

    output = torch.randn(64,3,224,224)

    for children in model.children():
        if isinstance(children, nn.Sequential):
            for sub_children in children:
                if any(substring.lower() in str(sub_children).lower() for substring in layer_list) and count != 0:
                    if len(output.size()) > 2:
                        linear_input = flattening(output)
                        net = WrapperNet(copy.deepcopy(sequence), nn.Linear(linear_input.size(dim=1),10))
                        net.seq.add_module(str(count), nn.Flatten())
                    else:
                        net = WrapperNet(copy.deepcopy(sequence), nn.Linear(output.size(dim=1),10))
                    net_list.append(net)
                sequence.add_module(str(count),sub_children)
                output = sub_children(output)
                count = count+1   
        else:
            if any(substring.lower() in str(children).lower() for substring in layer_list) and count != 0:
                if len(output.size()) > 2:
                    linear_input = flattening(output)
                    net = WrapperNet(copy.deepcopy(sequence), nn.Linear(linear_input.size(dim=1),10))
                    net.seq.add_module(str(count), nn.Flatten())
                else:
                    net = WrapperNet(copy.deepcopy(sequence), nn.Linear(output.size(dim=1),10))
                net_list.append(net)
            sequence.add_module(str(count),children)
            output = children(output)
            count = count+1

    return net_list