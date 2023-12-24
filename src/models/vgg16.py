import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision.models import vgg16
from training.training_loops import train_loop, test_loop
#from training.training_loops_with_gradient_info import train_loop, test_loop
from analysis.freezing_methods import normalizedGradientDifferenceFreezingProcedure, gradientNormChangeFreezingProcedure,\
randomSequentialFreezing, randomScatteredFreezing
from analysis.influence import sequential2wrappers, layerInfluenceAnalysis
import logging
import time
import copy

logger = logging.getLogger('Main Logger')

class VGG16(nn.Module):
    def __init__(self, num_classes, device):
        super(VGG16, self).__init__()
        layer_list = ('conv','linear')

        net = vgg16(weights=None)
        net.classifier[6] = nn.Linear(4096, 10)

        sequence = nn.Sequential()
        count = 0
        num_layers = 0

        for children in net.children():
            if isinstance(children,nn.Sequential):
                for sub_children in children:
                    if any(substring.lower() in str(sub_children).lower() for substring in layer_list):
                        num_layers = num_layers+1
                    sequence.add_module(str(count),sub_children)
                    count = count+1
                    if count == 32:
                        sequence.add_module(str(count),nn.Flatten())
                        count = count+1
            else:
                if any(substring.lower() in str(children).lower() for substring in layer_list):
                    num_layers = num_layers+1
                sequence.add_module(str(count),children)
                count = count+1
                if count == 32:
                    sequence.add_module(str(count),nn.Flatten())
                    count = count+1

        #sequence.add_module(str(count),nn.Linear(in_features=1000, out_features=num_classes,bias=True))
        #num_layers = num_layers+1
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.net = sequence
        self.device = device
        self.net.to(device)   
        
    def forward(self, x):
        x = self.net(x)
        return x

    def initialize(self):
        checkpoint = torch.load('../models/VGG11/init/init1')
        self.net.load_state_dict(checkpoint)
    
    def reset(self):
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def train\
    (self, dataloaders, decay=False, method=None, learning_rate=1e-2, loss_fn=nn.functional.cross_entropy, epochs=90):  

        optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

        if decay == True:
            # Learning rate decay
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

        # For plot
        net_acc_values = torch.zeros([epochs])
        net_loss_values = torch.zeros([epochs])
        count = 0

        if method == 'freezing app' or method == 'freezing calc'\
        or method == 'sequential freezing' or method == 'scattered freezing':
            # normalizedGradientDifferenceFreezingProcedure
            freezing_rate_values = torch.zeros([epochs,self.num_layers])
            last_lr = None
            frozen_net = None
            calculated_layer = None
            defreeze = False
            frequence = 2

            # Initialization of gradients arrays for normalizedGradientDifferenceFreezingProcedure
            layer_list = ('conv','linear')
            gradient_list = []
            abs_gradient_list = []

            for children in self.net:
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

        elif method == 'influence analysis':
            # influenceAnalysis
            accuracy_analysis_array = torch.zeros([epochs,self.num_layers])
            loss_analysis_array = torch.zeros([epochs,self.num_layers])

            input, y = next(iter(dataloaders['train']))
            net_list = sequential2wrappers(self.net, self.num_classes, torch.unsqueeze(input[0],dim=0).to(self.device), self.device)

            # Leaves initializations fixed for influenceAnalysis
            index = 1
            for wrp in net_list:
                checkpoint = torch.load(f'../models/VGG11/leavesInitializations/{index}Net')
                wrp.leaf.load_state_dict(checkpoint)
                index = index+1

            net_list.to(self.device)

        # Time measurement
        start_time = time.time()

        # Training loop
        logger.info('----------- VGG16 TRAINING -----------')
        for t in range(epochs):
            logger.info(f'Epoch {t+1}\n-------------------------------')

            if method == 'freezing app' or method == 'sequential freezing' or method == 'scattered freezing':
                logger.info('PARAMETERS THAT REQUIRE GRADIENT:')
                for name, param in self.net.named_parameters():
                    if param.requires_grad:
                        logger.info(name)

            if method == 'freezing app' or method == 'freezing calc'\
            or method == 'sequential freezing' or method == 'scattered freezing':
                if frozen_net != True:
                    grad_dict, grad_dict_abs = \
                    train_loop\
                    (dataloaders['train'], self.net, loss_fn, optimizer, copy.copy(gradient_list), copy.copy(abs_gradient_list))
                    net_acc_values[count], net_loss_values[count] = test_loop(dataloaders['test'], self.net, loss_fn)
            elif method == None:
                    train_loop(dataloaders['train'], self.net, loss_fn, optimizer)
                    net_acc_values[count], net_loss_values[count] = test_loop(dataloaders['test'], self.net, loss_fn)
            
            if method == 'freezing app':
                # normalizedGradientDifferenceFreezingProcedure
                if last_lr!=scheduler.get_last_lr():
                    last_lr = scheduler.get_last_lr()
                    #freezing_rate_values[count]
                    calculated_layer, defreeze, frozen_net = \
                    normalizedGradientDifferenceFreezingProcedure\
                    (calculated_layer,t+1,epochs,self.net,frequence,grad_dict,grad_dict_abs,defreeze,True)
                else:
                    calculated_layer, defreeze, frozen_net = \
                    normalizedGradientDifferenceFreezingProcedure\
                    (calculated_layer,t+1,epochs,self.net,frequence,grad_dict,grad_dict_abs,defreeze)
            elif method == 'scattered freezing':
                # scatteredFreezingProcedure
                calculated_layer = normalizedGradientDifferenceFreezingProcedure\
                (calculated_layer,t+1,epochs,self.net,frequence,grad_dict,grad_dict_abs,defreeze,True)
                defreeze, frozen_net = \
                randomScatteredFreezing(calculated_layer,t+1,epochs,self.net,frequence,self.num_layers,defreeze,True)
            elif method == 'sequential freezing':
                # sequentialFreezingProcedure
                calculated_layer, defreeze, frozen_net = \
                randomSequentialFreezing(calculated_layer,t+1,epochs,self.net,frequence,self.num_layers,defreeze,True)
            elif method == 'influence analysis':
                # influenceAnalysis
                accuracy_array, loss_array = layerInfluenceAnalysis(self.net, net_list, dataloaders['influence'])
                accuracy_analysis_array[t] = torch.cat((accuracy_array,torch.tensor([net_acc_values[count]])),0)
                loss_analysis_array[t] = torch.cat((loss_array,torch.tensor([net_loss_values[count]])),0)

            if decay == True:
                # Learning rate decay
                scheduler.step()

            count = count+1

        end_time = time.time()

        logger.info("Done!")
        logger.info(f'Total training time: {end_time-start_time}')