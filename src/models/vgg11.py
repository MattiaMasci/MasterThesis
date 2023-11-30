import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision.models import vgg11
from training.training_loops import train_loop, test_loop
#from training.training_loops_with_gradient_info import train_loop, test_loop
from analysis.freezing_methods import normalizedGradientDifferenceFreezingProcedure, gradientNormChangeFreezingProcedure
from analysis.influence import sequential2wrappers, layerInfluenceAnalysis
import logging
import time

logger = logging.getLogger('Main Logger')

class VGG11(nn.Module):
    def __init__(self, num_classes, device):
        super(VGG11, self).__init__()
        net = vgg11(weights=None)

        sequence = nn.Sequential()
        count = 0

        for children in net.children():
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
        self.num_classes = num_classes
        self.net = sequence
        self.device = device
        self.net.to(device)   

    def forward(self, x):
        x = self.net(x)
        return x

    def initialize(self):
        checkpoint = torch.load('../models/VGG11/weight0')
        self.net.load_state_dict(checkpoint)
    
    def reset(self):
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def train(self, dataloaders, learning_rate=1e-3, loss_fn=nn.functional.cross_entropy, epochs=150):  

        optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

        """
        # Learning rate decay
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        """

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

        # influenceAnalysis
        accuracy_analysis_array = torch.zeros([epochs,12])
        loss_analysis_array = torch.zeros([epochs,12])

        input, y = next(iter(dataloaders['train']))
        net_list = sequential2wrappers(self.net, self.num_classes, torch.unsqueeze(input[0],dim=0).to(self.device), self.device)

        # Leaves initializations fixed for influenceAnalysis
        index = 1
        for wrp in net_list:
            checkpoint = torch.load(f'../models/VGG11/leavesInitializations/{index}Net')
            wrp.leaf.load_state_dict(checkpoint)
            index = index+1

        net_list.to(self.device)
        """

        #i = 0

        # Time measurement
        start_time = time.time()

        # Training loop
        logger.info('----------- VGG11 TRAINING -----------')
        for t in range(epochs):
            logger.info(f'Epoch {t+1}\n-------------------------------')

            train_loop(dataloaders['train'], self.net, loss_fn, optimizer)#, gradient_list, abs_gradient_list)
            #checkpoint = torch.load('../models/VGG11/weight' + str(t+1))
            #net.load_state_dict(checkpoint)

            net_acc_values[count], net_loss_values[count] = test_loop(dataloaders['test'], self.net, loss_fn)

            """
            # Learning rate decay
            scheduler.step()

            # influenceAnalysis
            accuracy_array, loss_array = layerInfluenceAnalysis(self.net, net_list, dataloaders['influence'])
            accuracy_analysis_array[t] = torch.cat((accuracy_array,torch.tensor([net_acc_values[count]])),0)
            loss_analysis_array[t] = torch.cat((loss_array,torch.tensor([net_loss_values[count]])),0)

            # normalizedGradientDifferenceFreezingProcedure
            freezing_rate_values[count] = normalizedGradientDifferenceFreezingProcedure(t+1,epochs,self.net,1,grad_dict,grad_dict_abs)
            """

            count = count+1
            #i = i+1

        end_time = time.time()

        logger.info("Done!")
        logger.info(f'Total training time: {end_time-start_time}')

        """
        # normalizedGradientDifferenceFreezingProcedure
        torch.save(freezing_rate_values, '../plot/VGG11/freezingRateProcedure/freezing_rate50_true.pt')

        # influence Analysis
        torch.save(accuracy_analysis_array, '../plot/VGG11/influenceAnalysis/accuracy50.pt')
        torch.save(loss_analysis_array, '../plot/VGG11/influenceAnalysis/loss50.pt')
        """