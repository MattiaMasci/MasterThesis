import torch
from torch import nn
#from training.training_loops import train_loop, test_loop
from training.training_loops_with_gradient_info import train_loop, test_loop
from analysis.freezing_methods import normalizedGradientDifferenceFreezingProcedure, gradientNormChangeFreezingProcedure
from analysis.influence import sequential2wrappers, layerInfluenceAnalysis
import logging
import time

class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            nn.Flatten()
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        # Input size = 1x32x32
        x = self.net(x)
        return x
        
    def weights_init(self, first, second, third, fourth, fifth, sixth, seventh):
        # Initialization of first convolutional layer weights
        self.net[0].weight = nn.Parameter(first, requires_grad=True)
        
        # Initialization of second convolutional layer weights
        self.net[3].weight = nn.Parameter(second, requires_grad=True)
        
        # Initialization of third convolutional layer weights
        self.net[6].weight = nn.Parameter(third, requires_grad=True)
        
        # Initialization of fourth convolutional layer weights
        self.net[8].weight = nn.Parameter(fourth, requires_grad=True)
        
        # Initialization of fifth convolutional layer weights
        self.net[11].weight = nn.Parameter(fifth, requires_grad=True)
        
        # Initialization of first linear layer weights
        self.net[15].weight = nn.Parameter(sixth, requires_grad=True)
        
        # Initialization of second linear layer weights
        self.net[17].weight = nn.Parameter(seventh, requires_grad=True)

    def initialize():
        checkpoint = torch.load('../models/CIFAR-net.pt')
        self.net.load_state_dict(checkpoint)

    def train(self, dataloaders, learning_rate=1e-3, loss_fn=nn.functional.cross_entropy, epochs=50):
        # Parameters setting 
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)

        """
        # Learning rate decay
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        """

        # For plot
        net_acc_values = torch.zeros([epochs])
        net_loss_values = torch.zeros([epochs])
        count = 0

        """
        # normalizedGradientDifferenceFreezingProcedure
        freezing_rate_values = torch.zeros([epochs,7])
        freeze = False
        """

        # Influence analysis
        accuracy_analysis_array = torch.zeros([epochs,7])
        loss_analysis_array = torch.zeros([epochs,7])

        """
        # gradientNormChangeFreezingProcedure
        step = 1
        gradient_difference_norm_change_array = torch.zeros([epochs,7,((389//step)+1)])
        gradient_norm_difference_change_array = torch.zeros([epochs,7,((389//step)+1)])
        """

        # Training loop
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")

            """
            print('PARAMETERS THAT REQUIRE GRADIENT:')
            for name, param in net.named_parameters():
                if param.requires_grad:
                    print(name)
            print()
            """

            train_loop(dataloaders['train'], self.net, loss_fn, optimizer)
            net_acc_values[count], net_loss_values[count] = test_loop(dataloaders['test'], self.net, loss_fn)

            """
            # Learning rate decay
            scheduler.step()
            """

            # influence Analysis
            accuracy_temp, loss_temp = layerInfluenceAnalysis(self.net, 10, [3, 32, 32], 1)
            accuracy_temp[6] = net_acc_values[count]
            loss_temp[6] = net_loss_values[count]
            accuracy_analysis_array[t] = accuracy_temp
            loss_analysis_array[t] = loss_temp
            
            """
            # normalizedGradientDifferenceFreezingProcedure
            freezing_rate_values[count] = normalizedGradientDifferenceFreezingProcedure(t+1,epochs,net,1,grad_dict,abs_grad_dict)
            """

            count = count+1

            """ 
            if freeze == True:
                for param in net.parameters():
                    param.requires_grad = True
                    freeze = False

            if n!= None and n!=6: # Total number of layers (*modifica salta epoca)
                freeze = True
                # Layers freezing
                index = 0
                for param in net.parameters():
                    param.requires_grad = False
                    if index == ((n*2)+1):
                        break
                    index = index+1
            
            # gradientNormChangeFreezingProcedure
            gradient_difference_norm_change_array[t], gradient_norm_difference_change_array[t] = \
            gradientNormChangeFreezingProcedure(t+1,epochs,net,1,step,grad_dict)

            if n!= None: # Total number of layers
                if n==6:
                    break
                # Layers freezing
                index = 0
                for param in net.parameters():
                    param.requires_grad = False
                    if index == ((n*2)+1):
                        break
                    index = index+1
            """ 

        print("Done!")

        """
        # normalizedGradientDifferenceFreezingProcedure
        torch.save(freezing_rate_values, '../../plot/basicModel/freezingRateProcedure/freezing_rate50.pt')

        # gradientNormChangeFreezingProcedure
        torch.save(gradient_difference_norm_change_array, '../../plot/basicModel/gradientNormChanges/decay/gradient_difference_norm_change50.pt')
        torch.save(gradient_norm_difference_change_array, '../../plot/basicModel/gradientNormChanges/decay/gradient_norm_difference_change50.pt')

        # influence Analysis
        torch.save(accuracy_analysis_array, '../../plot/basicModel/InfluenceAnalysis/accuracy50.pt')
        torch.save(loss_analysis_array, '../../plot/basicModel/InfluenceAnalysis/loss50.pt')
        """