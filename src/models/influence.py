import torch
from torch import nn
import copy
from training_loops import train_loop, test_loop

# Model definition
class WrapperNet(nn.Module):
    def __init__(self, seq, leaf, lr=1e-3, loss_fn=nn.functional.cross_entropy, epochs=1):
        super(WrapperNet, self).__init__()
        self.seq = seq
        self.leaf = leaf
        self.loss_fn = loss_fn
        self.epochs = epochs        
        self.optimizer = torch.optim.SGD(self.parameters(),lr=lr)#, momentum=0.9, weight_decay=5e-4)

    def forward(self, x):
        x = self.seq(x)
        x = self.leaf(x)
        return x
    
    def reset(self):
        self.leaf.reset_parameters()

    def train(self, dataloaders, runs = 1, reset = False):   

        accuracy_sum = 0
        loss_sum = 0
        for r in range(runs):
            if(reset):
                self.reset()
            for t in range(self.epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                train_loop(dataloaders['train'], self, self.loss_fn, self.optimizer)
                accuracy, loss = test_loop(dataloaders['test'], self, self.loss_fn)
                accuracy_sum = accuracy_sum+accuracy
                loss_sum = loss_sum+loss
       
        return  accuracy_sum/runs, loss_sum/runs
         
def sequential2wrappers(seq_net, num_classes, input_sizes, device='cpu'):
    """ 
    Prende in input un modello di rete neurale e ritorna una lista di reti formate aggiungendo incrementalmente i vari 
    layer della rete originale
    """

    layer_list = ('conv','linear')

    layers_type = (nn.Linear,nn.Conv1d,nn.Conv2d, nn.Conv3d)

    # Composition of network
    count = 0

    net_list = nn.ModuleList()
    sequence = nn.Sequential()
    flattening = nn.Flatten()

    if len(input_sizes)>1:
        X = torch.randn(1,input_sizes[0],input_sizes[1],input_sizes[2]).to(device)
    else:
        X = torch.randn(1,input_sizes[0]).to(device)

    for children in seq_net:
        if isinstance(children, nn.Sequential):
            for sub_children in children:                
                if any(substring.lower() in str(sub_children).lower() for substring in layer_list) and count != 0:
                    # if it's a layer to which I need to attach a leaf
                    if len(X.size()) > 2:
                        linear_input = flattening(X)
                        net = WrapperNet(copy.deepcopy(sequence), nn.Linear(linear_input.size(dim=1),num_classes))
                        net.seq.add_module(str(count), nn.Flatten())
                    else:
                        net = WrapperNet(copy.deepcopy(sequence), nn.Linear(X.size(dim=1),num_classes))
                    net_list.append(net)

                sequence.add_module(str(count),sub_children)
                X = sub_children(X)
                count = count+1   
        else:
            if any(substring.lower() in str(children).lower() for substring in layer_list) and count != 0:
                # if it's a layer to which I need to attach a leaf
                if len(X.size()) > 2:
                    linear_input = flattening(X)
                    net = WrapperNet(copy.deepcopy(sequence), nn.Linear(linear_input.size(dim=1),num_classes))
                    net.seq.add_module(str(count), nn.Flatten())
                else:
                    net = WrapperNet(copy.deepcopy(sequence), nn.Linear(X.size(dim=1),num_classes))
                net_list.append(net)
            sequence.add_module(str(count),children)
            X = children(X)
            count = count+1

    return net_list

def layerInfluenceAnalysis(model, net_list, dataloaders , runs=1):
    """ 
    Prende in input il modello in training e ritorna in output i valori di accuracy e loss di n-1 (n = numero layer)
    modelli copia costruiti a partire dal primo aggiungendo incrementalmente i layer
    """

    print('\n----------- Analysis -----------\n')
    
    # Freezing of the net
    for param in model.parameters():
        param.requires_grad = False

    accuracy_array = torch.zeros([len(net_list)])
    loss_array = torch.zeros([len(net_list)])

    # Training of the nets
    index = 0
    for wrp in net_list:
        print('TRAINING OF ' + str(index+1) + ' TYPE OF NET\n')
        
        accuracy_array[index], loss_array[index] = wrp.train(dataloaders, runs)
            
        print('--------------------')

        index = index+1

    for param in model.parameters():
        param.requires_grad = True

    return accuracy_array, loss_array