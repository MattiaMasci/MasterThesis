import torch
import argparse
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from models.vgg11 import VGG11
from models.vgg13 import VGG13
from models.vgg16 import VGG16
from models.vgg19 import VGG19
from torch import nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from training.training_loops import train_loop, train_loop_with_gradient_info, test_loop
from analysis.freezing_methods import randomSequentialFreezing, randomScatteredFreezing,\
normalizedGradientDifferenceFreezingProcedure, gradientNormChangeFreezingProcedure, scatteredFreezing,\
randomScatteredFreezing, layerOut_eFreeze, layerOut_random
from analysis.influence import sequential2wrappers, layerInfluenceAnalysis
from utils.save_data import save_acc_and_loss, save_time
import logging, sys, time, copy

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='CFS')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default = 90, help='number of epochs to train (default: 90)')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate (default: 0.01)')
parser.add_argument('--decay', type=str2bool, default=True)
parser.add_argument('--training_method', type=str, default=None,
                    help='freezing scheme to apply',choices=['SFS','RSFS','SCFS','FRS','LES','LRS',None])
parser.add_argument('--freezing_period', type=int, default=1,
                    help='frequency (in terms of epochs) with which freezing is applied',
                    choices=[1,2,3,4,5])
parser.add_argument('--freezing_span_fraction', type=float,
                    help='duration of the freezing phase (percentage wirh respect to the freezig period)')
parser.add_argument('--init', type=str, default = 'init1', help='chosen initialization')
parser.add_argument('--num_classes', type=int, default = 10, help='number of classes in the dataset')
parser.add_argument('--optimizer_momentum', type=float, default = 0.9)
parser.add_argument('--optimizer_weight_decay', type=float, default = 5e-4)
parser.add_argument('--initial_percentage', type=int, default = 10,
                    help='percentage of warm-up epochs')

# Logging settings
logger = logging.getLogger('Main Logger')
logger.setLevel(logging.DEBUG)
std_out = logging.StreamHandler(stream=sys.stdout)
std_out.setLevel(logging.DEBUG)
#format = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s : %(message)s")
formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")
std_out.setFormatter(formatter)
logger.addHandler(std_out)

def datasetReduction(training_data, test_data, batch_size, ratio=0.5):

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

def main(args):
    logger.info(args)
    
    """
    dataloaders = datasetReduction(training_data, test_data, args.batch_size, 0.1)
    
    dataloaders['influence'] = dict(dataloaders)
    
    training_data = torch.load('../data/reducedDataset/subset_train.pt')
    test_data = torch.load('../data/reducedDataset/subset_test.pt')

    # Dataloaders for VGG11 training
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    """
    
    """
    # REDUCED DATALOADERS
    train_dataloader = torch.load('../data/dataloaders/dataset_reduction/train_dataloader_35000.pth')
    test_dataloader = torch.load('../data/dataloaders/dataset_reduction/test_dataloader_35000.pth')
    """
    
    train_dataloader = torch.load('../data/dataloaders/train_dataloader.pth')
    test_dataloader = torch.load('../data/dataloaders/test_dataloader.pth')

    dataloaders = {"train":train_dataloader,"test":test_dataloader}

    num_classes = args.num_classes

    method = args.training_method
    freezing_period = args.freezing_period
    freezing_span_fraction = args.freezing_span_fraction
    init = args.init

    decay = args.decay

    # Initialize model
    net = VGG11(num_classes, args.device)
    net.initialize(init)

    # Set hyperparameters
    epochs=args.epochs
    loss_fn = nn.functional.cross_entropy
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                momentum=args.optimizer_momentum, weight_decay=args.optimizer_weight_decay)

    if decay == True:
        # Learning rate decay
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        #scheduler = MultiStepLR(optimizer, milestones=[55,121], gamma=0.1)

    # For plot
    net_acc_values = torch.zeros([epochs])
    net_loss_values = torch.zeros([epochs])
    time_values = torch.zeros([epochs])
    count = 0

    logger.info(f'Selected method: {method}')

    if method is not None:
        # Variables used to manage freezing
        freezing_rate_values = torch.zeros([epochs,net.num_layers])
        last_lr = None
        initial_percentage = args.initial_percentage
        calculations = False
        frozen_net = False
        calculated_layer = None
        defreeze = 0
        warm_up_epochs = round(epochs / initial_percentage)

        # Initialization of gradients arrays for normalizedGradientDifferenceFreezingProcedure
        layer_list = ('conv','linear')
        gradient_list = []
        abs_gradient_list = []

        for children in net.net:
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

    # Average frozen layers calculation
    frozen_layers_sum = 0
    active_layers = 0

    # Time measurement
    start_time = time.time()

    # Training loop
    logger.info(f'----------- {net.__class__.__name__} TRAINING -----------')
    for t in range(epochs):
        logger.info(f'Epoch {t+1}\n-------------------------------')

        if method in ('FRS','SCFS','SFS','RSFS','LES','LRS'):
            logger.debug('PARAMETERS THAT REQUIRE GRADIENT CALCULATIONS:')
            for name, param in net.net.named_parameters():
                if param.requires_grad:
                    logger.debug(name)
                    active_layers = active_layers+1
            
            frozen_layers_sum = frozen_layers_sum+net.num_layers-(active_layers/2) 
            active_layers = 0
            logger.debug('\n-------------------------------')

        # Training and testing phases
        if method == 'FRS':
            if frozen_net != True:
                if (t+1) >= warm_up_epochs:
                    if last_lr!=scheduler.get_last_lr():
                        last_lr = scheduler.get_last_lr()
                        calculations = True
                if calculations == True and defreeze == 0 and ((t+1)-warm_up_epochs) % freezing_period == 0:
                    grad_dict, grad_dict_abs = \
                    train_loop_with_gradient_info\
                    (dataloaders['train'], net.net, loss_fn, optimizer, copy.copy(gradient_list), copy.copy(abs_gradient_list), calculations)
                else:
                    grad_dict, grad_dict_abs = \
                    train_loop_with_gradient_info\
                    (dataloaders['train'], net.net, loss_fn, optimizer, copy.copy(gradient_list), copy.copy(abs_gradient_list))
                net_acc_values[count], net_loss_values[count] = test_loop(dataloaders['test'], net.net, loss_fn)
        elif method == 'SCFS':
            if frozen_net != True:
                if ((t+1)-warm_up_epochs) % freezing_period == 0 and (t+1) >= warm_up_epochs:
                    grad_dict, grad_dict_abs = \
                    train_loop_with_gradient_info\
                    (dataloaders['train'], net.net, loss_fn, optimizer, copy.copy(gradient_list),\
                    copy.copy(abs_gradient_list), True)
                else:
                    grad_dict, grad_dict_abs = \
                    train_loop_with_gradient_info\
                    (dataloaders['train'], net.net, loss_fn, optimizer, copy.copy(gradient_list),\
                    copy.copy(abs_gradient_list))
                net_acc_values[count], net_loss_values[count] = test_loop(dataloaders['test'], net.net, loss_fn)
        elif method in ('SFS','RSFS','LES','LRS'):
            if frozen_net != True:
                train_loop(dataloaders['train'], net.net, loss_fn, optimizer)
                net_acc_values[count], net_loss_values[count] = test_loop(dataloaders['test'], net.net, loss_fn)
        else:
            train_loop(dataloaders['train'], net.net, loss_fn, optimizer)
            net_acc_values[count], net_loss_values[count] = test_loop(dataloaders['test'], net.net, loss_fn)

        # Application of freezing schemes
        if t < (epochs-1):
            if method == 'FRS':
                # normalizedGradientDifferenceFreezingProcedure
                calculated_layer, calculations, defreeze, frozen_net = \
                normalizedGradientDifferenceFreezingProcedure\
                (calculated_layer,t+1,epochs,net.net,freezing_period,freezing_span_fraction,\
                net.num_layers,frozen_net,grad_dict,grad_dict_abs,defreeze,calculations,initial_percentage)
            elif method == 'LES':
                # layerOut eFreeze
                defreeze, frozen_net = layerOut_eFreeze(t+1,epochs,net.net,freezing_period,freezing_span_fraction,\
                net.num_layers,frozen_net,defreeze,initial_percentage)
            elif method == 'LRS':
                # layerOut with random generation of freeze probability vector
                defreeze, frozen_net = layerOut_random(t+1,epochs,net.net,freezing_period,freezing_span_fraction,\
                net.num_layers,frozen_net,defreeze,initial_percentage)
            elif method == 'SCFS':
                # scatteredFreezingProcedure
                if ((t+1)-warm_up_epochs) % freezing_period == 0 and (t+1) >= warm_up_epochs:
                    calculated_layer = normalizedGradientDifferenceFreezingProcedure\
                    (calculated_layer,t+1,epochs,net.net,freezing_period,freezing_span_fraction,\
                    net.num_layers,frozen_net,grad_dict,grad_dict_abs,defreeze,True,initial_percentage)
                    defreeze, frozen_net = \
                    scatteredFreezing\
                    (calculated_layer,t+1,epochs,net.net,freezing_period,freezing_span_fraction,net.num_layers,\
                    frozen_net,defreeze,True,initial_percentage)
                else:
                    defreeze, frozen_net = \
                    scatteredFreezing\
                    (calculated_layer,t+1,epochs,net.net,freezing_period,freezing_span_fraction,net.num_layers,\
                    frozen_net,defreeze,False,initial_percentage)
            elif method == 'RSFS':
                # randomScatteredFreezingProcedure
                calculated_layer, defreeze, frozen_net = \
                randomScatteredFreezing\
                (t+1,epochs,net.net,freezing_period,freezing_span_fraction,net.num_layers,frozen_net,defreeze,initial_percentage)
            elif method == 'SFS':
                # sequentialFreezingProcedure
                calculated_layer, defreeze, frozen_net = \
                randomSequentialFreezing\
                (t+1,epochs,net.net,freezing_period,freezing_span_fraction,net.num_layers,frozen_net,defreeze,90)

        if decay == True:
            # Learning rate decay
            scheduler.step()

        epoch_time = time.time()
        time_values[count] = epoch_time-start_time

        count = count+1

    end_time = time.time()

    logger.info("Done!")
    logger.info(f'Total training time: {end_time-start_time}')
    
    # Aggiusta freezing span fraction
    """if method == 'FRS' or method == 'SFS' or method == 'SCFS'\
    or method == 'RSFS' or method == 'LES' or method == 'LRS':
        after_warm_up_epochs = epochs-warm_up_epochs
        if freezing_epochs == 0:
            logger.info(f'Sum of the frozen epochs : {after_warm_up_epochs}')
            logger.info(f'Sum of the frozen layers : {frozen_layers_sum}')
            logger.info(f'Frozen layers on average : {frozen_layers_sum/after_warm_up_epochs}')
        else:
            logger.info(f'Sum of the frozen epochs : {(after_warm_up_epochs-(after_warm_up_epochs//(freezing_epochs+1)))}')
            logger.info(f'Sum of the frozen layers : {frozen_layers_sum}')
            logger.info\
            (f'Frozen layers on average : {frozen_layers_sum/((after_warm_up_epochs-(after_warm_up_epochs//(freezing_epochs+1))))}')

    save_acc_and_loss(method,freezing_epochs,net,net_acc_values,net_loss_values)
    save_time(method,freezing_epochs,net,time_values)"""
        
if __name__ == "__main__":
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)