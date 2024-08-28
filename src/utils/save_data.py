import torch
import logging

logger = logging.getLogger('Main Logger')

def save_acc_and_loss(method,freezing_epochs,net,net_acc_values,net_loss_values):

    if method == 'FRS':
        if freezing_epochs == 0:
            torch.save(net_acc_values, f'../../plot/{net.__class__.__name__}/accuracy/{net.init}/freezing_accuracy.pt')
            torch.save(net_loss_values, f'../../plot/{net.__class__.__name__}/loss/{net.init}/freezing_loss.pt')
        else:
            torch.save(net_acc_values, f'../../plot/{net.__class__.__name__}/accuracy/{net.init}/{freezing_epochs}/freezing_accuracy.pt')
            torch.save(net_loss_values, f'../../plot/{net.__class__.__name__}/loss/{net.init}/{freezing_epochs}/freezing_loss.pt')
    elif method == 'SFS':
        if freezing_epochs == 0:
            torch.save(net_acc_values, f'../../plot/{net.__class__.__name__}/accuracy/{net.init}/random_seq_accuracy.pt')
            torch.save(net_loss_values, f'../../plot/{net.__class__.__name__}/loss/{net.init}/random_seq_loss.pt')
        else:
            torch.save(net_acc_values, f'../../plot/{net.__class__.__name__}/accuracy/{net.init}/{freezing_epochs}/random_seq_accuracy.pt')
            torch.save(net_loss_values, f'../../plot/{net.__class__.__name__}/loss/{net.init}/{freezing_epochs}/random_seq_loss.pt')
    elif method == 'SCFS':
        if freezing_epochs == 0:
            torch.save(net_acc_values, f'../../plot/{net.__class__.__name__}/accuracy/{net.init}/scat_accuracy.pt')
            torch.save(net_loss_values, f'../../plot/{net.__class__.__name__}/loss/{net.init}/scat_loss.pt')
        else:
            torch.save(net_acc_values, f'../../plot/{net.__class__.__name__}/accuracy/{net.init}/{freezing_epochs}/scat_accuracy.pt')
            torch.save(net_loss_values, f'../../plot/{net.__class__.__name__}/loss/{net.init}/{freezing_epochs}/scat_loss.pt')
    elif method == 'RSFS':
        if freezing_epochs == 0:
            torch.save(net_acc_values, f'../../plot/{net.__class__.__name__}/accuracy/{net.init}/random_scat_accuracy.pt')
            torch.save(net_loss_values, f'../../plot/{net.__class__.__name__}/loss/{net.init}/random_scat_loss.pt')
        else:
            torch.save(net_acc_values, f'../../plot/{net.__class__.__name__}/accuracy/{net.init}/{freezing_epochs}/random_scat_accuracy.pt')
            torch.save(net_loss_values, f'../../plot/{net.__class__.__name__}/loss/{net.init}/{freezing_epochs}/random_scat_loss.pt')
    elif method == 'LES':
        if freezing_epochs == 0:
            torch.save(net_acc_values, f'../../plot/{net.__class__.__name__}/accuracy/{net.init}/eFreeze_accuracy.pt')
            torch.save(net_loss_values, f'../../plot/{net.__class__.__name__}/loss/{net.init}/eFreeze_loss.pt')
        else:
            torch.save(net_acc_values, f'../../plot/{net.__class__.__name__}/accuracy/{net.init}/{freezing_epochs}/eFreeze_accuracy.pt')
            torch.save(net_loss_values, f'../../plot/{net.__class__.__name__}/loss/{net.init}/{freezing_epochs}/eFreeze_loss.pt')
    elif method == 'LRS':
        if freezing_epochs == 0:
            torch.save(net_acc_values, f'../../plot/{net.__class__.__name__}/accuracy/{net.init}/layerOutRandom_accuracy.pt')
            torch.save(net_loss_values, f'../../plot/{net.__class__.__name__}/loss/{net.init}/layerOutRandom_loss.pt')
        else:
            torch.save(net_acc_values, f'../../plot/{net.__class__.__name__}/accuracy/{net.init}/{freezing_epochs}/layerOutRandom_accuracy.pt')
            torch.save(net_loss_values, f'../../plot/{net.__class__.__name__}/loss/{net.init}/{freezing_epochs}/layerOutRandom_loss.pt')
    else:
        torch.save(net_acc_values, f'../../plot/{net.__class__.__name__}/accuracy/{net.init}/accuracy.pt')
        torch.save(net_loss_values, f'../../plot/{net.__class__.__name__}/loss/{net.init}/loss.pt')

def save_time(method,freezing_epochs,net,time_values):
    
    if method == 'SFS':
        if freezing_epochs == 0:
            torch.save(time_values, f'../../plot/{net.__class__.__name__}/time/{net.init}/random_seq_time.pt')
        else:
            torch.save(time_values, f'../../plot/{net.__class__.__name__}/time/{net.init}/{freezing_epochs}/random_seq_time.pt')
    else:
        torch.save(time_values, f'../../plot/{net.__class__.__name__}/time/{net.init}/time.pt')