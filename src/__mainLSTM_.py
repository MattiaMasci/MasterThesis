import torch
import numpy as np
from data.load_HAR_dataset import load_X, load_y
from training.HAR_training import train
from torch import nn
from models.LSTM import LSTMModel, init_weights
from utils.Functions import plot, evaluate, save_data
import config as cfg
import data.data_HAR_file as df
import logging, sys

parser = argparse.ArgumentParser(description='CFS')
parser.add_argument('--freezing_schema', type=str, default=None,
                    help='freezing scheme to apply',choices=['SFS','RSFS','LES','LRS',None])
parser.add_argument('--freezing_frequence', type=int, default=1,
                    help='frequency (in terms of epochs) with which freezing is applied',
                    choices=[1,2,3,4,5])
parser.add_argument('--init', type=str, default = 'init1', help='chosen initialization')

# Logging settings
logger = logging.getLogger('Main Logger')
logger.setLevel(logging.DEBUG)
std_out = logging.StreamHandler(stream=sys.stdout)
std_out.setLevel(logging.DEBUG)
#format = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s : %(message)s")
formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")
std_out.setFormatter(formatter)
logger.addHandler(std_out)

# Data file to load X and y values

X_train_signals_paths = df.X_train_signals_paths
X_test_signals_paths = df.X_test_signals_paths

y_train_path = df.y_train_path
y_test_path = df.y_test_path

# LSTM Neural Network's internal structure

n_hidden = cfg.n_hidden
n_classes = cfg.n_classes
epochs = cfg.n_epochs
learning_rate = cfg.learning_rate
weight_decay = cfg.weight_decay
clip_val = cfg.clip_val
diag = cfg.diag

def main(args):
    logger.info(args)

    """
    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)
    """

    X_train = torch.load('../data/HAR_data/X_train.pt')
    X_test = torch.load('../data/HAR_data/X_test.pt')

    y_train = torch.load('../data/HAR_data/y_train.pt')
    y_test = torch.load('../data/HAR_data/y_test.pt')

    # Input Data
    training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
    test_data_count = len(X_test)  # 2947 testing series
    n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = len(X_train[0][0])  # 9 input parameters per timestep


    # Some debugging info
    logger.debug("Some useful info to get an insight on dataset's shape and normalisation:")
    logger.debug("(X shape, y shape, every X's mean, every X's standard deviation)")
    logger.debug(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    logger.debug("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

    method = args.freezing_schema
    freezing_epochs = args.freezing_frequence-1
    init = args.init

    arch = cfg.arch
    net = LSTMModel()
    
    net.load_state_dict(torch.load(f'../models/LSTM/initializations/{init}.pt'))

    logger.info(f'Selected method: {method}')
    logger.info(f'Initialization in use: {init}')

    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    net = net.float()
    params = train(net, method, freezing_epochs, X_train, y_train, X_test, y_test, opt=opt,\
    criterion=criterion, epochs=epochs, clip_val=clip_val)

    """
    evaluate(net, X_test, y_test, criterion)
    plot(params['epochs'], params['train_loss'], params['test_loss'], 'loss', learning_rate, init)
    plot(params['epochs'], params['train_accuracy'], params['test_accuracy'], 'accuracy', learning_rate, init)
    plot(params['lr'], params['train_loss'], params['test_loss'], 'loss_lr', learning_rate)
    """

    save_data(method,freezing_epochs,init,params)

if __name__ == "__main__":
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)