import torch
from torch import nn
import torch.nn.functional as F
import config as cfg
from torch.autograd import Variable
import logging

logger = logging.getLogger('Main Logger')

n_classes = cfg.n_classes
n_input = cfg.n_input
n_hidden = cfg.n_hidden
drop_prob = cfg.drop_prob
n_layers = cfg.n_layers
batch_size = cfg.batch_size
bidir = cfg.bidir
n_residual_layers = cfg.n_residual_layers
n_highway_layers = cfg.n_highway_layers

class LSTMModel(nn.Module):

    def __init__(self, n_input=n_input, n_hidden=n_hidden, n_layers=n_layers,
                 n_classes=n_classes, drop_prob=drop_prob):
        super(LSTMModel, self).__init__()

        # # LSTM layers
        self.n_layers = n_layers
        # # total layers
        self.num_layers = 7
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.n_input = n_input

        self.lstm1 = nn.LSTM(n_input, n_hidden, self.n_layers)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, self.n_layers)
        self.lstm3 = nn.LSTM(n_hidden, n_hidden, self.n_layers)
        self.lstm4 = nn.LSTM(n_hidden, n_hidden, self.n_layers)

        self.fc1 = nn.Linear(n_hidden, n_hidden//2)
        self.fc2 = nn.Linear(n_hidden//2, n_hidden//4)
        self.fc3 = nn.Linear(n_hidden//4, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        x, hidden1 = self.lstm1(x, hidden)
        x, hidden2 = self.lstm2(x, hidden1)
        x, hidden3 = self.lstm3(x, hidden2)
        x, hidden4 = self.lstm4(x, hidden3)
        #x = self.dropout(x)
        out = x[-1]
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = F.softmax(out, dim=1)

        return out

    def init_hidden(self, batch_size):
        ''' Initialize hidden state'''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # if (train_on_gpu):
        if (torch.cuda.is_available() ):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

def init_weights(m):

    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)