# Define the architecture:
arch = {
	'name' : 'LSTM',
	'bidir' : False,
	'clip_val' : 10,
	'drop_prob' : 0.5,
	'n_epochs_hold' : 100,
	'n_layers' : 1,
	'learning_rate' : 0.0005,
	'weight_decay' : 0.001,
	'n_residual_layers' : 0,
	'n_highway_layers' : 1,
	'diag' : 'Architecure chosen is baseline LSTM',
	'save_file' : 'results_lstm.txt'
}

# This will set the values according to that architecture
bidir = arch['bidir']
clip_val = arch['clip_val']
drop_prob = arch['drop_prob']
n_epochs_hold = arch['n_epochs_hold']
n_layers = arch['n_layers']
learning_rate = arch['learning_rate']
weight_decay = arch['weight_decay']
n_highway_layers = arch['n_highway_layers']
n_residual_layers = arch['n_residual_layers']

# Diagnostics
diag = arch['diag']
save_file = arch['save_file']

# General settings:
n_classes = 6
n_input = 9
n_hidden = 32
batch_size = 256
n_epochs = 180