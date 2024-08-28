import torch
from torch import nn
import numpy as np
from training.HAR_testing import test
from utils.Functions import extract_batch_size, getLRScheduler
import torch.nn.utils.clip_grad as clip_grad
from models.LSTM import LSTMModel, init_weights
from analysis.freezing_methods import randomSequentialFreezing, randomScatteredFreezing, layerOut_eFreeze, layerOut_random
import config as cfg
import logging
import time
import math
import wandb

logger = logging.getLogger('Main Logger')

def train(method, freezing_epochs, X_train, y_train, X_test, y_test, criterion, epochs=100, clip_val=15, config=None):
    with wandb.init(config=config):
        config = wandb.config

        learning_rate = config.lr
        epochs = config.epochs
        n_hidden = config.n_hidden
        batch_size = config.batch_size

        net = LSTMModel(n_hidden)
        net.apply(init_weights)

        opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
        net = net.float()

        logger.info('----------- LSTM TRAINING -----------')

        #sched = getLRScheduler(optimizer=opt)
        #if (train_on_gpu):
        if (torch.cuda.is_available() ):
            net.cuda()

        train_losses = []
        net.train()

        epoch_train_losses = []
        epoch_train_acc = []
        epoch_test_losses = []
        epoch_test_acc = []
        params = {
            'epochs' : [],
            'train_loss' : [],
            'test_loss' : [],
            'lr' : [],
            'train_accuracy' : [],
            'test_accuracy' : []
        }

        if method == 'sequential_freezing' or method == 'random_scattered_freezing' or method == 'eFreeze' or method == 'layerOut_random':
            # normalizedGradientDifferenceFreezingProcedure
            freezing_rate_values = torch.zeros([epochs,net.num_layers])
            last_lr = None
            initial_percentage = 10
            calculations = False
            calculated_layer = None
            #freezing_epochs = 1
            defreeze = 0
            frequence = freezing_epochs+1
            warm_up_epochs = epochs // initial_percentage

        frozen_net = False

        # Average frozen layers calculation
        frozen_layers_sum = 0
        active_layers = 0
        lstm_params = 0

        # Time measurement
        start_time = time.time()

        """
        before_l1_hh = net.lstm1.weight_hh_l0.clone()
        before_l1_ih = net.lstm1.weight_ih_l0.clone()
        before_l2_hh = net.lstm2.weight_hh_l0.clone()
        before_l2_ih = net.lstm2.weight_ih_l0.clone()
        before_l3_hh = net.lstm3.weight_hh_l0.clone()
        before_l3_ih = net.lstm3.weight_ih_l0.clone()
        before_l4_hh = net.lstm4.weight_hh_l0.clone()
        before_l4_ih = net.lstm4.weight_ih_l0.clone()
        before_fc1 = net.fc1.weight.clone()
        before_fc2 = net.fc2.weight.clone()
        before_fc3 = net.fc3.weight.clone()
        """

        for epoch in range(epochs):
            logger.info(f'Epoch {epoch+1}\n-------------------------------')
            train_losses = []
            step = 1

            h = net.init_hidden(batch_size)

            train_accuracy = 0
            train_len = len(X_train)

            if method == 'sequential_freezing' or method == 'random_scattered_freezing' or method == 'eFreeze' or method == 'layerOut_random':
                logger.info('PARAMETERS THAT REQUIRE GRADIENT:')
                for name, param in net.named_parameters():
                    if param.requires_grad:
                        logger.info(name)
                        active_layers = active_layers+1
                        if 'lstm' in name:
                            lstm_params = lstm_params+1
                
                frozen_layers_sum = frozen_layers_sum+net.num_layers-((lstm_params/4)+((active_layers-lstm_params)/2))
                active_layers = 0
                lstm_params = 0

            if frozen_net != True:
                while step * batch_size <= train_len:
                    batch_xs = extract_batch_size(X_train, step, batch_size)
                    # batch_ys = one_hot_vector(extract_batch_size(y_train, step, batch_size))
                    batch_ys = extract_batch_size(y_train, step, batch_size)

                    inputs, targets = torch.from_numpy(batch_xs), torch.from_numpy(batch_ys.flatten('F'))
                    #if (train_on_gpu):
                    if (torch.cuda.is_available() ):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    h = tuple([each.data for each in h])
                    opt.zero_grad()

                    output = net(inputs.float(), h)
                    # print("lenght of inputs is {} and target value is {}".format(inputs.size(), targets.size()))
                    train_loss = criterion(output, targets.long())
                    train_losses.append(train_loss.item())

                    top_p, top_class = output.topk(1, dim=1)
                    equals = top_class == targets.view(*top_class.shape).long()
                    train_accuracy += torch.mean(equals.type(torch.FloatTensor))
                    equals = top_class

                    train_loss.backward()
                    clip_grad.clip_grad_norm_(net.parameters(), clip_val)
                    opt.step()

                    wandb.log({'loss': train_loss.data.item()})

                    if (step-1) % 20 == 0:
                        logger.info(f'loss: {train_loss:>7f}  [{step*batch_size:>5d}/{train_len:>5d}]')

                    step += 1
                train_accuracy_avg = train_accuracy/(step-1)
                train_loss_avg = np.mean(train_losses)
            else:
                train_accuracy_avg = 0
                train_loss_avg = 0

            p = opt.param_groups[0]['lr']
            params['lr'].append(p)
            params['epochs'].append(epoch)
            #sched.step()
            epoch_train_losses.append(train_loss_avg)
            epoch_train_acc.append(train_accuracy_avg)
            test_loss, test_f1score, test_accuracy = test(net, X_test, y_test, criterion, test_batch=len(X_test))
            epoch_test_losses.append(test_loss)
            epoch_test_acc.append(test_accuracy)
            logger.info(f'Test Error: \nAccuracy: {(100*test_accuracy):>0.1f}%, Avg Loss: {test_loss:>8f}, F1 score: {test_f1score:>0.6f} \n')

            if method == 'eFreeze':
                # layerOut eFreeze
                defreeze, frozen_net = layerOut_eFreeze(epoch+1,epochs,net,frequence,freezing_epochs,net.num_layers,frozen_net,defreeze)
            elif method == 'layerOut_random':
                # layerOut with random generation of freeze probability vector
                defreeze, frozen_net = layerOut_random(epoch+1,epochs,net,frequence,freezing_epochs,net.num_layers,frozen_net,defreeze)
            elif method == 'random_scattered_freezing':
                # randomScatteredFreezingProcedure
                defreeze, frozen_net = \
                randomScatteredFreezing\
                (calculated_layer,epoch+1,epochs,net,frequence,freezing_epochs,net.num_layers,frozen_net,defreeze)
            elif method == 'sequential_freezing':
                # sequentialFreezingProcedure
                calculated_layer, defreeze, frozen_net = \
                randomSequentialFreezing\
                (calculated_layer,epoch+1,epochs,net,frequence,freezing_epochs,net.num_layers,frozen_net,defreeze)

            """
            logger.info(f'lstm1 hh: {torch.equal(net.lstm1.weight_hh_l0, before_l1_hh)}')
            logger.info(f'lstm1 ih: {torch.equal(net.lstm1.weight_ih_l0, before_l1_ih)}')
            logger.info(f'lstm2 hh: {torch.equal(net.lstm2.weight_hh_l0, before_l2_hh)}')
            logger.info(f'lstm2 ih: {torch.equal(net.lstm2.weight_ih_l0, before_l2_ih)}')
            logger.info(f'lstm3 hh: {torch.equal(net.lstm3.weight_hh_l0, before_l3_hh)}')
            logger.info(f'lstm3 ih: {torch.equal(net.lstm3.weight_ih_l0, before_l3_ih)}')
            logger.info(f'lstm4 hh: {torch.equal(net.lstm4.weight_hh_l0, before_l4_hh)}')
            logger.info(f'lstm4 ih: {torch.equal(net.lstm4.weight_ih_l0, before_l4_ih)}')
            logger.info(f'fc1: {torch.equal(net.fc1.weight, before_fc1)}')
            logger.info(f'fc2: {torch.equal(net.fc2.weight, before_fc2)}')
            logger.info(f'fc3: {torch.equal(net.fc3.weight, before_fc3)}')

            before_l1_hh = net.lstm1.weight_hh_l0.clone()
            before_l1_ih = net.lstm1.weight_ih_l0.clone()
            before_l2_hh = net.lstm2.weight_hh_l0.clone()
            before_l2_ih = net.lstm2.weight_ih_l0.clone()
            before_l3_hh = net.lstm3.weight_hh_l0.clone()
            before_l3_ih = net.lstm3.weight_ih_l0.clone()
            before_l4_hh = net.lstm4.weight_hh_l0.clone()
            before_l4_ih = net.lstm4.weight_ih_l0.clone()
            before_fc1 = net.fc1.weight.clone()
            before_fc2 = net.fc2.weight.clone()
            before_fc3 = net.fc3.weight.clone()
            """

        end_time = time.time()
        logger.info("Done!")
        logger.info(f'Total training time: {end_time-start_time}')
        if method == 'sequential_freezing' or method == 'random_scattered_freezing' or method == 'eFreeze' or method == 'layerOut_random':
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

        params['train_loss'] = epoch_train_losses
        params['test_loss'] = epoch_test_losses
        params['train_accuracy'] = epoch_train_acc
        params['test_accuracy'] = epoch_test_acc
        #return params