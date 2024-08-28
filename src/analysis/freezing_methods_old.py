import torch
from torch import nn
import logging
from random import randint, sample

logger = logging.getLogger('Main Logger')

layerOut_array_VGG11 = torch.tensor([0.9, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.1])
layerOut_array_LSTM = torch.tensor([0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1])

def freeze(model, array):
    """ 
    Freezing procedure
    """

    layer_list = ('lstm','conv','linear')

    if not(torch.all(array == True)):
        logger.info('--------- FREEZING PROCEDURE ---------')
        # Layers freezing
        index = 0
        for children in model.children():
            if isinstance(children, nn.Sequential):
                for sub_children in children:
                    if any(substring.lower() in str(sub_children).lower() for substring in layer_list):
                        if array[index] == True:
                            for param in children.parameters():
                                param.requires_grad = False
                        index = index+1
            else:
                if any(substring.lower() in str(children).lower() for substring in layer_list):
                    if array[index] == True:
                        for param in children.parameters():
                            param.requires_grad = False
                    index = index+1

        logger.info('--------- FREEZING PROCEDURE TERMINATED ---------')

    else:
        logger.info('--------- FREEZING PROCEDURE ---------')
        for param in model.parameters():
            param.requires_grad = False
        logger.info('--------- FREEZING PROCEDURE TERMINATED ---------')
        return True

def deFreeze(model):
    """ 
    De-freezing procedure
    """

    logger.info('--------- DE-FREEZING PROCEDURE ---------')
    for param in model.parameters():
        param.requires_grad = True
    logger.info('--------- DE-FREEZING PROCEDURE TERMINATED ---------')

def normalizedGradientDifferenceFreezingProcedure(calculated_layer, current_epoch, total_epochs, model,\
frequence, freezing_epochs, num_layers, frozen, grad_dict, grad_dict_abs, defreeze=0, calculations=False, initial_percentage=10):
    """ 
    Take as input the index of the current epoch, the number of total epochs, the model being trained,
    the frequency with which you want to freeze and a sorted dictionary with a gradient tensor for each layer,
    and returns an index relating to the layer to freeze
    """
    
    if current_epoch >= (total_epochs // initial_percentage):
        if defreeze == 0:
            if calculations == True and (current_epoch-(total_epochs // initial_percentage)) % frequence == 0:
                calculations = False
    
                # Freezing decisions part
                logger.info('--------- PARAMETERS CALCULATION PROCEDURE ---------')
                freezingRate_array = torch.zeros(len(grad_dict))-1

                for layer_counter in range(len(grad_dict)):
                    if grad_dict[layer_counter] != None:
                        numerator_totalSummation = torch.sum(abs(grad_dict[layer_counter]))
                        denominator_totalSummation = torch.sum(grad_dict_abs[layer_counter])
                        freezingRate_array[layer_counter] = 1 - (numerator_totalSummation/denominator_totalSummation)
                    
                # Array standardization
                logger.info(f"Tensor before normalize:\n{freezingRate_array}")

                mean, std= torch.mean(freezingRate_array[freezingRate_array!=-1]), \
                    torch.std(freezingRate_array[freezingRate_array!=-1])
                logger.info(f"Mean and Std before Normalize:\n{mean},{std}")

                freezingRate_array[freezingRate_array==-1] = float('-inf')
                standardized_freezingRate_array = freezingRate_array.clone()

                if not(torch.isnan(std)): standardized_freezingRate_array  = (standardized_freezingRate_array-mean)/std

                standardized_freezingRate_array[0] = freezingRate_array[0]
                standardized_freezingRate_array[layer_counter] = freezingRate_array[layer_counter]
                logger.info(f"Tensor after Normalize:\n{standardized_freezingRate_array}")

                # Maximum subarray sum
                cum_sum = torch.cumsum(standardized_freezingRate_array[standardized_freezingRate_array!=float('-inf')],dim=0)
                count = (standardized_freezingRate_array[standardized_freezingRate_array==float('-inf')].size(dim=0))
                calculated_layer = torch.argmax(cum_sum)+count

                logger.info(f'Cumulative sum array:\n{cum_sum}')
                logger.info(f'Calculated argmax: {calculated_layer+1}')
                logger.info('--------- PARAMETERS CALCULATION PROCEDURE TERMINATED ---------')
    
            if (current_epoch-(total_epochs // initial_percentage)) % frequence == 0:
                if frequence == 1:
                    deFreeze(model)
                    frozen = False
                array = torch.zeros(num_layers, dtype=torch.bool)
                array[:calculated_layer+1] = True

                frozen = freeze(model,array)
                defreeze = freezing_epochs

        else:
            defreeze = defreeze-1
            if defreeze == 0:
                deFreeze(model)
                frozen = False
    
    #return freezingRate_array
    return calculated_layer, calculations, defreeze, frozen

def gradientNormChangeFreezingProcedure(current_epoch, total_epochs, model, frequence, step, grad_dict):
    """ 
    Takes as input the index of the current epoch, the number of total epochs, the model being trained,
    the frequency with which you want to freeze, the frequency with which you want to control changes in the gradient
    is a sorted dictionary with a gradient tensor for each layer, and returns an index relative to the layer 
    to freeze
    """

    layer_list = ('conv','conv','linear')
    
    if current_epoch % frequence == 0:
        # Freezing decisions part
        if current_epoch>=0: #(total_epochs // 10):
            logger.info('--------- FREEZING PROCEDURE ---------')
            total_number_iterations = grad_dict[len(grad_dict)-1].size()[0]
            frozen_layer = 0
            freeze = False
            for children in model.children():
                if isinstance(children, nn.Sequential):
                    for sub_children in children:
                        if any(substring.lower() in str(sub_children).lower() for substring in layer_list):
                            for param in sub_children.parameters():
                                if not(param.requires_grad):
                                    freeze = True
                            if freeze == True: 
                                frozen_layer = frozen_layer+1
                                freeze = False
                else:
                    if any(substring.lower() in str(children).lower() for substring in layer_list):
                        for param in children.parameters():
                            if not(param.requires_grad):
                                freeze = True
                        if freeze == True: 
                            frozen_layer = frozen_layer+1
                            freeze = False

            # Array utilizzato per il metodo
            gradient_norm_array = torch.zeros(len(grad_dict))+float('inf')
            # Arrays utilizzati per i plot
            gradient_difference_norm_change_array = torch.zeros([7,(((total_number_iterations-2)//step)+1)])
            gradient_norm_difference_change_array = torch.zeros([7,(((total_number_iterations-2)//step)+1)])
            for i in range(frozen_layer,len(grad_dict)):
                somma = 0
                z = 0
                for j in range(1,total_number_iterations,step):
                    previous_iteration_norm = torch.norm(grad_dict[i][j-1])
                    gradient_norm_change = (abs(previous_iteration_norm-torch.norm(grad_dict[i][j])))/previous_iteration_norm
                    gradient_norm_change2 = (torch.norm(grad_dict[i][j-1]-grad_dict[i][j]))
                    gradient_difference_norm_change_array[i,z] = gradient_norm_change
                    gradient_norm_difference_change_array[i,z] = gradient_norm_change2
                    z = z+1
                    somma = somma + gradient_norm_change
                gradient_norm_array[i] = somma

            return gradient_difference_norm_change_array, gradient_norm_difference_change_array
        
            """gradient_norm_array = gradient_norm_array/(((total_number_iterations-2)//step)+1)
            print('Gradient norm change array:')
            print(gradient_norm_array)

            if (torch.argmin(gradient_norm_array) == frozen_layer):
                print('Freeze layer: '+ str(frozen_layer+1))
                print()
                return frozen_layer
            print()
            """

def randomSequentialFreezing(calculated_layer, current_epoch, total_epochs, model, frequence, freezing_epochs,\
num_layers, frozen, defreeze=0, initial_percentage=10):
    """ 
    Take as input the index of the current epoch, the number of total epochs, the model being trained and
    the frequency with which you want to freeze and returns a random index relating to the layer to be frozen
    """

    if current_epoch >= (round(total_epochs / initial_percentage)):
        if defreeze == 0:
            if (current_epoch-round(total_epochs / initial_percentage)) % frequence == 0:
                if frequence == 1:
                    deFreeze(model)
                    frozen = False
                logger.info('--------- PARAMETERS CALCULATION PROCEDURE ---------')
                # Freezing decisions part
                calculated_layer = randint(0, num_layers-1)
                logger.info(f'Calculated argmax: {calculated_layer+1}')
                logger.info('--------- PARAMETERS CALCULATION PROCEDURE TERMINATED ---------')
                array = torch.zeros(num_layers, dtype=torch.bool)
                array[:calculated_layer+1] = True
                frozen = freeze(model,array)
                defreeze = freezing_epochs
        else:
            defreeze = defreeze-1
            if defreeze == 0:
                deFreeze(model)
                frozen = False

    return calculated_layer, defreeze, frozen

def scatteredFreezing(calculated_layer, current_epoch, total_epochs, model, frequence, freezing_epochs,\
num_layers, frozen, defreeze=0, calculations=False, initial_percentage=10):
    """ 
    Take as input the index of the current epoch, the number of total epochs, the model being trained and
    the frequency with which you want to freeze and returns a random index relating to the layer to be frozen
    """

    if defreeze == 0:
        if calculations == True:
            logger.info('--------- PARAMETERS CALCULATION PROCEDURE ---------')
            # Freezing decisions part
            indices = sample(range(0, num_layers), calculated_layer+1)
            array = torch.zeros(num_layers, dtype=torch.bool)
            array[indices] = True
            logger.info('--------- PARAMETERS CALCULATION PROCEDURE TERMINATED ---------')
            
        if current_epoch >= (total_epochs // initial_percentage) and (current_epoch-(total_epochs // initial_percentage)) % frequence == 0:
            if frequence == 1:
                deFreeze(model)
                frozen = False
            frozen = freeze(model,array)
            defreeze = freezing_epochs
    else:
        defreeze = defreeze-1
        if defreeze == 0:
            deFreeze(model)
            frozen = False

    return defreeze, frozen

def randomScatteredFreezing(calculated_layer, current_epoch, total_epochs, model, frequence, freezing_epochs,\
num_layers, frozen, defreeze=0, initial_percentage=10):
    """ 
    Take as input the index of the current epoch, the number of total epochs, the model being trained and
    the frequency with which you want to freeze and returns a random index relating to the layer to be frozen
    """

    if current_epoch >= (total_epochs // initial_percentage):
        if defreeze == 0:
            if (current_epoch-(total_epochs // initial_percentage)) % frequence == 0:
                if frequence == 1:
                    deFreeze(model)
                    frozen = False
                logger.info('--------- PARAMETERS CALCULATION PROCEDURE ---------')
                # Freezing decisions part
                calculated_layer = randint(1, num_layers)
                logger.info(f'Calculated argmax: {calculated_layer}')
                indices = sample(range(0, num_layers), calculated_layer)
                array = torch.zeros(num_layers, dtype=torch.bool)
                array[indices] = True
                logger.info('--------- PARAMETERS CALCULATION PROCEDURE TERMINATED ---------')
                frozen = freeze(model,array)
                defreeze = freezing_epochs
                
        else:
            defreeze = defreeze-1
            if defreeze == 0:
                deFreeze(model)
                frozen = False

    return defreeze, frozen

def layerOut_eFreeze(current_epoch, total_epochs, model, frequence, freezing_epochs, num_layers, frozen,\
defreeze=0, initial_percentage=10):
    """ 
    Implementation of the LayerOut - eFreeze procedure
    """
    
    if current_epoch >= (total_epochs // initial_percentage):
        if defreeze == 0:
            if (current_epoch-(total_epochs // initial_percentage)) % frequence == 0:
                if frequence == 1:
                    deFreeze(model)
                    frozen = False
                logger.info('--------- PARAMETERS CALCULATION PROCEDURE ---------')
                # Freezing decisions part
                x = torch.bernoulli(layerOut_array_LSTM)
                calculated_layer = torch.count_nonzero(x)
                logger.info(f'Calculated argmax: {calculated_layer}')
                array = x > 0
                logger.info('--------- PARAMETERS CALCULATION PROCEDURE TERMINATED ---------')
                frozen = freeze(model,array)
                defreeze = freezing_epochs
        else:
            defreeze = defreeze-1
            if defreeze == 0:
                deFreeze(model)
                frozen = False

    return defreeze, frozen

def layerOut_random(current_epoch, total_epochs, model, frequence, freezing_epochs, num_layers, frozen,\
defreeze=0, initial_percentage=10):
    """ 
    Implementation of the LayerOut - with random probability vector procedure
    """
    
    if current_epoch >= (total_epochs // initial_percentage):
        if defreeze == 0:
            if (current_epoch-(total_epochs // initial_percentage)) % frequence == 0:
                if frequence == 1:
                    deFreeze(model)
                    frozen = False
                logger.info('--------- PARAMETERS CALCULATION PROCEDURE ---------')
                # Freezing decisions part
                x = torch.bernoulli(torch.rand(num_layers))
                calculated_layer = torch.count_nonzero(x)
                logger.info(f'Calculated argmax: {calculated_layer}')
                array = x > 0
                logger.info('--------- PARAMETERS CALCULATION PROCEDURE TERMINATED ---------')
                frozen = freeze(model,array)
                defreeze = freezing_epochs
        else:
            defreeze = defreeze-1
            if defreeze == 0:
                deFreeze(model)
                frozen = False

    return defreeze, frozen