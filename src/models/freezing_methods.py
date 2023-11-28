import torch
from torch import nn
import logging

logger = logging.getLogger('Main Logger')

def normalizedGradientDifferenceFreezingProcedure(current_epoch, total_epochs, model, frequence, grad_dict, grad_dict_abs):
    """ 
    Prende in input l'indice dell'epoca corrente, il numero di epoche totali, il modello in training,
    la frequenza con cui si vuole freezare e un dizionario ordinato con un tensore gradiente per ogni layer,
    e ritorna un indice relativo al layer da freezare (da 0 a n.layer-1)
    """

    layer_list = ('conv','linear')
    
    if current_epoch % frequence == 0:
        # Freezing decisions part
        if current_epoch>=0: #(total_epochs // 10):
            logger.info('--------- FREEZING PROCEDURE ---------')
            freezingRate_array = torch.zeros(len(grad_dict))-1
            layer_counter = -1

            for children in model.children():
                if isinstance(children, nn.Sequential):
                    for sub_children in children:
                        if any(substring.lower() in str(sub_children).lower() for substring in layer_list):
                            layer_counter = layer_counter+1
                            if grad_dict[layer_counter] != None:
                                numerator_totalSummation = torch.sum(abs(grad_dict[layer_counter]))
                                denominator_totalSummation = torch.sum(grad_dict_abs[layer_counter])
                    
                                freezingRate_array[layer_counter] = 1 - (numerator_totalSummation/denominator_totalSummation)
                           
                else:
                    if any(substring.lower() in str(children).lower() for substring in layer_list):
                        layer_counter = layer_counter+1
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
            n = torch.argmax(cum_sum)+count
        
            logger.info(f'Cumulative sum array: {cum_sum}')
            logger.info(f'Calculated argmax: {n}')
            return freezingRate_array #n, 

def gradientNormChangeFreezingProcedure(current_epoch, total_epochs, model, frequence, step, grad_dict):
    """ 
    Prende in input l'indice dell'epoca corrente, il numero di epoche totali, il modello in training,
    la frequenza con cui si vuole freezare, la frequenza con cui si vogliono controllare i cambiamenti nel gradiente
    e un dizionario ordinato con un tensore gradiente per ogni layer, e ritorna un indice relativo al layer 
    da freezare (da 0 a n.layer-1)    
    """

    layer_list = ('conv','linear')
    
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