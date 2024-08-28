import torch
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import config as cfg
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import data.data_HAR_file as df
import logging

logger = logging.getLogger('Main Logger')

LABELS = df.LABELS
n_classes = cfg.n_classes
n_epochs_hold = cfg.n_epochs_hold
n_epochs_decay = cfg.batch_size - n_epochs_hold
epochs = cfg.n_epochs
# Define function to generate batches of a particular size

def extract_batch_size(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch = np.empty(shape)

    for i in range(batch_size):
        index = ((step - 1) * batch_size + i) % len(_train)
        batch[i] = _train[index]

    return batch

# Define to function to create one-hot encoding of output labels

def one_hot_vector(y_, n_classes=n_classes):
    # e.g.:
    # one_hot(y_=[[2], [0], [5]], n_classes=6):
    #     return [[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def getLRScheduler(optimizer):
    def lambdaRule(epoch):
        lr_l = 1.0 - max(0, epoch - n_epochs_hold) / float(n_epochs_decay + 1)
        return lr_l

    schedular = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdaRule)
    #schedular = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return schedular

def plot(x_arg, param_train, param_test, label, lr, init):
    plt.figure()
    plt.plot(x_arg, param_train, color='blue', label='train')
    plt.plot(x_arg, param_test, color='red', label='test')
    plt.legend()
    if (label == 'accuracy'):
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.title('Training and Test Accuracy', fontsize=20)
        plt.savefig('../plot/LSTM/Accuracy_' + str(epochs) + str(lr) + str(init) + '-' + str(cfg.batch_size) + '.png')
        plt.show()
    elif (label == 'loss'):
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training and Test Loss', fontsize=20)
        plt.savefig('../plot/LSTM/Loss_' + str(epochs) + str(lr) + str(init) + '-' + str(cfg.batch_size) + '.png')
        plt.show()
    else:
        plt.xlabel('Learning rate', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training loss and Test loss with learning rate', fontsize=20)
        plt.savefig('../plot/LSTM/Loss_lr_' + str(epochs) + str(lr) + '.png')
        plt.show()

def evaluate(net, X_test, y_test, criterion):
    test_batch = len(X_test)
    net.eval()
    test_h = net.init_hidden(test_batch)
    inputs, targets = torch.from_numpy(X_test), torch.from_numpy(y_test.flatten('F'))
    if (torch.cuda.is_available() ):
            inputs, targets = inputs.cuda(), targets.cuda()

    test_h = tuple([each.data for each in test_h])
    output = net(inputs.float(), test_h)
    test_loss = criterion(output, targets.long())
    top_p, top_class = output.topk(1, dim=1)
    targets = targets.view(*top_class.shape).long()
    equals = top_class == targets

    if (torch.cuda.is_available() ):
            top_class, targets = top_class.cpu(), targets.cpu()

    test_accuracy = torch.mean(equals.type(torch.FloatTensor))
    test_f1score = metrics.f1_score(top_class, targets, average='macro')


    logger.info(f"Final loss is: {test_loss.item():>8f}")
    logger.info(f"Final accuracy is: {(100*test_accuracy):>0.1f}%")
    logger.info(f"Final f1 score is: {test_f1score:>0.6f}")

    confusion_matrix = metrics.confusion_matrix(top_class, targets)
    logger.info("---------Confusion Matrix--------")
    logger.info(f'\n{confusion_matrix}')
    normalized_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
    plotConfusionMatrix(normalized_confusion_matrix)

def plotConfusionMatrix(normalized_confusion_matrix):
    plt.figure()
    plt.imshow(
        normalized_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("../plot/LSTM/confusion_matrix.png")
    plt.show()

def save_data():
    """
    Save accuracy, loss and time values
    """
    if method == 'SFS':
        tensor = torch.load(f'../plot/LSTM/comparisons/time/SFS/time_{freezing_epochs}.pt')
        tensor = torch.cat((tensor,torch.tensor([params['time']])),0)
        torch.save(tensor, f'../plot/LSTM/comparisons/time/SFS/time_{freezing_epochs}.pt')

        if freezing_epochs == 0:
            torch.save(params['test_accuracy'], f'../plot/LSTM/comparisons/accuracy/{init}/random_seq_accuracy.pt')
            torch.save(params['test_loss'], f'../plot/LSTM/comparisons/loss/{init}/random_seq_loss.pt')
        else:
            torch.save(params['test_accuracy'], f'../plot/LSTM/comparisons/accuracy/{init}/{freezing_epochs}/random_seq_accuracy.pt')
            torch.save(params['test_loss'], f'../plot/LSTM/comparisons/loss/{init}/{freezing_epochs}/random_seq_loss.pt')
    elif method == 'RSFS':
        tensor = torch.load(f'../plot/LSTM/comparisons/time/RSFS/time_{freezing_epochs}.pt')
        tensor = torch.cat((tensor,torch.tensor([params['time']])),0)
        torch.save(tensor, f'../plot/LSTM/comparisons/time/RSFS/time_{freezing_epochs}.pt')

        if freezing_epochs == 0:
            torch.save(params['test_accuracy'], f'../plot/LSTM/comparisons/accuracy/{init}/random_scat_accuracy.pt')
            torch.save(params['test_loss'], f'../plot/LSTM/comparisons/loss/{init}/random_scat_loss.pt')
        else:
            torch.save(params['test_accuracy'], f'../plot/LSTM/comparisons/accuracy/{init}/{freezing_epochs}/random_scat_accuracy.pt')
            torch.save(params['test_loss'], f'../plot/LSTM/comparisons/loss/{init}/{freezing_epochs}/random_scat_loss.pt')
    elif method == 'LES':
        tensor = torch.load(f'../plot/LSTM/comparisons/time/LES/time_{freezing_epochs}.pt')
        tensor = torch.cat((tensor,torch.tensor([params['time']])),0)
        torch.save(tensor, f'../plot/LSTM/comparisons/time/LES/time_{freezing_epochs}.pt')

        if freezing_epochs == 0:
            torch.save(params['test_accuracy'], f'../plot/LSTM/comparisons/accuracy/{init}/eFreeze_accuracy.pt')
            torch.save(params['test_loss'], f'../plot/LSTM/comparisons/loss/{init}/eFreeze_loss.pt')
        else:
            torch.save(params['test_accuracy'], f'../plot/LSTM/comparisons/accuracy/{init}/{freezing_epochs}/eFreeze_accuracy.pt')
            torch.save(params['test_loss'], f'../plot/LSTM/comparisons/loss/{init}/{freezing_epochs}/eFreeze_loss.pt')
    elif method == 'LRS':
        tensor = torch.load(f'../plot/LSTM/comparisons/time/LRS/time_{freezing_epochs}.pt')
        tensor = torch.cat((tensor,torch.tensor([params['time']])),0)
        torch.save(tensor, f'../plot/LSTM/comparisons/time/LRS/time_{freezing_epochs}.pt')

        if freezing_epochs == 0:
            torch.save(params['test_accuracy'], f'../plot/LSTM/comparisons/accuracy/{init}/layerOutRandom_accuracy.pt')
            torch.save(params['test_loss'], f'../plot/LSTM/comparisons/loss/{init}/layerOutRandom_loss.pt')
        else:
            torch.save(params['test_accuracy'], f'../plot/LSTM/comparisons/accuracy/{init}/{freezing_epochs}/layerOutRandom_accuracy.pt')
            torch.save(params['test_loss'], f'../plot/LSTM/comparisons/loss/{init}/{freezing_epochs}/layerOutRandom_loss.pt')
    else:
        tensor = torch.load(f'../plot/LSTM/comparisons/time/time.pt')
        tensor = torch.cat((tensor,torch.tensor([params['time']])),0)
        torch.save(tensor, f'../plot/LSTM/comparisons/time/time.pt')

        torch.save(params['test_accuracy'], f'../plot/LSTM/comparisons/accuracy/{init}/accuracy.pt')
        torch.save(params['test_loss'], f'../plot/LSTM/comparisons/loss/{init}/loss.pt')