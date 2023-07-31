import torch
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable
from copy import deepcopy
import time
from tqdm import tqdm
import numpy as np
import random

# model training function (hopefully a general training function)
def train_model(x_train,y_train,x_test,y_test,model, criterion, optimizer,device,seed, num_epochs=25,batch_size = 64):
    since=time.time()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print ('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print ('-'*10)
        training_loss = single_batch_train(x_train,y_train,model,criterion,optimizer,device,seed,batch_size=batch_size)
        training_loss = np.average(training_loss)
        print('epoch: \t', epoch, '\t training loss: \t', training_loss)
        # now evaluate the model on validation set
        epoch_acc = eval_model(x_test,y_test,model,criterion,device,batch_size=batch_size)

        # deep copy the model
        if epoch_acc>best_acc:
            best_acc = epoch_acc
            best_model_wts = deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model,best_acc

def eval_model(x_test,y_test,model,criterion,device,batch_size = 64):
    since = time.time()
    model.eval()
    test_loss = [] # cost function error
    correct = 0.0
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    outputs = model(x_test)
    loss = criterion(outputs,y_test)
    test_loss.append(loss.item())
    _,preds = torch.max(outputs,1)
    correct += torch.sum(preds == y_test)

    test_loss = np.average(test_loss)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss,
        correct.float() / len(y_test)
    ))
    print ('time for the full set of accuraices',time.time()-since)
    return correct.float()/len(y_test)

def single_batch_train (x_train,y_train,model,criterion, optimizer,device,seed,batch_size = 64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model.train()
    permutation = torch.randperm(x_train.size()[0])
    training_loss = []
    for i in range(0,x_train.size()[0],batch_size):
        indices = permutation[i:i+batch_size]
        batch_x,batch_y = x_train[indices],y_train[indices]
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs,batch_y)

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return training_loss