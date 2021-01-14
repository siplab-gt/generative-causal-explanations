"""
    train_mnist_classifier.py
    
    Trains MNIST/FMNIST classifier for use in figure-generating scripts.
"""

from load_mnist import *
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch
import time
import scipy.io as sio
import os

# --- options ---
dataset = 'mnist'             # 'mnist' or 'fmnist'
class_use = np.array([3,8])   # classes to select from dataset
batch_size = 64               # training batch size
c_dim = 1                     # number of channels in the input image
lr = 0.1                      # sgd learning rate
momentum = 0.5                # sgd momentum term
img_size = 28                 # size of each image dimension
gamma = 0.7                   # adam momentum term
epochs = 50                   # number of training epochs
save_folder_root = './pretrained_models'

class_use_str = np.array2string(class_use)
y_dim = class_use.shape[0]
newClass = range(0,y_dim)
test_size = 100
save_folder = os.path.join(save_folder_root, dataset + '_' + class_use_str[1:(len(class_use_str)-1):2] + '_classifier')


# --- load data ---
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if dataset == 'mnist':
    trX, trY, tridx = load_mnist_classSelect('train',class_use,newClass)
    vaX, vaY, vaidx = load_mnist_classSelect('val',class_use,newClass)
    teX, teY, teidx = load_mnist_classSelect('test',class_use,newClass)
elif dataset == 'fmnist':
    trX, trY, tridx = load_fashion_mnist_classSelect('train',class_use,newClass)
    vaX, vaY, vaidx = load_fashion_mnist_classSelect('val',class_use,newClass)
    teX, teY, teidx = load_fashion_mnist_classSelect('test',class_use,newClass)
else:
    print('dataset must be ''mnist'' or ''fmnist''!')


# --- train ---
batch_idxs = len(trX) // batch_size    
batch_idxs_val = len(vaX) // test_size    
ce_loss = nn.CrossEntropyLoss()
#
from models.CNN_classifier import CNN
classifier = CNN(y_dim).to(device)
optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=momentum)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
#
loss_total = np.zeros((epochs*batch_idxs))
test_loss_total = np.zeros((epochs))
percent_correct = np.zeros((epochs))
start_time = time.time()
counter = 0
for epoch in range(0,epochs):
    for idx in range(0, batch_idxs):
        batch_labels = torch.from_numpy(trY[idx*batch_size:(idx+1)*batch_size]).long().to(device)
        batch_images = trX[idx*batch_size:(idx+1)*batch_size]
        batch_images_torch = torch.from_numpy(batch_images)
        batch_images_torch = batch_images_torch.permute(0,3,1,2).float()
        batch_images_torch = batch_images_torch.to(device)
        
        optimizer.zero_grad()
        prob_output,output = classifier(batch_images_torch)
        loss = ce_loss(output,batch_labels)
        loss.backward()
        optimizer.step()
        
        loss_total[counter] = loss.item()
        counter = counter+1
        
        print ("[Train Epoch %d/%d] [Batch %d/%d] time: %4.4f [loss: %f]" % (epoch, epochs, idx, batch_idxs,time.time() - start_time,
                                                             loss.item()))
    # compute validation loss
    test_loss = 0.0
    correct = 0
    for idx in range(0, batch_idxs_val):
        val_labels = torch.from_numpy(vaY[idx*test_size:(idx+1)*test_size]).long().to(device)
        val_images = vaX[idx*test_size:(idx+1)*test_size]
        val_images_torch = torch.from_numpy(val_images)
        val_images_torch = val_images_torch.permute(0,3,1,2).float()
        val_images_torch = val_images_torch.to(device)
        
        prob_output_val,output_val = classifier(val_images_torch)
        pred = prob_output_val.argmax(dim=1)
        test_loss +=  ce_loss(output_val,val_labels)
        correct += pred.eq(val_labels.view_as(pred)).sum().item()/float(test_size)
    test_loss = test_loss/batch_idxs_val
    percent_correct[epoch] = 100.0*correct/batch_idxs_val
    print ("[Test Epoch %d/%d] [loss: %f] [corr: %f]" % (epoch, epochs, test_loss.item(), percent_correct[epoch]))
    test_loss_total[epoch] =  test_loss.item()
    scheduler.step()
    
    torch.save({
        'step': counter,
        'epoch': epoch,
        'batch': idx,
        'model_state_dict_classifier': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_total,
        }, os.path.join(save_folder, 'model.pt'))
    sio.savemat(os.path.join(save_folder, 'training-info.mat'),
        {'loss_total'      : loss_total[:counter],
         'percent_correct' : percent_correct[:epoch],
         'test_loss_total' : test_loss_total[:epoch],
         'class_use'       : class_use,
         'batch_size'      : batch_size,
         'c_dim'           : c_dim,
         'lr'              : lr,
         'momentum'        : momentum,
         'img_size'        : img_size,
         'gamma'           : gamma,
         'epochs'          : epochs})
