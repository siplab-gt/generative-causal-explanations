#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sklearn.decomposition
import sklearn.linear_model
from load_mnist import *
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch
import argparse
import time
import scipy.io as sio
import os
import torchvision.transforms as transforms
import torchvision.models as models
from imagenet_zebra_gorilla_dataloader import Imagenet_Gor_Zeb
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='imagenet_zeb_gor', help='folder name')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--c_dim', type=int, default=1, help='number of color channels in the input image')
parser.add_argument('--lr', type=float, default=0.001, help='SGD: learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD: momentum term')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--gamma', type=float, default=0.7, help='adam: momentum term')

opt = parser.parse_args()
print(opt)

batch_size = opt.batch_size
c_dim = opt.c_dim
lr = opt.lr
y_dim = 2

test_size = 32

#save_folder = '/home/mnorko/Documents/Tensorflow/causal_vae/' + opt.model + '_batch' + str(batch_size)  + '_lr' + str(opt.lr) + '/'
save_folder = '/home/mnorko/Documents/Tensorflow/causal_vae/results/' + opt.model + '_batch' + str(batch_size) + '_lr' + str(opt.lr) + '/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load data
transform_train = transforms.Compose([
    transforms.RandomCrop(128, padding=4,pad_if_needed = True),
    transforms.RandomHorizontalFlip(),
    transforms.Resize([32,32]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])


transform_test = transforms.Compose([
    transforms.CenterCrop(128),
    transforms.Resize([32,32]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])


file_open = open('imagenet_zebra_gorilla.pkl','rb')
train_fileType_2 = pickle.load(file_open)
train_imgName_2 = pickle.load(file_open)
train_imgLabel_2 = pickle.load(file_open)

val_fileType_2 =pickle.load(file_open)
val_imgName_2 = pickle.load(file_open)
val_imgLabel_2 =pickle.load(file_open)


file_open.close()
train_set = Imagenet_Gor_Zeb(train_imgName_2,train_imgLabel_2,train_fileType_2,transforms = transform_train)
datasetLen = len(train_set)
trainloader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,shuffle= True)
val_set = Imagenet_Gor_Zeb(val_imgName_2,val_imgLabel_2,val_fileType_2,transforms = transform_test) 
valloader = torch.utils.data.DataLoader(val_set,batch_size = test_size,shuffle= True)

dataiter = iter(trainloader)
dataiter_val = iter(valloader)
val_inputs_torch,val_labels = dataiter_val.next()
#sample_inputs_torch.to(device)
val_inputs_torch = val_inputs_torch.to(device)
vak_labels = val_labels.to(device)

 
ce_loss = nn.CrossEntropyLoss()

from cnnClassifierModel_imagenet import CNN
classifier = CNN(y_dim).to(device)
optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=opt.momentum)
scheduler = StepLR(optimizer, step_size=1, gamma=opt.gamma)

batch_idxs = datasetLen // opt.batch_size  
batch_idxs_val = len(valloader) // test_size 

loss_total = np.zeros((opt.epochs*batch_idxs))
test_loss_total = np.zeros((opt.epochs))
percent_correct = np.zeros((opt.epochs))
start_time = time.time()
counter = 0
for epoch in range(0,opt.epochs):
    for i, data in enumerate(trainloader,0):
        inputs,labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        prob_output,output = classifier(inputs)
        loss = ce_loss(output,labels)
        loss.backward()
        optimizer.step()
        
        loss_total[counter] = loss.item()
        counter = counter+1
        
        print ("[Train Epoch %d/%d] [Batch %d/%d] time: %4.4f [loss: %f]" % (epoch, opt.epochs, i,batch_idxs,time.time() - start_time,
                                                             loss.item()))
    #Get validation loss
    test_loss = 0.0
    correct = 0
    count_val = 0
    for jj, data_val in enumerate(valloader,0):       
        val_inputs_torch,val_labels = data_val
        val_inputs_torch = val_inputs_torch.to(device)
        val_labels = val_labels.to(device)
        prob_output_val,output_val = classifier(val_inputs_torch)
        pred = prob_output_val.argmax(dim=1)
        test_loss +=  ce_loss(output_val,val_labels)
        correct += pred.eq(val_labels.view_as(pred)).sum().item()/float(test_size)
        count_val = count_val+1
    test_loss = test_loss/batch_idxs_val
    percent_correct[epoch] = 100.0*correct/batch_idxs_val
    print('count_val: ' + str(count_val) + ' batch_val: ' + str(batch_idxs_val))
    print ("[Test Epoch %d/%d] [loss: %f] [corr: %f]" % (epoch, opt.epochs,test_loss.item(),percent_correct[epoch]))
    test_loss_total[epoch] =  test_loss.item()   
    #scheduler.step()
    
    torch.save({
        'step': counter,
        'epoch': epoch,
        'batch': i,
        'model_state_dict_classifier': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_total,
        }, save_folder + 'network_batch' + str(opt.batch_size) + '.pt')
    sio.savemat(save_folder + 'lossVal.mat',{'loss_total':loss_total[:counter],'percent_correct':percent_correct[:epoch],'test_loss_total':test_loss_total[:epoch]});
