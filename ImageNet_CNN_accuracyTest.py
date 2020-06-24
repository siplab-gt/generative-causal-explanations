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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

from cnnImageNetClassifierModel import CNN
classImagenetIdx = [366,340]
classifier_model = models.vgg16_bn(pretrained=True)
classifier = CNN(classifier_model,classImagenetIdx).to(device)
        


test_size = 32

#save_folder = '/home/mnorko/Documents/Tensorflow/causal_vae/' + opt.model + '_batch' + str(batch_size)  + '_lr' + str(opt.lr) + '/'
save_folder = '/home/mnorko/Documents/Tensorflow/causal_vae/results/classificationOutput_imageNet_32Resize/'
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



batch_idxs = datasetLen // opt.batch_size  
batch_idxs_val = len(valloader) // test_size 

loss_total = np.zeros((opt.epochs*batch_idxs))
test_loss_total = np.zeros((opt.epochs))
percent_correct = np.zeros((opt.epochs))
start_time = time.time()
counter = 0
correct = 0
total_pred = np.zeros((datasetLen,2))
total_labels = np.zeros((datasetLen))
for i, data in enumerate(trainloader,0):
    inputs,labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    prob_output,output = classifier(inputs)
    pred = prob_output.argmax(dim=1)
    total_pred[counter*opt.batch_size:(counter+1)*opt.batch_size,:] = prob_output.detach().cpu().numpy()
    total_labels[counter*opt.batch_size:(counter+1)*opt.batch_size] = labels.detach().cpu().numpy()
    correct += pred.eq(labels.view_as(pred)).sum().item()/float(batch_size)
    
    counter = counter+1
    
    print ("[Batch %d/%d] time: %4.4f" % (i,batch_idxs,time.time() - start_time))
    
percent_correct = 100.0*correct/counter


sio.savemat(save_folder + 'perCorrect.mat',{'percent_correct':percent_correct,'total_pred':total_pred,'total_labels':total_labels});
