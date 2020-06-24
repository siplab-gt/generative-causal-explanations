#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
   
    def __init__(self,y_dim):
        """
        Initialize classifier
        Inputs:
        - y_dim : number of classes
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, y_dim)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        
    def forward(self, x):
        """
        Perform classification using the CNN classifier
        
        Inputs:
        - x : input data sample
        
        Outputs:
        - output: unnormalized output
        - prob_out: probability output
        """
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        output = self.fc3(x)
        prob_out = F.softmax(output)

        
        return prob_out,output