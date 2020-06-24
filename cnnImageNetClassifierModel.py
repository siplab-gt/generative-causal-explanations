#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
   
    def __init__(self,model,class_ids):
        """
        Initialize classifier
        Inputs:
        - y_dim : number of classes
        """
        super(CNN, self).__init__()
        self.model = model
        self.class_ids = class_ids
        
        
    def forward(self, x):
        """
        Perform classification using the CNN classifier
        
        Inputs:
        - x : input data sample
        
        Outputs:
        - output: unnormalized output
        - prob_out: probability output
        """
        x = torch.nn.functional.interpolate(x,scale_factor=4,mode = 'bilinear')
        class_out = self.model(x)
        class_select = class_out[:,self.class_ids]
        m = torch.nn.Softmax(dim = 1)
        prob_out = m(class_select)
        
        return prob_out,class_select
