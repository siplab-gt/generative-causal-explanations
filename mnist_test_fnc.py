#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division
import time
import datetime
import re

import numpy as np
import scipy.io as sio
import scipy as sp
import scipy.linalg

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

import loss_functions
import causaleffect
import plotting
import util

import matplotlib.pyplot as plt
import os

def sweepLatentFactors(zeta,decoder,classifier,device,img_size,c_dim,y_dim,intervene_flag):
    numSamp = zeta.shape[0]
    z_dim = zeta.shape[1]
    t = np.arange(-3,3.01,0.25)
    imgOut = np.zeros((numSamp,z_dim,t.shape[0],img_size,img_size,c_dim))
    probOut = np.zeros((numSamp,z_dim,t.shape[0],y_dim))
    latentOut = np.zeros((numSamp,z_dim,t.shape[0],z_dim))
    for k in range(0,numSamp):
        for m in range(0,z_dim):
            zeta_use = zeta[k,:].detach().cpu().numpy()
            zeta_val = zeta[k,m].detach().cpu().numpy()
            zeta_use_torch = torch.unsqueeze(torch.from_numpy(zeta_use),0).to(device)
            count_t = 0
            for t_idx in t:
                if intervene_flag == True:
                    zeta_use_torch[0,m] = t_idx
                else:
                    zeta_use_torch[0,m] = zeta_val+t_idx
                Xhat = decoder(zeta_use_torch)
                yhat = classifier(Xhat)[0]
                Xhat = Xhat.permute(0,2,3,1).detach().cpu().numpy()
                imgOut[k,m,count_t,:,:,:] = Xhat
                probOut[k,m,count_t,:] = yhat.detach().cpu().numpy()
                latentOut[k,m,count_t,:] = zeta_use_torch.detach().cpu().numpy()
                count_t = count_t+1
    return imgOut, probOut, latentOut
    
