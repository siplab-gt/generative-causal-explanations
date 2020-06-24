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
#import gif
import os

from util import *
from load_mnist import *
from mnist_test_fnc import sweepLatentFactors
from informationFlow import information_flow

# --- parameters ---
z_dim    = 8
c_dim    = 1
img_size = 28
class_use = np.array([3,8])
alpha_dim = 1
latent_sweep_vals = np.linspace(-2,2,25)
latent_sweep_plot = [0,4,8,12,16,20,24]
classifier_file = 'pretrained_models/mnist_38_classifier/model.pt'
vae_file = 'pretrained_models/mnist_38_vae/model.pt'

# --- initialize ---
class_use_str = np.array2string(class_use)    
y_dim = class_use.shape[0]
newClass = range(0,y_dim)
nsweep = len(latent_sweep_vals)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- load test data ---
test_size = 64
X, Y, tridx = load_mnist_classSelect('train',class_use,newClass)
vaX, vaY, vaidx = load_mnist_classSelect('val',class_use,newClass)
sample_inputs = vaX[0:test_size]
sample_labels = vaY[0:test_size]
sample_inputs_torch = torch.from_numpy(sample_inputs)
sample_inputs_torch = sample_inputs_torch.permute(0,3,1,2).float().to(device)     
ntrain = X.shape[0]

# --- load VAE ---
from VAEModel_CNN import Decoder, Encoder
checkpoint_vae = torch.load(vae_file, map_location=device)
encoder = Encoder(z_dim,c_dim,img_size).to(device)
decoder = Decoder(z_dim,c_dim,img_size).to(device)
encoder.load_state_dict(checkpoint_vae['model_state_dict_encoder'])
decoder.load_state_dict(checkpoint_vae['model_state_dict_decoder'])

# --- load classifier ---
from cnnClassifierModel import CNN
checkpoint_model = torch.load(classifier_file, map_location=device)
classifier = CNN(y_dim).to(device)
classifier.load_state_dict(checkpoint_model['model_state_dict_classifier'])


#%% generate latent factor sweep plot

sample_ind = np.concatenate((np.where(vaY == 0)[0][:1],
                             np.where(vaY == 1)[0][:1]))
cols = [[0.047,0.482,0.863],[1.000,0.761,0.039],[0.561,0.788,0.227]]
border_size = 0
nsamples = len(sample_ind)
latentsweep_vals = [-3., -2., -1., 0., 1., 2., 3.]
Xhats = np.zeros((z_dim,nsamples,len(latentsweep_vals),img_size,img_size,1))
yhats = np.zeros((z_dim,nsamples,len(latentsweep_vals)))
for isamp in range(nsamples):
    x = torch.from_numpy(np.expand_dims(vaX[sample_ind[isamp]],0))
    x_torch = x.permute(0,3,1,2).float().to(device)
    z = encoder(x_torch)[0][0].detach().numpy()
    for latent_dim in range(z_dim):
        for (ilatentsweep, latentsweep_val) in enumerate(latentsweep_vals):
            ztilde = z.copy()
            ztilde[latent_dim] += latentsweep_val
            xhat = decoder(torch.unsqueeze(torch.from_numpy(ztilde),0).to(device))
            yhat = np.argmax(classifier(xhat)[0].detach().numpy())
            img = 1.-xhat.permute(0,2,3,1).detach().numpy().squeeze()
            Xhats[latent_dim,isamp,ilatentsweep,:,:,0] = img
            yhats[latent_dim,isamp,ilatentsweep] = yhat

for isamp in range(nsamples):
    fig, axs = plt.subplots(z_dim, len(latentsweep_vals))
    for latent_dim in range(z_dim):
        for (ilatentsweep, latentsweep_val) in enumerate(latentsweep_vals):
            img = Xhats[latent_dim,isamp,ilatentsweep,:,:,0].squeeze()
            axs[latent_dim,ilatentsweep].imshow(img, cmap='gray', interpolation='nearest')
            axs[latent_dim,ilatentsweep].set_xticks([])
            axs[latent_dim,ilatentsweep].set_yticks([])
    if False:
        print('Exporting sample %d...' % (isamp))
        plt.savefig('./figs/fig_comparison_ours_samp%d.svg' % (isamp), bbox_inches=0)

print('Columns - latent values in sweep: ' + str(latentsweep_vals))
print('Rows - latent dimension')

