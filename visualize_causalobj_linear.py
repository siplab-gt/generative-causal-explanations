from __future__ import division
import time
import datetime
import re
from enum import Enum
import pickle

import numpy as np
import scipy.io as sio
import scipy as sp
import scipy.linalg

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch

from loss_functions import *
import causaleffect
import plotting
import util

import matplotlib.pyplot as plt
import gif
import os


# --- set test parameters ---
thetas_alpha = np.linspace(0, np.pi, 49)
thetas_beta = np.linspace(0, np.pi, 49)
params = {
    "batch_size"     : 100,
    "z_dim"          : 2,
    "z_dim_true"     : 2,
    "x_dim"          : 2,
    "y_dim"          : 1,
    "alpha_dim"      : 1,
    "ntrain"         : 5000,
    "Nbeta"          : 2500,
    "Nalpha"         : 500,
    "gamma"          : 0.001,
    "break_up_ce"    : False,
    "decoder_net"    : "linGauss",
    "classifier_net" : "oneHyperplane"}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- construct projection matrices ---
# 'true' orthogonal columns used to generate data from latent factors
Wsquare = np.identity(params["z_dim"])
W = Wsquare[:,:params["z_dim_true"]]
# columns of W corresponding to causal factors (alphas)
Wa = W[:,:params["alpha_dim"]]
# columns of W corresponding to noncausal factors (betas)
Wb = W[:,params["alpha_dim"]:]
# form projection matrices
PW  = util.formProjMat(W)
PWa = util.formProjMat(Wa)
PWb = util.formProjMat(Wb)
# convert to torch matrices
PW_torch  = torch.from_numpy(PW).float()
PWa_torch = torch.from_numpy(PWa).float()
PWb_torch = torch.from_numpy(PWb).float()


# --- construct data ---
Z = np.random.randn(params["ntrain"], params["z_dim_true"])
X = np.matmul(Z, W.T)


# --- initialize decoder ---
from linGaussModel import Decoder
decoder = Decoder(params["x_dim"], params["z_dim"])
decoder.apply(util.weights_init_normal)
What = Variable(torch.mul(torch.randn(params["x_dim"], params["z_dim"],
                                      dtype=torch.float),0.5), requires_grad=True)


# --- initialize classifier ---
from synthetic_classifiers import OneHyperplaneClassifier
classifier = OneHyperplaneClassifier(params["x_dim"],
                                     params["y_dim"],
                                     PWa_torch,
                                     a = Wa.reshape((1,2)))
classifier.apply(util.weights_init_normal)


# --- compute causal effects ---
CEs = np.zeros((len(thetas_alpha),len(thetas_beta),4))
for ia, theta_alpha in enumerate(thetas_alpha):
    for ib, theta_beta in enumerate(thetas_beta):
        print('Computing causal effect for alpha=%.2f (%d/%d), beta=%.2f (%d/%d)...' % 
              (theta_alpha,ia,len(thetas_alpha),theta_beta,ib,len(thetas_beta)))
        # form generative map for this (theta1, theta2)
        what1 = np.array([[np.cos(theta_alpha)],[np.sin(theta_alpha)]])
        what2 = np.array([[np.cos(theta_beta)],[np.sin(theta_beta)]])
        What = torch.from_numpy(np.hstack((what1,what2))).float()
        # sample-based estimate of causal effect
        CEs[ia,ib,0] = -causaleffect.ind_uncond(   params, decoder, classifier, device, What=What)[0]
        CEs[ia,ib,1] = -causaleffect.ind_cond(     params, decoder, classifier, device, What=What)[0]
        CEs[ia,ib,2] = -causaleffect.joint_uncond( params, decoder, classifier, device, What=What)[0]
        CEs[ia,ib,3] = -causaleffect.joint_cond(   params, decoder, classifier, device, What=What)[0]


# --- save results ---        
print('Done! Saving results...')
sio.savemat('results/visualize_causalobj_linear.mat',
            {'CEs' : CEs,
             'thetas_alpha' : thetas_alpha,
             'thetas_beta' : thetas_beta,
             'params' : params})
print('Done!')


#%% make debug plot (see visualize_causalobj_plot.m for plots in paper)

import numpy as np
import matplotlib.pyplot as plt
import plotting
import pickle

#with open('visualize_causalobj_onehyperplane_thetaalphabeta_results.pickle', 'rb') as f:
#    results = pickle.load(f)

clim = [np.min(results["CEs"]), np.max(results["CEs"])]

fig, axs = plt.subplots(1, 4, figsize=(20,5))

plt.axes(axs[0])
im = plotting.plotsurface(axs[0],
                      results["thetas_alpha"]/np.pi*180.,
                      results["thetas_beta"]/np.pi*180.,
                      results["CEs"][:,:,0],
                      clim)
axs[0].set_title('Causal effect\n(independent, conditional)')
plt.xticks(ticks=(0,45,90,135,180))
plt.yticks(ticks=(0,45,90,135,180))

plt.axes(axs[1])
im = plotting.plotsurface(axs[1],
                      results["thetas_alpha"]/np.pi*180.,
                      results["thetas_beta"]/np.pi*180.,
                      results["CEs"][:,:,1],
                      clim)
axs[1].set_title('Causal effect\n(independent, unconditional)')
plt.xticks(ticks=(0,45,90,135,180))
plt.yticks(ticks=(0,45,90,135,180))

plt.axes(axs[2])
im = plotting.plotsurface(axs[2],
                      results["thetas_alpha"]/np.pi*180.,
                      results["thetas_beta"]/np.pi*180.,
                      results["CEs"][:,:,2],
                      clim)
axs[2].set_title('Causal effect\n(joint, conditional)')
plt.xticks(ticks=(0,45,90,135,180))
plt.yticks(ticks=(0,45,90,135,180))

plt.axes(axs[3])
im = plotting.plotsurface(axs[3],
                      results["thetas_alpha"]/np.pi*180.,
                      results["thetas_beta"]/np.pi*180.,
                      results["CEs"][:,:,3],
                      clim)
axs[3].set_title('Causal effect\n(joint, unconditional)')
plt.xticks(ticks=(0,45,90,135,180))
plt.yticks(ticks=(0,45,90,135,180))

for i in range(4):
    axs[i].set_xlabel(r'$\theta(w_{\alpha})$')
    axs[i].set_ylabel(r'$\theta(w_{\beta})$')
plt.tight_layout(pad=3)
fig.colorbar(im, ax=axs[:], shrink=0.5)