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

import loss_functions
import causaleffect
import plotting
import util

import matplotlib.pyplot as plt
import gif
import os


# --- initialization ---
thetas_alpha1  = np.linspace(0, np.pi, 49) # 9+8n for samples at multiples of pi/8
thetas_alpha2  = np.linspace(0, np.pi, 49) #np.pi*np.array([0.0, 0.25, 0.375, 0.50, 0.75])
params = {
    "z_dim"          : 2,
    "z_dim_true"     : 2,
    "x_dim"          : 2,
    "y_dim"          : 2,
    "alpha_dim"      : 2,
    "ntrain"         : 5000,
    "Nbeta"          : 2500, # TODO 2500
    "Nalpha"         : 500, # TODO 500
    "ksig"           : 100,
    "gamma"          : 0.001,
    "break_up_ce"    : False,
    "theta1"         : 0.*np.pi,        # classifier decision boundary angle 1
    "theta2"         : 0.5*np.pi,       # classifier decision boundary angle 2
    "decoder_net"    : "linGauss"}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datasize = (len(thetas_alpha1),len(thetas_alpha2))
data = {'loglik'         : np.zeros(datasize),
        'ce_iu'          : np.zeros(datasize),
        'ce_ic'          : np.zeros(datasize),
        'ce_ju'          : np.zeros(datasize),
        'ce_jc'          : np.zeros(datasize),
        'nentropy'       : np.zeros(datasize),
        'expcondentropy' : np.zeros(datasize),
        'adj_iu_ic'      : np.zeros(datasize),
        'adj_iu_ju'      : np.zeros(datasize),
        'adj_ju_jc'      : np.zeros(datasize),
        'adj_ic_jc'      : np.zeros(datasize)}


# --- construct projection matrices ---
W = np.array([[np.cos(params["theta1"]), np.cos(params["theta2"])],
              [np.sin(params["theta1"]), np.sin(params["theta2"])]])
w1 = W[:,:1]
w2 = W[:,1:]
# form projection matrices
PW  = util.formProjMat(W)
Pw1 = util.formProjMat(w1)
Pw2 = util.formProjMat(w2)
# convert to torch matrices
PW_torch  = torch.from_numpy(PW).float()
Pw1_torch = torch.from_numpy(Pw1).float()
Pw2_torch = torch.from_numpy(Pw2).float()


# --- construct data ---
Z = np.random.randn(params["ntrain"], params["z_dim_true"])
X = np.matmul(Z, W.T)


# --- initialize decoder ---
from linGaussModel import Decoder
decoder = Decoder(params["x_dim"], params["z_dim"])
decoder.apply(util.weights_init_normal)


# --- initialize classifier ---
from synthetic_classifiers import TwoHyperplaneClassifier
classifier = TwoHyperplaneClassifier(params["x_dim"],
                                     params["y_dim"],
                                     Pw1_torch,
                                     Pw2_torch,
                                     a1 = w1.reshape((1,2)),
                                     a2 = w2.reshape((1,2)),
                                     ksig = params["ksig"])
classifier.apply(util.weights_init_normal)
What = Variable(torch.mul(torch.randn(params["x_dim"], params["z_dim"],
                                      dtype=torch.float),0.5), requires_grad=True)


# --- visualize classifier for debug ---
visualizeClassifier = False
if visualizeClassifier:
    x_sweep = np.linspace(-2,2,100)
    y_sweep = np.linspace(-2,2,100)
    Yhat = np.zeros((len(x_sweep),len(y_sweep)))
    for ix, x in enumerate(x_sweep):
        for iy, y in enumerate(y_sweep):
            xhat = torch.tensor([x,y]).unsqueeze(0).float()
            Yhat[ix,iy] = classifier(xhat)[0].detach().numpy()[0][0]
    fig, axs = plt.subplots(1, 1, figsize=(5,5))
    im = plotting.plotsurface(axs, x_sweep, y_sweep, Yhat, clim=[0, 1])


#%%

for ia1, theta_alpha1 in enumerate(thetas_alpha1):
    for ia2, theta_alpha2 in enumerate(thetas_alpha2):
        print('Computing causal effect for alpha1=%.2f (%d/%d), alpha2=%.2f (%d/%d)...' % 
              (theta_alpha1,ia1,len(thetas_alpha1),theta_alpha2,ia2,len(thetas_alpha2)))
        # construct generative map
        what1 = np.array([[np.cos(theta_alpha1)],[np.sin(theta_alpha1)]])
        what2 = np.array([[np.cos(theta_alpha2)],[np.sin(theta_alpha2)]])
        What  = torch.from_numpy(np.hstack((what1,what2))).float()
        # sample-based estimate of causal effect
        nce_iu, info_iu = causaleffect.ind_uncond(   params, decoder, classifier, device, What=What)
        nce_ic, _       = causaleffect.ind_cond(     params, decoder, classifier, device, What=What)
        nce_ju, _       = causaleffect.joint_uncond( params, decoder, classifier, device, What=What)
        nce_jc, _       = causaleffect.joint_cond(   params, decoder, classifier, device, What=What)
        # compute likelihood
        data["loglik"][ia1,ia2] = -loss_functions.linGauss_NLL_loss(
            torch.from_numpy(X).float(), What, params["gamma"])
        # store results
        data["ce_iu"][ia1,ia2] = -nce_iu.detach().numpy()
        data["ce_ic"][ia1,ia2] = -nce_ic.detach().numpy()
        data["ce_ju"][ia1,ia2] = -nce_ju.detach().numpy()
        data["ce_jc"][ia1,ia2] = -nce_jc.detach().numpy()
        data["nentropy"][ia1,ia2] = info_iu["negHYhat"].detach().numpy()
        data["expcondentropy"][ia1,ia2] = info_iu["EalphaHYhatgivenalpha"].detach().numpy()
        data["adj_iu_ic"][ia1,ia2] = data["ce_iu"][ia1,ia2] - data["ce_ic"][ia1,ia2]
        data["adj_iu_ju"][ia1,ia2] = data["ce_iu"][ia1,ia2] - data["ce_ju"][ia1,ia2]
        data["adj_ju_jc"][ia1,ia2] = data["ce_ju"][ia1,ia2] - data["ce_jc"][ia1,ia2]
        data["adj_ic_jc"][ia1,ia2] = data["ce_ic"][ia1,ia2] - data["ce_jc"][ia1,ia2]


# --- save results ---
print('Done! Saving results...')
sio.savemat('results/visualize_causalobj_and.mat',
            {'data' : data,
             'theta1' : params["theta1"],
             'theta2' : params["theta2"],
             'thetas_alpha1' : thetas_alpha1,
             'thetas_alpha2' : thetas_alpha2,
             'params' : params})
print('Done!')


#%% plot slices (see visualize_causalobj_plot.m for plots in paper)

import numpy as np
import matplotlib.pyplot as plt
import plotting
import pickle

data_ce = np.concatenate((data["ce_iu"],
                          data["ce_ic"],
                          data["ce_ju"],
                          data["ce_jc"]))
clim_ce = [np.min(data_ce), np.max(data_ce)]
clim_ll = [np.min(data["loglik"]), np.max(data["loglik"])]
a1 = thetas_alpha1/np.pi*180.
a2 = thetas_alpha2/np.pi*180.

slice_inds = range(len(thetas_alpha2))
ticks = [0,45,90,135,180]

fig, axs = plt.subplots(2, 4, figsize=(20,12))
ims = [[] for i in range(8)]
ls = [[] for i in range(8)]

# --- plot 0: likelihood ---
ls[0] = axs[0,0].plot(a1, data["loglik"][:,slice_inds])
axs[0,0].set_title(r'$\theta(w_{\alpha_1})=%.0f, \theta(w_{\alpha_2})=%.0f$: Likelihood' % 
                   (params["theta1"]/np.pi*180., params["theta2"]/np.pi*180.))

# --- subplot 1: entropy ---
axs[0,1].plot(a1,-2*np.log(1/2)*np.ones_like(a1),color='gray',linestyle=':',
              label='2*0.69 nats')
axs[0,1].plot(a1,2*(-1./4*np.log(1./4)-3./4*np.log(3./4))*np.ones_like(a1),
              color='gray',linestyle=':', label='2*0.56 nats')
ls[1] = axs[0,1].plot(a1, data["nentropy"][:,slice_inds])
axs[0,1].set_title(r'$2 H(p(\widehat{Y}))$')

# --- subplot 2: negative expected conditional entropy ---
ls[2] = axs[0,2].plot(a1, data["expcondentropy"][:,slice_inds])
axs[0,2].set_title(r'$\sum_i -E_{\alpha_i}[H(p(\widehat{Y}|\alpha_i))]$')

# --- subplot 3: adjustment for ind/conditional ---
ls[3] = axs[0,3].plot(a1, data["adj_iu_ic"][:,slice_inds])
axs[0,3].set_title(r'$\sum_i I(\alpha_i;\alpha_{-i},\beta|\widehat{Y})$')

# --- subplot 4: causal effect (ind/uncond) ---
ls[4] = axs[1,0].plot(a1, data["ce_iu"][:,slice_inds])
axs[1,0].set_title('Causal effect\n(indep/uncond)')
axs[1,0].set_ylim(clim_ce)

# --- subplot 5: causal effect (ind/cond) ---
ls[5] = axs[1,1].plot(a1, data["ce_ic"][:,slice_inds])
axs[1,1].set_title('Causal effect\n(indep/cond)')
axs[1,1].set_ylim(clim_ce)

# --- subplot 6: causal effect (joint/uncond) ---
ls[6] = axs[1,2].plot(a1, data["ce_ju"][:,slice_inds])
axs[1,2].set_title('Causal effect\n(joint/uncond)')
axs[1,2].set_ylim(clim_ce)

# --- subplot 7: causal effect (joint/cond) ---
ls[7] = axs[1,3].plot(a1, data["ce_jc"][:,slice_inds])
axs[1,3].set_title('Causal effect\n(joint/cond)')
axs[1,3].set_ylim(clim_ce)

for i in range(8):
    for j in range(len(slice_inds)):
        ls[i][j].set_label('$\\theta(\\widehat{w}_{\\alpha_2}) = %.0f^{\\circ}$' % a2[slice_inds[j]])
for i in range(2):
    for j in range(4):
        axs[i,j].set_xticks(ticks)
        axs[i,j].set_xlim([0,180])
        axs[i,j].grid(True)
        axs[i,j].set_xlabel(r'$\theta(\widehat{w}_{\alpha_1})$')
        axs[i,j].legend(loc='lower left',mode='expand',ncol=2)
fig.tight_layout()