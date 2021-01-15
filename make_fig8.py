"""
    make_fig8.py
    
    Reproduces Figure 8 in O'Shaughnessy et al., 'Generative causal
    explanations of black-box classifiers,' Proc. NeurIPS 2020: empirical
    results for causal/information flow objectives with linear/gaussian
    generative map and linear classifier.

    Note: this script creates the file ./results/fig8.mat. The matlab script
    make_fig8_fig9_fig10.m creates the final plots in the paper.
"""

import numpy as np
import scipy.io as sio
from torch.autograd import Variable
import torch
from loss_functions import *
import causaleffect_lingauss_decoder
import plotting
import util
import matplotlib.pyplot as plt


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
    "Nbeta"          : 100, #TODO - 2500 in paper
    "Nalpha"         : 25, #TODO - 500 in paper
    "gamma"          : 0.001,
    "break_up_ce"    : False,
    "decoder_net"    : "linGauss",
    "classifier_net" : "oneHyperplane"}
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


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
from models.linearGaussian import Decoder
decoder = Decoder(params["x_dim"], params["z_dim"])
decoder.apply(util.weights_init_normal)
What = Variable(torch.mul(torch.randn(params["x_dim"], params["z_dim"],
                                      dtype=torch.float),0.5), requires_grad=True)


# --- initialize classifier ---
from models.toy_classifiers import OneHyperplaneClassifier
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
        CEs[ia,ib,0] = -causaleffect_lingauss_decoder.ind_uncond(
            params, decoder, classifier, device, What=What)[0]
        CEs[ia,ib,1] = -causaleffect_lingauss_decoder.ind_cond(
            params, decoder, classifier, device, What=What)[0]
        CEs[ia,ib,2] = -causaleffect_lingauss_decoder.joint_uncond(
            params, decoder, classifier, device, What=What)[0]
        CEs[ia,ib,3] = -causaleffect_lingauss_decoder.joint_cond(
            params, decoder, classifier, device, What=What)[0]


# --- save results ---        
print('Done! Saving results...')
sio.savemat('results/fig8.mat',
            {'CEs' : CEs,
             'thetas_alpha' : thetas_alpha,
             'thetas_beta' : thetas_beta,
             'params' : params})
print('Done!')


# --- make debug plot ---
# NOTE: plots in paper made using make_fig8_fig9.m,
# which uses the .mat file generated above
import numpy as np
import matplotlib.pyplot as plt
import plotting
clim = [np.min(CEs), np.max(CEs)]
fig, axs = plt.subplots(1, 4, figsize=(20,5))
# ind_cond
plt.axes(axs[0])
im = plotting.plotsurface(axs[0],
                      thetas_alpha/np.pi*180.,
                      thetas_beta/np.pi*180.,
                      CEs[:,:,0],
                      clim)
axs[0].set_title('Causal effect\n(independent, conditional)')
plt.xticks(ticks=(0,45,90,135,180))
plt.yticks(ticks=(0,45,90,135,180))
# ind_uncond
plt.axes(axs[1])
im = plotting.plotsurface(axs[1],
                      thetas_alpha/np.pi*180.,
                      thetas_beta/np.pi*180.,
                      CEs[:,:,1],
                      clim)
axs[1].set_title('Causal effect\n(independent, unconditional)')
plt.xticks(ticks=(0,45,90,135,180))
plt.yticks(ticks=(0,45,90,135,180))
# joint_cond
plt.axes(axs[2])
im = plotting.plotsurface(axs[2],
                      thetas_alpha/np.pi*180.,
                      thetas_beta/np.pi*180.,
                      CEs[:,:,2],
                      clim)
axs[2].set_title('Causal effect\n(joint, conditional)')
plt.xticks(ticks=(0,45,90,135,180))
plt.yticks(ticks=(0,45,90,135,180))
# joint_uncond
plt.axes(axs[3])
im = plotting.plotsurface(axs[3],
                      thetas_alpha/np.pi*180.,
                      thetas_beta/np.pi*180.,
                      CEs[:,:,3],
                      clim)
axs[3].set_title('Causal effect\n(joint, unconditional)')
plt.xticks(ticks=(0,45,90,135,180))
plt.yticks(ticks=(0,45,90,135,180))
# format
for i in range(4):
    axs[i].set_xlabel(r'$\theta(w_{\alpha})$')
    axs[i].set_ylabel(r'$\theta(w_{\beta})$')
plt.tight_layout(pad=3)
fig.colorbar(im, ax=axs[:], shrink=0.5)