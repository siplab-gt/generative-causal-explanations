"""
    make_fig18_fig19.py
    
    Produces Figures 18 and 19 in O'Shaughnessy et al., 'Generative causal
    explanations of black-box classifiers,' Proc. NeurIPS 2020: final
    value of causal effect and data fidelity terms in objective for
    various capacities of VAE.

    Note: this script creates the file ./results/fig18.mat. The matlab script
    make_fig18.m creates the final plots in the paper.
"""

import numpy as np
import torch
import util
import plotting
import matplotlib.pyplot as plt
from GCE import GenerativeCausalExplainer
import os


# --- parameters ---
# dataset
data_classes = np.array([0,3,4])
# classifier
classifier_path = './pretrained_models/fmnist_034_classifier'
# vae
K = 2
L = 4
train_steps = 8000
Nalpha = 100
Nbeta = 25
lam = 0.05
batch_size = 32
lr = 1e-4
filts_per_layer = [4,8,16,32,48,64]
lambdas = np.logspace(-3,-1,10)
# other
randseed = 0
gce_path = './pretrained_models/vae_capacity'
retrain_gce = True
save_gce = True


# --- initialize ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if randseed is not None:
    np.random.seed(randseed)
    torch.manual_seed(randseed)


# --- load data ---
from load_mnist import load_fashion_mnist_classSelect
X, Y, tridx = load_fashion_mnist_classSelect('train', data_classes,
                    range(0,len(data_classes)))
vaX, vaY, vaidx = load_fashion_mnist_classSelect('val', data_classes,
                    range(0,len(data_classes)))
ntrain, nrow, ncol, c_dim = X.shape
x_dim = nrow*ncol


# --- load classifier ---
from models.CNN_classifier import CNN
classifier = CNN(len(data_classes)).to(device)
checkpoint = torch.load('%s/model.pt' % classifier_path,
                        map_location=device)
classifier.load_state_dict(checkpoint['model_state_dict_classifier'])


# --- initialize VAE and train GCE ---
from models.CVAE import Decoder, Encoder
data = {
    'loss' : np.zeros((len(filts_per_layer),len(lambdas),train_steps)),
    'loss_ce' : np.zeros((len(filts_per_layer),len(lambdas),train_steps)),
    'loss_nll' : np.zeros((len(filts_per_layer),len(lambdas),train_steps)),
    'Ijoint' : np.zeros((len(filts_per_layer),len(lambdas))),
    'Is' : np.zeros((len(filts_per_layer),len(lambdas),K+L))}
for (i_f, nfilt) in enumerate(filts_per_layer):
    for (i_l, lam) in enumerate(lambdas):
        filename = 'model_%dfilters_lambda%g.pt' % (nfilt, lam)
        if retrain_gce:
            print('=== %d FILTERS PER LAYER, LAMBDA = %g ===' % (nfilt, lam))
            # initialize VAE
            encoder = Encoder(K+L, c_dim, x_dim,
                filt_per_layer=nfilt).to(device)
            decoder = Decoder(K+L, c_dim, x_dim,
                filt_per_layer=nfilt).to(device)
            encoder.apply(util.weights_init_normal)
            decoder.apply(util.weights_init_normal)
            # train GCE
            gce = GenerativeCausalExplainer(classifier, decoder, encoder,
                device, debug_print=False)
            traininfo = gce.train(X, K, L,
                steps=train_steps,
                Nalpha=Nalpha,
                Nbeta=Nbeta,
                lam=lam,
                batch_size=batch_size,
                lr=lr)
            if save_gce:
                if not os.path.exists(gce_path):
                    os.makedirs(gce_path)
                torch.save((gce, traininfo), os.path.join(gce_path, filename))
        else: # load pretrained model
            gce, traininfo = torch.load(os.path.join(gce_path, filename))
        # get data
        gce.encoder.eval()
        gce.decoder.eval()
        torch.cuda.empty_cache()
        data['loss'][i_f,i_l,:] = traininfo['loss']
        data['loss_ce'][i_f,i_l,:] = traininfo['loss_ce']
        data['loss_nll'][i_f,i_l,:] = traininfo['loss_nll']
        data['Ijoint'][i_f,i_l] = gce.informationFlow()
        data['Is'][i_f,i_l,:] = gce.informationFlow_singledim(dims=range(K+L))
        # save figures for explanation
        sample_ind = np.concatenate((np.where(vaY == 0)[0][:3],
                                     np.where(vaY == 1)[0][:3],
                                     np.where(vaY == 2)[0][:2]))
        x = torch.from_numpy(vaX[sample_ind])
        zs_sweep = [-3., -2., -1., 0., 1., 2., 3.]
        Xhats, yhats = gce.explain(x, zs_sweep)
        if not os.path.exists('./figs/fig19/'):
            os.makedirs('./figs/fig19/')
        plotting.plotExplanation(1.-Xhats, yhats,
            save_path='./figs/fig19/%dfilters_lambda%g'%(nfilt,lam))
        plt.close('all')
# save all results to file
from scipy.io import savemat
savemat('./results/fig18.mat', {'data' : data})