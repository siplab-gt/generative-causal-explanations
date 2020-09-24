"""
    make_vae_capacity.py
    
    Produces Figure X in O'Shaughnessy et al., 'Generative causal
    explanations of black-box classifiers,' arXiv, June 2020:
    information flow and explanation quality as VAE capacity is varied.
"""

import numpy as np
import torch
import util
import plotting
from GCE import GenerativeCausalExplainer

# --- parameters ---
# dataset
data_classes = np.array([0,3,4])
# classifier
classifier_path = 'pretrained_models/fmnist_034_classifier'
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
        # get data
        gce.encoder.eval()
        gce.decoder.eval()
        torch.cuda.empty_cache()
        torch.save(gce, 'results/gce_vae_capacity_%dfilters_lambda%g.pth' \
            % (nfilt, lam))
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
        plotting.plotExplanation(1.-Xhats, yhats,
            save_path='figs/fig_vae_capacity_%dfilters_lambda%g'%(nfilt,lam))
# save all results to file
from scipy.io import savemat
savemat('results/vae_capacity_data.mat', {'data' : data})
