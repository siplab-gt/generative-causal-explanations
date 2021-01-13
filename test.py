#%%
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
from load_mnist import load_fashion_mnist_classSelect
X, Y, tridx = load_fashion_mnist_classSelect('train', data_classes,
                    range(0,len(data_classes)))
vaX, vaY, vaidx = load_fashion_mnist_classSelect('val', data_classes,
                    range(0,len(data_classes)))
ntrain, nrow, ncol, c_dim = X.shape
x_dim = nrow*ncol

#%%
from models.CVAE import Decoder, Encoder
for (i,nfilt) in enumerate(filts_per_layer):
    encoder = Encoder(K+L, c_dim, x_dim, filt_per_layer=nfilt)
    decoder = Decoder(K+L, c_dim, x_dim, filt_per_layer=nfilt)
    params_encoder = np.sum([p.numel() for p in encoder.parameters()])
    params_decoder = np.sum([p.numel() for p in decoder.parameters()])
    print('%d filters per layer: %d parameters in encoder; %d parameters in decoder' %
        (nfilt, params_encoder, params_decoder))
# %%
