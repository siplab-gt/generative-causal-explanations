"""
    make_fig3.py
    
    Produces Figure 3 in O'Shaughnessy et al., 'Generative causal
    explanations of black-box classifiers,' arXiv, June 2020: global
    explanation for CNN classifier trained on MNIST 3/8.
"""

#%%
%load_ext autoreload
%autoreload 2
import numpy as np
import torch
import util

# --- parameters ---
# dataset
data_classes = [3, 8]
# classifier
classifier_path = 'pretrained_models/mnist_38_classifier'
# vae
vae_path = 'pretrained_models/mnist_38_vae'
K = 1
L = 7
# other
randseed = 0

# --- initialize ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if randseed is not None:
    np.random.seed(randseed)
    torch.manual_seed(randseed)

# --- load data ---
from load_mnist import load_mnist_classSelect
X, Y, tridx = load_mnist_classSelect('train', data_classes,
                    range(0,len(data_classes)))
vaX, vaY, vaidx = load_mnist_classSelect('val', data_classes,
                    range(0,len(data_classes)))
ntrain, nrow, ncol, c_dim = X.shape
x_dim = nrow*ncol

# load classifier
from models.CNN_classifier import CNN
classifier = CNN(len(data_classes)).to(device)
checkpoint = torch.load('%s/model.pt' % classifier_path,
                        map_location=device)
classifier.load_state_dict(checkpoint['model_state_dict_classifier'])

# initialize VAE
from models.CVAE import Decoder, Encoder
encoder = Encoder(K+L, c_dim, x_dim).to(device)
decoder = Decoder(K+L, c_dim, x_dim).to(device)
encoder.apply(util.weights_init_normal)
decoder.apply(util.weights_init_normal)

# %% train GCE
from GCE import GenerativeCausalExplainer
torch.autograd.set_detect_anomaly(True)
gce = GenerativeCausalExplainer(classifier, decoder, encoder, device)
gce.train(X, K, L)

# %%
