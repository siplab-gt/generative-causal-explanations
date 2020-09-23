import numpy as np
import torch
import util

import old.causaleffect as ceold
import causaleffect as ce

from load_mnist import *

classifier_file = 'pretrained_models/mnist_38_classifier/model.pt'
vae_file = 'pretrained_models/mnist_38_vae/model.pt'

# --- initialize ---
K = 1
L = 7
M = 2
c_dim = 1
img_size = 28*28
class_use = np.array([3,8])
class_use_str = np.array2string(class_use)    
y_dim = class_use.shape[0]
newClass = range(0,M)
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
from models.CVAE import Decoder, Encoder
checkpoint_vae = torch.load(vae_file, map_location=device)
encoder = Encoder(K+L,c_dim,img_size).to(device)
decoder = Decoder(K+L,c_dim,img_size).to(device)
encoder.apply(util.weights_init_normal)
decoder.apply(util.weights_init_normal)

# --- load classifier ---
from models.CNN_classifier import CNN
checkpoint_model = torch.load(classifier_file, map_location=device)
classifier = CNN(y_dim).to(device)
classifier.load_state_dict(checkpoint_model['model_state_dict_classifier'])

# --- compute causal effect ---
params_old = {'Nalpha' : 50, 'Nbeta' : 50, 'decoder_net' : 'VAE', 'z_dim' : K+L, 'alpha_dim' : K, 'y_dim' : M}
params = {'Nalpha' : 50, 'Nbeta' : 50, 'K' : K, 'L' : L, 'M' : M}
ntrials = 10
for i in range(ntrials):
    encoder.apply(util.weights_init_normal)
    decoder.apply(util.weights_init_normal)
    Iold = ceold.joint_uncond(params_old, decoder, classifier, device)[0].detach().numpy()
    I = ce.joint_uncond(params, decoder, classifier)[0].detach().numpy()
    print('Trial %d/%d: old=%f, new=%f (err=%g)' % \
        (i, ntrials, Iold, I, np.linalg.norm(I-Iold)/np.linalg.norm(Iold)))