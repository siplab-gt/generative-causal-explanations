"""
    make_fig5cd.py
    
    Reproduces Figure 5(c-d) in O'Shaughnessy et al., 'Generative causal
    explanations of black-box classifiers,' Proc. NeurIPS 2020: qualitative
    results for explanation of CNN classifier trained on fashion MNIST classes
    {0,3,4}.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import util
import load_mnist
import scipy.io as sio
import os
from GCE import GenerativeCausalExplainer


# --- parameters ---
# gce
K = 2
L = 4
train_steps = 8000
Nalpha = 25
Nbeta = 100
lam = 0.05
batch_size = 64
lr = 5e-4
# dataset
c_dim = 1
img_size = 28
data_classes = np.array([0,3,4]) # fmnist class indices to extract
# plot
latent_sweep_vals = np.linspace(-2,2,25)
latent_sweep_plot = [0,4,8,12,16,20,24]
classifier_path = 'pretrained_models/fmnist_034_classifier/'
gce_path = 'pretrained_models/fmnist_034_gce/'
retrain_gce = False # train explanatory VAE from scratch
save_gce = False # save/overwrite pretrained explanatory VAE at gce_path


# --- initialize ---
z_dim = K+L
y_dim = data_classes.shape[0]
ylabels = range(0,y_dim)
nsweep = len(latent_sweep_vals)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = {'z_dim' : K+L,
          'alpha_dim' : K,
          'No' : 100,
          'Ni' : 25,
          'decoder_net' : 'VAE_CNN',
          'break_up_ce' : False}


# --- load test data ---
X, Y, tridx = load_mnist.load_fashion_mnist_classSelect('train', data_classes, ylabels)
vaX, vaY, vaidx = load_mnist.load_fashion_mnist_classSelect('val', data_classes, ylabels)
ntrain, nrow, ncol, c_dim = X.shape
x_dim = nrow*ncol


# --- load classifier ---
from models.CNN_classifier import CNN
checkpoint_model = torch.load(os.path.join(classifier_path,'model.pt'), map_location=device)
classifier = CNN(y_dim).to(device)
classifier.load_state_dict(checkpoint_model['model_state_dict_classifier'])


# --- train/load GCE ---
from models.CVAE import Decoder, Encoder
if retrain_gce:
    encoder = Encoder(K+L, c_dim, x_dim).to(device)
    decoder = Decoder(K+L, c_dim, x_dim).to(device)
    encoder.apply(util.weights_init_normal)
    decoder.apply(util.weights_init_normal)
    gce = GenerativeCausalExplainer(classifier, decoder, encoder, device)
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
        torch.save(gce, os.path.join(gce_path,'model.pt'))
        sio.savemat(os.path.join(gce_path, 'training-info.mat'),
            {'classifier_path' : classifier_path, 'K' : K, 'L' : L,
             'train_steps' : train_steps, 'Nalpha' : Nalpha, 'Nbeta' : Nbeta,
             'lam' : lam, 'batch_size' : batch_size, 'lr' : lr,
             'data_classes' : data_classes})
else: # load pretrained model
    gce = torch.load(os.path.join(gce_path,'model.pt'), map_location=device)


# --- generate latent factor sweep plots ---
# get sample points from each class
sample_ind = np.concatenate((np.where(vaY == 0)[0][:3],
                             np.where(vaY == 1)[0][:3],
                             np.where(vaY == 2)[0][:2]))
cols = [[0.047,0.482,0.863],[1.000,0.761,0.039],[0.561,0.788,0.227]]
border_size = 3
nsamples = len(sample_ind)
latentsweep_vals = [-3., -2., -1., 0., 1., 2., 3.]
Xhats = np.zeros((z_dim,nsamples,len(latentsweep_vals),img_size,img_size,1))
yhats = np.zeros((z_dim,nsamples,len(latentsweep_vals)))
# generate images
for isamp in range(nsamples):
    x = torch.from_numpy(np.expand_dims(vaX[sample_ind[isamp]],0))
    x_torch = x.permute(0,3,1,2).float().to(device)
    z = gce.encoder(x_torch)[0][0].cpu().detach().numpy()
    for latent_dim in range(z_dim):
        for (ilatentsweep, latentsweep_val) in enumerate(latentsweep_vals):
            ztilde = z.copy()
            ztilde[latent_dim] += latentsweep_val
            xhat = gce.decoder(torch.unsqueeze(torch.from_numpy(ztilde),0).to(device))
            yhat = np.argmax(classifier(xhat)[0].cpu().detach().numpy())
            img = 1.-xhat.permute(0,2,3,1).cpu().detach().numpy().squeeze()
            Xhats[latent_dim,isamp,ilatentsweep,:,:,0] = img
            yhats[latent_dim,isamp,ilatentsweep] = yhat
# create and format plots
for latent_dim in range(z_dim):
    fig, axs = plt.subplots(nsamples, len(latentsweep_vals))
    for isamp in range(nsamples):
        for (ilatentsweep, latentsweep_val) in enumerate(latentsweep_vals):
            img = Xhats[latent_dim,isamp,ilatentsweep,:,:,0].squeeze()
            yhat = int(yhats[latent_dim,isamp,ilatentsweep])
            img_bordered = np.tile(np.expand_dims(np.array(cols[yhat]),(0,1)),(img_size+2*border_size,img_size+2*border_size,1))
            img_bordered[border_size:-border_size,border_size:-border_size,:] = \
                np.tile(np.expand_dims(img,2),(1,1,3))
            axs[isamp,ilatentsweep].imshow(img_bordered, interpolation='nearest')
            axs[isamp,ilatentsweep].axis('off')
    axs[0,round(len(latentsweep_vals)/2)-1].set_title('Sweep latent dimension %d' % (latent_dim+1))
    if True:
        print('Exporting latent dimension %d...' % (latent_dim+1))
        plt.savefig('./figs/fig5cd_%d.svg' % (latent_dim+1), bbox_inches=0)

print('Columns - latent values in sweep: ' + str(latentsweep_vals))
print('Rows - sample indices in vaX: ' + str(sample_ind))