"""
    make_fig5ab.py
    
    Reproduces Figure 5(a-b) in O'Shaughnessy et al., 'Generative causal
    explanations of black-box classifiers,' Proc. NeurIPS 2020: quantitative
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


# --- compute and plot information flow ---
info_flow = gce.informationFlow_singledim(range(0,K+L))
cols = {'golden_poppy' : [1.000,0.761,0.039],
        'bright_navy_blue' : [0.047,0.482,0.863],
        'rosso_corsa' : [0.816,0.000,0.000]}
x_labels = ('$\\alpha_1$', '$\\alpha_2$', '$\\beta_1$', '$\\beta_2$', '$\\beta_3$', '$\\beta_4$')
fig, ax = plt.subplots()
ax.bar(range(z_dim), info_flow, color=[
    cols['rosso_corsa'], cols['rosso_corsa'], cols['bright_navy_blue'],
    cols['bright_navy_blue'], cols['bright_navy_blue'], cols['bright_navy_blue']])
plt.xticks(range(z_dim), x_labels)
ax.yaxis.grid(linewidth='0.3')
plt.ylabel('Information flow to $\\widehat{Y}$')
plt.title('Information flow of individual causal factors')
plt.savefig('./figs/fig5a.svg')
plt.savefig('./figs/fig5a.pdf')


# --- compute classifier accuracy after 'removing' latent factors ---
classifier_accuracy_original = np.zeros(z_dim)
Yhat = np.zeros((len(vaX)))
Yhat_reencoded = np.zeros((len(vaX)))
Yhat_aspectremoved = np.zeros((z_dim, len(vaX)))

for i_samp in range(len(vaX)):
    x = torch.from_numpy(vaX[i_samp:i_samp+1,:,:,:]).permute(0,3,1,2).float().to(device)
    Yhat[i_samp] = np.argmax(classifier(x)[0].cpu().detach().numpy())
    z = gce.encoder(x)[0]
    xhat = gce.decoder(z)
    Yhat_reencoded[i_samp] = np.argmax(classifier(xhat)[0].cpu().detach().numpy())
    for i_latent in range(z_dim):   
        z = gce.encoder(x)[0]
        z[0,i_latent] = torch.randn((1))
        xhat = gce.decoder(z)
        Yhat_aspectremoved[i_latent,i_samp] = np.argmax(classifier(xhat)[0].cpu().detach().numpy())

classifier_accuracy = np.mean(vaY == Yhat)
classifier_accuracy_reencoded = np.mean(vaY == Yhat_reencoded)
classifier_accuracy_aspectremoved = np.zeros((z_dim))
for i in range(z_dim):
    classifier_accuracy_aspectremoved[i] = np.mean(vaY == Yhat_aspectremoved[i,:])


# --- plot classifier accuracy ---
cols = {'black' : [0.000, 0.000, 0.000],
        'golden_poppy' : [1.000,0.761,0.039],
        'bright_navy_blue' : [0.047,0.482,0.863],
        'rosso_corsa' : [0.816,0.000,0.000]}
x_labels = ('orig','reenc','$\\alpha_1$', '$\\alpha_2$', '$\\beta_1$', '$\\beta_2$',
            '$\\beta_3$', '$\\beta_4$')
fig, ax = plt.subplots()
ax.yaxis.grid(linewidth='0.3')
ax.bar(range(z_dim+2), np.concatenate(([classifier_accuracy],
                                       [classifier_accuracy_reencoded],
                                       classifier_accuracy_aspectremoved)),
       color=[cols['black'], cols['black'], cols['rosso_corsa'],
              cols['rosso_corsa'], cols['bright_navy_blue'],
              cols['bright_navy_blue'], cols['bright_navy_blue'],
              cols['bright_navy_blue']])
plt.xticks(range(z_dim+2), x_labels)
plt.ylim((0.2,1.0))
plt.yticks((0.2,0.4,0.6,0.8,1.0))#,('0.5','','0.75','','1.0'))
plt.ylabel('Classifier accuracy')
plt.title('Classifier accuracy after removing aspect')
plt.savefig('./figs/fig5b.svg')
plt.savefig('./figs/fig5b.pdf')