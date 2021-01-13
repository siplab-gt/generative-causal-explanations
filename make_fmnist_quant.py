#%%
from __future__ import division
import numpy as np
import torch
import matplotlib.pyplot as plt

from util import *
from load_mnist import *
from mnist_test_fnc import sweepLatentFactors

# --- parameters ---
# classes 0,3,4
y_dim     = 3
z_dim     = 6
alpha_dim = 2
c_dim     = 1
img_size  = 28
class_use = np.array([0,3,4])
i0 = 1 # class 0 = t-shirt/top
i3 = 3 # class 3 = dress
i4 = 0 # class 4 = coat
latent_sweep_vals = np.linspace(-2,2,25)
latent_sweep_plot = [0,4,8,12,16,20,24]
classifier_file = 'pretrained_models/fmnist_034_classifier/model.pt'
vae_file = 'pretrained_models/fmnist_034_vae_zdim6_alphadim2_lambda0.05/model.pt'

# --- initialize ---
class_use_str = np.array2string(class_use)    
y_dim = class_use.shape[0]
newClass = range(0,y_dim)
nsweep = len(latent_sweep_vals)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = {'z_dim' : z_dim,
          'alpha_dim' : alpha_dim,
          'No' : 100,
          'Ni' : 25,
          'decoder_net' : 'VAE_CNN',
          'break_up_ce' : False}

#%% --- load test data ---
X, Y, tridx = load_fashion_mnist_classSelect('train',class_use,newClass)
vaX, vaY, vaidx = load_fashion_mnist_classSelect('val',class_use,newClass)
ntrain = X.shape[0]

#%% --- load VAE ---
from models.CVAE import Decoder, Encoder
checkpoint_vae = torch.load(vae_file, map_location=device)
encoder = Encoder(z_dim,c_dim,img_size**2).to(device)
decoder = Decoder(z_dim,c_dim,img_size**2).to(device)
encoder.load_state_dict(checkpoint_vae['model_state_dict_encoder'])
decoder.load_state_dict(checkpoint_vae['model_state_dict_decoder'])

#%% --- load classifier ---
from models.CNN_classifier import CNN
checkpoint_model = torch.load(classifier_file, map_location=device)
classifier = CNN(y_dim).to(device)
classifier.load_state_dict(checkpoint_model['model_state_dict_classifier'])

#%% --- encode example points to latent space ---
x0_torch = torch.from_numpy(np.expand_dims(vaX[i0],0)).permute(0,3,1,2).float().to(device)
x3_torch = torch.from_numpy(np.expand_dims(vaX[i3],0)).permute(0,3,1,2).float().to(device)
x4_torch = torch.from_numpy(np.expand_dims(vaX[i4],0)).permute(0,3,1,2).float().to(device)
latent_x0 = encoder(x0_torch)[0].cpu().detach().numpy()
latent_x3 = encoder(x3_torch)[0].cpu().detach().numpy()
latent_x4 = encoder(x4_torch)[0].cpu().detach().numpy()
decoded_x0 = decoder(torch.unsqueeze(torch.from_numpy(latent_x0),0).float().to(device)).cpu().detach().numpy().squeeze()
decoded_x3 = decoder(torch.unsqueeze(torch.from_numpy(latent_x3),0).float().to(device)).cpu().detach().numpy().squeeze()
decoded_x4 = decoder(torch.unsqueeze(torch.from_numpy(latent_x4),0).float().to(device)).cpu().detach().numpy().squeeze()
print('Latent representation of sample 0 (validation set index %d): %s' % (i0, str(latent_x0)))
print('Latent representation of sample 3 (validation set index %d): %s' % (i3, str(latent_x3)))
print('Latent representation of sample 4 (validation set index %d): %s' % (i4, str(latent_x4)))

#%% --- generate images from latent sweep ---
latent_sweep = np.zeros((z_dim, nsweep, img_size, img_size))
for ilatent in range(z_dim):
    for (isweep, v) in enumerate(latent_sweep_vals):
        latent_vec = np.zeros((z_dim))
        latent_vec[ilatent] = v
        latent_vec_torch = torch.unsqueeze(torch.from_numpy(latent_vec),0).float().to(device)
        latent_sweep[ilatent,isweep,:,:] = decoder(latent_vec_torch).cpu().detach().numpy()
        
#%% --- plot latent sweep ---
fig, axs = plt.subplots(z_dim, len(latent_sweep_plot))
for ilatent in range(z_dim):
    for (isweep, sweep_idx) in enumerate(latent_sweep_plot):
        img = 1.-latent_sweep[ilatent,sweep_idx,:,:].squeeze()
        axs[ilatent,isweep].imshow(img, cmap='gray', interpolation='nearest')
        axs[ilatent,isweep].set_xticks([])
        axs[ilatent,isweep].set_yticks([])
print('columns correspond to latent values %s' % str(latent_sweep_vals[latent_sweep_plot]))
plt.savefig('./figs/fig_fmnist_quant_latentsweep.svg', bbox_inches=0)


#%% --- compute information flow ---
from informationFlow import information_flow_singledim
info_flow = np.zeros((z_dim))
for i in range(z_dim):
    print('Computing information flow for latent dimension %d/%d...' % (i+1,z_dim))
    info_flow[i] = information_flow_singledim(i, params, decoder, classifier, device)

# --- plot information flow ---
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
plt.savefig('./figs/fig_fmnist_quant_if.svg')
plt.savefig('./figs/fig_fmnist_quant_if.pdf')

#%% --- compute classifier accuracy ---
classifier_accuracy_original = np.zeros(z_dim)
Yhat = np.zeros((len(vaX)))
Yhat_reencoded = np.zeros((len(vaX)))
Yhat_aspectremoved = np.zeros((z_dim, len(vaX)))

for i_samp in range(len(vaX)):
    x = torch.from_numpy(vaX[i_samp:i_samp+1,:,:,:]).permute(0,3,1,2).float().to(device)
    Yhat[i_samp] = np.argmax(classifier(x)[0].cpu().detach().numpy())
    z = encoder(x)[0]
    xhat = decoder(z)
    Yhat_reencoded[i_samp] = np.argmax(classifier(xhat)[0].cpu().detach().numpy())
    for i_latent in range(z_dim):   
        z = encoder(x)[0]
        z[0,i_latent] = torch.randn((1))
        xhat = decoder(z)
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
plt.savefig('./figs/fig_fmnist_quant_accuracy.svg')
plt.savefig('./figs/fig_fmnist_quant_accuracy.pdf')