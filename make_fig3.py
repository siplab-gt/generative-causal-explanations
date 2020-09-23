"""
    make_fig3.py
    
    Produces Figure 3 in O'Shaughnessy et al., 'Generative causal
    explanations of black-box classifiers,' arXiv, June 2020: global
    explanation for CNN classifier trained on MNIST 3/8 digits.
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
train_steps = 8000
Nalpha = 25
Nbeta = 100
lam = 0.05
batch_size = 64
lr = 5e-4
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
gce = GenerativeCausalExplainer(classifier, decoder, encoder, device)
traininfo = gce.train(X, K, L,
                      steps=train_steps,
                      Nalpha=Nalpha,
                      Nbeta=Nbeta,
                      lam=lam,
                      batch_size=batch_size,
                      lr=lr)
#torch.save(gce, 'results/gce.pth')
#gce_loaded = torch.load('results/gce.pth', map_location=device)

# %% create figure
import matplotlib.pyplot as plt
# compute global explanation
sample_ind = np.concatenate((np.where(vaY == 0)[0][:4],
                             np.where(vaY == 1)[0][:4]))
nsamples = len(sample_ind)
zs_sweep = [-3., -2., -1., 0., 1., 2., 3.]
Xhats = np.zeros((K+L,nsamples,len(zs_sweep),int(np.sqrt(x_dim)),int(np.sqrt(x_dim)),1))
yhats = np.zeros((K+L,nsamples,len(zs_sweep)))
for isamp in range(nsamples):
    x = torch.from_numpy(np.expand_dims(vaX[sample_ind[isamp]],0))
    x_torch = x.permute(0,3,1,2).float().to(device)
    z = encoder(x_torch)[0][0].detach().cpu().numpy()
    for latent_dim in range(K+L):
        for (iz, z_sweep) in enumerate(zs_sweep):
            ztilde = z.copy()
            ztilde[latent_dim] += z_sweep
            xhat = decoder(torch.unsqueeze(torch.from_numpy(ztilde),0).to(device))
            yhat = np.argmax(classifier(xhat)[0].detach().cpu().numpy())
            img = 1.-xhat.permute(0,2,3,1).detach().cpu().numpy().squeeze()
            Xhats[latent_dim,isamp,iz,:,:,0] = img
            yhats[latent_dim,isamp,iz] = yhat
# make plots
cols = [[0.047,0.482,0.863],[1.000,0.761,0.039],[0.561,0.788,0.227]]
border_size = 3
for latent_dim in range(K+L):
    fig, axs = plt.subplots(nsamples, len(zs_sweep))
    for isamp in range(nsamples):
        for (iz, z_sweep) in enumerate(zs_sweep):
            img = Xhats[latent_dim,isamp,iz,:,:,0].squeeze()
            yhat = int(yhats[latent_dim,isamp,iz])
            img_bordered = np.tile(np.expand_dims(np.array(cols[yhat]),(0,1)),
                (int(np.sqrt(x_dim))+2*border_size,int(np.sqrt(x_dim))+2*border_size,1))
            img_bordered[border_size:-border_size,border_size:-border_size,:] = \
                np.tile(np.expand_dims(img,2),(1,1,3))
            axs[isamp,iz].imshow(img_bordered, interpolation='nearest')
            axs[isamp,iz].axis('off')
    axs[0,round(len(zs_sweep)/2)-1].set_title('Sweep latent dimension %d' % (latent_dim+1))
    if True:
        print('Exporting latent dimension %d...' % (latent_dim+1))
        plt.savefig('./figs/fig_mnist_qual_latentdim%d.svg' % (latent_dim+1), bbox_inches=0)

print('Columns - latent values in sweep: ' + str(zs_sweep))
print('Rows - sample indices in vaX: ' + str(sample_ind))
# %%
