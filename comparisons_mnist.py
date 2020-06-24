#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from load_mnist import *

import torch
import tensorflow as tf
import keras
import keras.engine

import skimage

import matplotlib.pyplot as plt
from skimage.color import label2rgb

parameters = {
    'lime' : {'K' : 5},
    'shap' : {},
    'ig'   : {'steps' : 50},
    'l2x'  : {'k' : 4,
              'batchsize' : 1000}}

"""
Generate explanations of MNIST 3/8 classifier using other methods
"""

#%% load dataset

class_use = np.array([3,8])
class_use_str = np.array2string(class_use)    
y_dim = class_use.shape[0]
newClass = range(0,y_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_size = 64
trX, trY, tr_idx = load_mnist_classSelect('train',class_use,newClass)
vaX, vaY, va_idx = load_mnist_classSelect('val',class_use,newClass)
trX_3ch = np.tile(trX, (1,1,1,3))
vaX_3ch = np.tile(vaX, (1,1,1,3))
sample_inputs = vaX[0:test_size]
sample_inputs_torch = torch.from_numpy(sample_inputs)
sample_inputs_torch = sample_inputs_torch.permute(0,3,1,2).float().to(device)        
ntrain = trX.shape[0]

# data sample to provide local explanation for
x3 = vaX[np.where(1-vaY)[0][0]]
x8 = vaX[np.where(vaY)[0][0]]


#%% load trained classifier

from cnnClassifierModel import CNN
classifier = CNN(y_dim).to(device)
batch_orig = 64
checkpoint = torch.load('./pretrained_models/mnist_38_classifier/model.pt',
                        map_location=device)
classifier.load_state_dict(checkpoint['model_state_dict_classifier'])

trYhat = classifier(torch.from_numpy(trX).permute(0,3,1,2).float())[0].detach().numpy()
vaYhat = classifier(torch.from_numpy(vaX).permute(0,3,1,2).float())[0].detach().numpy()
classifier_accuracy_train = np.sum(np.round(trYhat[:,1]) == trY) / len(trY)
classifier_accuracy_val = np.sum(np.round(vaYhat[:,1]) == vaY) / len(vaY)


#%%
"""
Compute integrated gradients explanation
INPUTS
 - x : data point to explain - np.array of shape (28, 28, 1)
 - i_class : target class to explain - int
OUTPUTS
 - ig_explanation : explanation for x in class i_class - np.array of shape (28, 28, 1)
"""
def integrated_gradients(x, i_class):
    baseline = 0.*np.expand_dims(x,0)
    # scale input
    xs_scaled = np.zeros((parameters['ig']['steps']+1,*x.shape))
    for i in range(parameters['ig']['steps']+1):
        xs_scaled[i,:,:,:] = baseline + (float(i)/parameters['ig']['steps'])*(np.expand_dims(x,0)-baseline)
    xs_scaled = torch.from_numpy(xs_scaled).permute(0,3,1,2).float().to(device)
    xs_scaled.requires_grad = True
    # perform classification
    ce_loss = torch.nn.CrossEntropyLoss()
    _, output = classifier(xs_scaled)
    loss = ce_loss(output, torch.from_numpy(i_class*np.ones(parameters['ig']['steps']+1)).long().to(device))
    # compute gradient of class (i_class) output wrt input
    loss.backward()
    grads = xs_scaled.grad.detach().numpy().transpose(0,2,3,1)
    # compute integrated gradients
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = np.average(grads, axis=0)
    integrated_gradients = ((np.expand_dims(x,0)-baseline)*avg_grads).squeeze()
    return integrated_gradients

ig_explanation_3 = integrated_gradients(x3, 0)
ig_explanation_8 = integrated_gradients(x8, 1)


#%% generate LIME explanations

"""
Classifier pipeline for lime
INPUTS
 - numpy array of shape (nsamp, 28, 28, 3)
OUTPUTS
 - classifier probabilities of shape (nsamp, nclass)
"""
def classifier_pipeline_lime(X):
    X = X[:,:,:,0:1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_torch = torch.from_numpy(X).permute(0,3,1,2).float().to(device)
    y = classifier(X_torch)[0].detach().numpy()
    return y

import os
os.chdir('otheralgs/lime')
import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
os.chdir('../..')
lime_explainer = lime_image.LimeImageExplainer(verbose = False)
segmenter = SegmentationAlgorithm('quickshift',
                                  kernel_size = 1,
                                  max_dist = 200,
                                  ratio = 0.0)
lime_explanation_3 = lime_explainer.explain_instance(np.tile(x3, (1,1,3)),
                        classifier_fn = classifier_pipeline_lime, 
                        top_labels=2,
                        hide_color=0,
                        num_samples=10000,
                        segmentation_fn=segmenter)
lime_image_3, lime_mask_3 = lime_explanation_3.get_image_and_mask(0,
                        positive_only=False,
                        num_features=parameters["lime"]["K"],
                        hide_rest=False,
                        min_weight = 0.01)
lime_explanation_8 = lime_explainer.explain_instance(np.tile(x8, (1,1,3)),
                        classifier_fn = classifier_pipeline_lime, 
                        top_labels=2,
                        hide_color=0,
                        num_samples=10000,
                        segmentation_fn=segmenter)
lime_image_8, lime_mask_8 = lime_explanation_8.get_image_and_mask(1,
                        positive_only=False,
                        num_features=parameters["lime"]["K"],
                        hide_rest=False,
                        min_weight = 0.01)


#%% generate L2X explanations

# From author's code
class Sample_Concrete(keras.layers.Layer):
	"""
	Layer for sample Concrete / Gumbel-Softmax variables. 
    
	"""
	def __init__(self, tau0, k, **kwargs):
		self.tau0 = tau0
		self.k = k
		super(Sample_Concrete, self).__init__(**kwargs)

	def call(self, logits):   
		# logits: [BATCH_SIZE, d]
		logits_ = keras.backend.expand_dims(logits, -2)# [BATCH_SIZE, 1, d]

		batch_size = tf.shape(logits_)[0]
		d = tf.shape(logits_)[2]
		uniform = tf.random_uniform(shape =(batch_size, self.k, d), 
			minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
			maxval = 1.0)

		gumbel = - keras.backend.log(-keras.backend.log(uniform))
		noisy_logits = (gumbel + logits_)/self.tau0
		samples = keras.backend.softmax(noisy_logits)
		samples = keras.backend.max(samples, axis = 1) 

		# Explanation Stage output.
		threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
		discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
		
		return keras.backend.in_train_phase(samples, discrete_logits)

	def compute_output_shape(self, input_shape):
		return input_shape

# P(S|X)
tau = 0.1
input_shape = (28,28,1)
model_input = keras.layers.Input(shape=input_shape,
                          dtype='float32',
                          name = 's/input')
net = keras.layers.Conv2D(32,
                          kernel_size = (2,2),
                          activation = 'relu',
                          padding = 'same',
                          kernel_regularizer = keras.regularizers.l2(1e-3),
                          name = 's/conv1')(model_input)
net = keras.layers.MaxPooling2D(pool_size = (2,2),
                          padding = 'same',
                          name='s/maxpool1')(net)
net = keras.layers.Conv2D(32,
                          kernel_size = (2,2),
                          padding = 'same',
                          activation = 'relu',
                          kernel_regularizer = keras.regularizers.l2(1e-3),
                          name = 's/conv2')(net)
net = keras.layers.MaxPooling2D(pool_size = (2,2),
                          padding = 'same',
                          name='s/maxpool2')(net)
net = keras.layers.Conv2D(1,
                          kernel_size = (2,2),
                          padding = 'same',
                          activation = 'relu',
                          kernel_regularizer = keras.regularizers.l2(1e-3),
                          name = 's/conv3')(net)
#net = keras.layers.MaxPooling2D(pool_size = (2,2),
#                          padding = 'same',
#                          name = 's/maxpool3')(net)
net = keras.layers.Flatten(name = 's/flatten')(net)
#logits = keras.layers.Dense(49,
#                          activation = 'relu',
#                          name='s/dense')(net)
samples_flat = Sample_Concrete(tau,
                          parameters['l2x']['k'],
                          name = 's/sample')(net)
model_pSgivenX = keras.models.Model(model_input, samples_flat)
model_pSgivenX.compile(loss = None,
                       optimizer = 'rmsprop',
                       metrics = [None])
#print('\n === Architecture for p(S|X) === \n')
#print(model_pSgivenX.summary())

# q(Y|X_S)
samples = keras.layers.Reshape((7,7,1))(samples_flat)
upsampled_samples = keras.layers.UpSampling2D(
                          size = (4,4),
                          interpolation = 'nearest',
                          name = 'upsample')(samples)
qYgivenXS_input = keras.layers.Multiply(name = 'multiply')([model_input, upsampled_samples])
net = keras.layers.Conv2D(32,
                          kernel_size = (2,2),
                          activation = 'relu',
                          padding = 'same',
                          kernel_regularizer = keras.regularizers.l2(1e-3),
                          name = 'conv1')(qYgivenXS_input)
net = keras.layers.MaxPooling2D(pool_size = (2,2),
                          padding = 'same',
                          name = 'maxpool1')(net)
net = keras.layers.Conv2D(32,
                          kernel_size = (2,2),
                          padding = 'same',
                          activation = 'relu',
                          kernel_regularizer = keras.regularizers.l2(1e-3),
                          name = 'conv2')(net)
net = keras.layers.MaxPooling2D(pool_size = (2,2),
                          padding = 'same',
                          name = 'maxpool2')(net)
net = keras.layers.Flatten(name = 'flatten')(net)
preds = keras.layers.Dense(2,
                          activation = 'softmax',
                          kernel_regularizer = keras.regularizers.l2(1e-3),
                          name = 'dense')(net)
model_qYgivenXS = keras.models.Model(model_input, preds)
#print('\n === Architecture for q(Y|X_S) === \n')
#print(model_qYgivenXS.summary())

# train
adam = keras.optimizers.Adam(lr = 1e-3)
model_qYgivenXS.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc']) 
model_qYgivenXS.fit(trX, trYhat,
              validation_data = (vaX, vaYhat),
              epochs = 10,
              batch_size = parameters['l2x']['batchsize'])

# compute explanations
scores = model_pSgivenX.predict(vaX,
                                verbose = 0,
                                batch_size = parameters['l2x']['batchsize'])
l2x_explanation_3_downsample = model_pSgivenX.predict(np.expand_dims(x3,0)).reshape((7,7))
l2x_explanation_8_downsample = model_pSgivenX.predict(np.expand_dims(x8,0)).reshape((7,7))
l2x_explanation_3 = skimage.transform.pyramid_expand(l2x_explanation_3_downsample,
                                                     upscale = 4,
                                                     sigma = 0,
                                                     order = 0,
                                                     multichannel = False)
l2x_explanation_8 = skimage.transform.pyramid_expand(l2x_explanation_8_downsample,
                                                     upscale = 4,
                                                     sigma = 0,
                                                     order = 0,
                                                     multichannel = False)


#%% generate SHAP explanations

import shap
trX_sample = trX[np.random.choice(trX.shape[0], 1000, replace=False)]
trX_sample_torch = torch.from_numpy(trX_sample).permute(0,3,1,2).float()
shap_explainer = shap.DeepExplainer(classifier, trX_sample_torch)
x3_torch = torch.from_numpy(np.expand_dims(x3,0)).permute(0,3,1,2).float()
x8_torch = torch.from_numpy(np.expand_dims(x8,0)).permute(0,3,1,2).float()
shap_explanation_3 = shap_explainer.shap_values(x3_torch)[0].squeeze()
shap_explanation_8 = shap_explainer.shap_values(x8_torch)[1].squeeze()


#%% plot explanations - single combined plot

import plotting
import matplotlib.colors
import scipy.io as sio

fig, axs = plt.subplots(2,5, figsize = (12,4))
axs[0,0].set_title('Input image')
axs[0,0].imshow(1.-np.squeeze(x3), cmap='gray')
axs[1,0].imshow(1.-np.squeeze(x8), cmap='gray')

#cmap = plt.get_cmap("bwr")
cols = {'golden_poppy' : [1.0, 0.761, 0.039],
        'bright_navy_blue' : [0.047, 0.482, 0.863],
        'rosso_corsa' : [0.816, 0.000, 0.000]}
cdict = {'red' :   [[0.0, cols['bright_navy_blue'][0], cols['bright_navy_blue'][0]],
                    [0.5, 1.0, 1.0],
                    [1.0, cols['rosso_corsa'][0], cols['rosso_corsa'][0]]],
         'green' : [[0.0, cols['bright_navy_blue'][1], cols['bright_navy_blue'][1]],
                    [0.5, 1.0, 1.0],
                    [1.0, cols['rosso_corsa'][1], cols['rosso_corsa'][1]]],
         'blue' :  [[0.0, cols['bright_navy_blue'][2], cols['bright_navy_blue'][2]],
                    [0.5, 1.0, 1.0],
                    [1.0, cols['rosso_corsa'][2], cols['rosso_corsa'][2]]]}
"""cdict = {'red' :   [[0.0, cols['bright_navy_blue'][0], cols['bright_navy_blue'][0]],
                    [0.5, 1.0, 1.0],
                    [1.0, cols['golden_poppy'][0], cols['golden_poppy'][0]]],
         'green' : [[0.0, cols['bright_navy_blue'][1], cols['bright_navy_blue'][1]],
                    [0.5, 1.0, 1.0],
                    [1.0, cols['golden_poppy'][1], cols['golden_poppy'][1]]],
         'blue' :  [[0.0, cols['bright_navy_blue'][2], cols['bright_navy_blue'][2]],
                    [0.5, 1.0, 1.0],
                    [1.0, cols['golden_poppy'][2], cols['golden_poppy'][2]]]}"""
cmap = matplotlib.colors.LinearSegmentedColormap('causalvae_cmap', segmentdata=cdict, N=256)

axs[0,1].set_title('LIME')
try:
    #axs[0,1].imshow(label2rgb(3-lime_mask_3, lime_image_3 / 255., bg_label = 0, alpha = 0.5),
    #                interpolation = 'nearest')
    axs[0,1].imshow(lime_mask_3,
                    vmin = -2,
                    vmax = 2,
                    cmap = cmap,
                    interpolation = 'nearest')
    plotting.outline_mask(axs[0,1], x3 > 0, bounds=(0,27,0,27))
    axs[1,1].imshow(lime_mask_8,
                    vmin = -2,
                    vmax = 2,
                    cmap = cmap,
                    interpolation = 'nearest')
    plotting.outline_mask(axs[1,1], x8 > 0, bounds=(0,27,0,27))
except:
    pass

axs[0,2].set_title('DeepSHAP')
try:
    shap_range = [np.min(np.concatenate((shap_explanation_3, shap_explanation_8))),
                  np.max(np.concatenate((shap_explanation_3, shap_explanation_8)))]
    shap_cmap = [np.min((shap_range[0], -shap_range[1])),
                 np.max((shap_range[1], -shap_range[0]))]
    axs[0,2].imshow(shap_explanation_3,
                    vmin = shap_cmap[0],
                    vmax = shap_cmap[1],
                    cmap = cmap,
                    interpolation = 'nearest')
    plotting.outline_mask(axs[0,2], x3 > 0, bounds=(0,27,0,27))
    axs[1,2].imshow(shap_explanation_8,
                    vmin = shap_cmap[0],
                    vmax = shap_cmap[1],
                    cmap = cmap,
                    interpolation = 'nearest')
    plotting.outline_mask(axs[1,2], x8 > 0, bounds=(0,27,0,27))
except:
    pass

axs[0,3].set_title('IG')
try:
    ig_range = [np.min(np.concatenate((ig_explanation_3, ig_explanation_8))),
                  np.max(np.concatenate((ig_explanation_3, ig_explanation_8)))]
    ig_cmap = [np.min((ig_range[0], -ig_range[1])),
                 np.max((ig_range[1], -ig_range[0]))]
    axs[0,3].imshow(ig_explanation_3,
                    vmin = ig_cmap[0],
                    vmax = ig_cmap[1],
                    cmap = cmap,
                    interpolation = 'nearest')
    plotting.outline_mask(axs[0,3], x3 > 0, bounds=(0,27,0,27))
    axs[1,3].imshow(ig_explanation_8,
                    vmin = ig_cmap[0],
                    vmax = ig_cmap[1],
                    cmap = cmap,
                    interpolation = 'nearest')
    plotting.outline_mask(axs[1,3], x8 > 0, bounds=(0,27,0,27))
except:
    pass

axs[0,4].set_title('L2X')
try:
    axs[0,4].imshow(l2x_explanation_3,
                    vmin = -2,
                    vmax = 2,
                    cmap = cmap,
                    interpolation = 'nearest')
    plotting.outline_mask(axs[0,4], x3 > 0, bounds=(0,27,0,27))
    axs[1,4].imshow(l2x_explanation_8,
                    vmin = -2,
                    vmax = 2,
                    cmap = cmap,
                    interpolation = 'nearest')
    plotting.outline_mask(axs[1,4], x8 > 0, bounds=(0,27,0,27))
except:
    pass

for i in range(0,2):
    for j in range(0,5):
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])
        
#plt.savefig('./figs/fig_comparison_matplotlib.svg', bbox_inches=0)


#%% generate plot from saved results

try:
    results = sio.loadmat('/Users/matthewoshaughnessy/Dropbox/class38/selectAlpha__mnist_JOINT_UNCOND_zdim8_alpha1_No100_Ni25_lam0.005_class38/sweeplatentfactors.mat',
                          variable_names = ['imgOut_real'])
    ind_sweep_3 = [4,7,10,13,16,19,22];
    cvae_explanation_3_sweep = results['imgOut_real'][0,:,:,:,:,0].squeeze()
    fig, axs = plt.subplots(1,len(ind_sweep_3))
    for (i,dim) in enumerate(ind_sweep_3):
        axs[i].imshow(1.-cvae_explanation_3_sweep[0,dim,:,:].squeeze(),
                      cmap='gray', interpolation='nearest')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.savefig('./figs/fig_comparison_cvae_3.svg', bbox_inches=0)
    ind_sweep_8 = [4,7,10,13,16,19,22];
    fig, axs = plt.subplots(1,len(ind_sweep_8))
    for (i,dim) in enumerate(ind_sweep_8):
        axs[i].imshow(1.-cvae_explanation_8_sweep[0,dim,:,:].squeeze(),
                      cmap='gray', interpolation='nearest')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.savefig('./figs/fig_comparison_cvae_8.svg', bbox_inches=0)
except:
    print('invalid path to results data')

        
#%% export single plots

# MNIST 3
fig = plt.figure()
fig.set_size_inches((1,1))
ax = plt.axes([0,0,1,1])
plt.imshow(1.-np.squeeze(x3), cmap='gray', interpolation='nearest')
plt.axis('off')
plt.savefig('./figs/fig_comparison_mnist_3.svg', bbox_inches=0)

# MNIST 8
fig = plt.figure()
fig.set_size_inches((1,1))
ax = plt.axes([0,0,1,1])
plt.imshow(1.-np.squeeze(x8), cmap='gray', interpolation='nearest')
plt.axis('off')
plt.savefig('./figs/fig_comparison_mnist_8.svg', bbox_inches=0)

# LIME 3
fig = plt.figure()
fig.set_size_inches((1,1))
ax = plt.axes([0,0,1,1])
plt.imshow(lime_mask_3, vmin=-2, vmax=2, cmap=cmap, interpolation='nearest')
plotting.outline_mask(ax, x3 > 0, bounds=(0,27,0,27))
plt.axis('off')
plt.savefig('./figs/fig_comparison_lime_3.svg', bbox_inches=0)

# LIME 8
fig = plt.figure()
fig.set_size_inches((1,1))
ax = plt.axes([0,0,1,1])
plt.imshow(lime_mask_8, vmin=-2, vmax=2, cmap=cmap, interpolation='nearest')
plotting.outline_mask(ax, x8 > 0, bounds=(0,27,0,27))
plt.axis('off')
plt.savefig('./figs/fig_comparison_lime_8.svg', bbox_inches=0)

# SHAP 3
shap_range = [np.min(np.concatenate((shap_explanation_3, shap_explanation_8))),
              np.max(np.concatenate((shap_explanation_3, shap_explanation_8)))]
shap_cmap = [np.min((shap_range[0], -shap_range[1])),
             np.max((shap_range[1], -shap_range[0]))]
fig = plt.figure()
fig.set_size_inches((1,1))
ax = plt.axes([0,0,1,1])
plt.imshow(shap_explanation_3, vmin=shap_cmap[0], vmax=shap_cmap[1],
           cmap=cmap, interpolation='nearest')
plotting.outline_mask(ax, x3 > 0, bounds=(0,27,0,27))
plt.axis('off')
plt.savefig('./figs/fig_comparison_shap_3.svg', bbox_inches=0)

# SHAP 8
shap_range = [np.min(np.concatenate((shap_explanation_3, shap_explanation_8))),
              np.max(np.concatenate((shap_explanation_3, shap_explanation_8)))]
shap_cmap = [np.min((shap_range[0], -shap_range[1])),
             np.max((shap_range[1], -shap_range[0]))]
fig = plt.figure()
fig.set_size_inches((1,1))
ax = plt.axes([0,0,1,1])
plt.imshow(shap_explanation_8, vmin=shap_cmap[0], vmax=shap_cmap[1],
           cmap=cmap, interpolation='nearest')
plotting.outline_mask(ax, x8 > 0, bounds=(0,27,0,27))
plt.axis('off')
plt.savefig('./figs/fig_comparison_shap_8.svg', bbox_inches=0)

# IG 3
ig_range = [np.min(np.concatenate((ig_explanation_3, ig_explanation_8))),
              np.max(np.concatenate((ig_explanation_3, ig_explanation_8)))]
ig_cmap = [np.min((ig_range[0], -ig_range[1])),
             np.max((ig_range[1], -ig_range[0]))]
fig = plt.figure()
fig.set_size_inches((1,1))
ax = plt.axes([0,0,1,1])
plt.imshow(ig_explanation_3, vmin=ig_cmap[0], vmax=ig_cmap[1],
           cmap=cmap, interpolation='nearest')
plotting.outline_mask(ax, x3 > 0, bounds=(0,27,0,27))
plt.axis('off')
plt.savefig('./figs/fig_comparison_ig_3.svg', bbox_inches=0)

# IG 8
ig_range = [np.min(np.concatenate((ig_explanation_3, ig_explanation_8))),
              np.max(np.concatenate((ig_explanation_3, ig_explanation_8)))]
ig_cmap = [np.min((ig_range[0], -ig_range[1])),
             np.max((ig_range[1], -ig_range[0]))]
fig = plt.figure()
fig.set_size_inches((1,1))
ax = plt.axes([0,0,1,1])
plt.imshow(ig_explanation_8, vmin=ig_cmap[0], vmax=ig_cmap[1],
           cmap=cmap, interpolation='nearest')
plotting.outline_mask(ax, x8 > 0, bounds=(0,27,0,27))
plt.axis('off')
plt.savefig('./figs/fig_comparison_ig_8.svg', bbox_inches=0, dpi=160)

# L2X 3
ig_range = [np.min(np.concatenate((ig_explanation_3, ig_explanation_8))),
              np.max(np.concatenate((ig_explanation_3, ig_explanation_8)))]
ig_cmap = [np.min((ig_range[0], -ig_range[1])),
             np.max((ig_range[1], -ig_range[0]))]
fig = plt.figure()
fig.set_size_inches((1,1))
ax = plt.axes([0,0,1,1])
plt.imshow(l2x_explanation_3, vmin=-2, vmax=2, cmap=cmap, interpolation='nearest')
plotting.outline_mask(ax, x3 > 0, bounds=(0,27,0,27))
plt.axis('off')
plt.savefig('./figs/fig_comparison_l2x_3.svg', bbox_inches=0)

# L2X 8
ig_range = [np.min(np.concatenate((ig_explanation_3, ig_explanation_8))),
              np.max(np.concatenate((ig_explanation_3, ig_explanation_8)))]
ig_cmap = [np.min((ig_range[0], -ig_range[1])),
             np.max((ig_range[1], -ig_range[0]))]
fig = plt.figure()
fig.set_size_inches((1,1))
ax = plt.axes([0,0,1,1])
plt.imshow(l2x_explanation_8, vmin=-2, vmax=2, cmap=cmap, interpolation='nearest')
plotting.outline_mask(ax, x8 > 0, bounds=(0,27,0,27))
plt.axis('off')
plt.savefig('./figs/fig_comparison_l2x_8.svg', bbox_inches=0)

