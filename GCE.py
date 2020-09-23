import time
import datetime
import re

import numpy as np
import scipy.io as sio
import scipy as sp
import scipy.linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as torchmodels
from torch.autograd import Variable
import pickle

import loss_functions
import causaleffect
import plotting
import util

import matplotlib.pyplot as plt
import os

from util import *
from load_mnist import *
from mnist_test_fnc import sweepLatentFactors

class GenerativeCausalExplainer:

    """
    :param classifier: classifier to explain
    :param decoder: decoder model
    :param encoder: encoder model
    :param device: pytorch device object
    :param save_output: save model output when training
    :param save_dir: directory to save model outputs when training
    :param debug_print: print debug messages
    """
    def __init__(self, classifier, decoder, encoder, device,
                 save_output = False,
                 save_dir = None,
                 debug_print = True):

        # initialize
        super(GenerativeCausalExplainer, self).__init__()
        self.classifier = classifier
        self.decoder = decoder
        self.encoder = encoder
        self.device = device
        self.params = {'save_output' : save_output,
                       'save_dir'    : save_dir,
                       'debug_print' : debug_print}
        if save_dir is not None and not os.path.exists(self.params['save_dir']):
            os.makedirs(self.params['save_dir'])
        if self.params['debug_print']:
            print("Parameters:")
            print(self.params)


    """
    :param X: training data (not necessarily same as classifier training data)
    :param K: number of causal factors
    :param L: number of noncausal factors
    :param steps: number of training steps
    :param Nalpha: number of causal latent factors
    :param Nbeta: number of noncausal latent factors
    :param lam: regularization parameter
    :param causal_obj: causal objective {'IND_UNCOND','IND_COND','JOINT_UNCOND','JOINT_COND'}
    :param batch_size: batch size for training
    :param lr: learning rate for adam optimizer
    :param b1: beta1 parameter for adam optimizer
    :param b2: beta2 parameter for adam optimizer
    :param use_ce: if false, do not use causal effect part of objective
    """
    def train(self, X, K, L,
              steps = 50000,
              Nalpha = 50,
              Nbeta = 50,
              lam = 0.0001,
              causal_obj = 'JOINT_UNCOND',
              batch_size = 100,
              lr = 0.0001,
              b1 = 0.5,
              b2 = 0.999,
              use_ce = True):
    
        # initialize
        ntrain = X.shape[0]
        sample_input = torch.from_numpy(X[0]).unsqueeze(0).float().permute(0,3,1,2)
        M = self.classifier(sample_input.to(self.device))[0].shape[1]
        self.train_params = {
                 'K'                 : K,
                 'L'                 : L,
                 'steps'             : steps,
                 'Nalpha'            : Nalpha,
                 'Nbeta'             : Nbeta,
                 'lambda'            : lam,
                 'causal_obj'        : causal_obj,
                 'batch_size'        : batch_size,
                 'lr'                : lr,
                 'b1'                : b1,
                 'b2'                : b2,
                 'use_ce'            : use_ce}
        ceparams = {
                  'Nalpha'           : Nalpha,
                  'Nbeta'            : Nbeta,
                  'K'                : K,
                  'L'                : L,
                  'z_dim'            : K+L,
                  'M'                : M}
        debug = {'loss'              : np.zeros((steps)),
                 'loss_ce'           : np.zeros((steps)),
                 'loss_nll'          : np.zeros((steps)),
                 'loss_nll_logdet'   : np.zeros((steps)),
                 'loss_nll_quadform' : np.zeros((steps)),
                 'loss_nll_mse'      : np.zeros((steps)),
                 'loss_nll_kld'      : np.zeros((steps))}

        # initialize for training
        opt_params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        self.opt = torch.optim.Adam(opt_params, lr=lr, betas=(b1, b2))
        start_time = time.time()

        # training loop
        for k in range(0, steps):
        
            # reset gradient
            self.opt.zero_grad()

            # compute negative log-likelihood
            randIdx = np.random.randint(0, ntrain, batch_size)
            Xbatch = torch.from_numpy(X[randIdx]).float().permute(0,3,1,2).to(self.device)
            z, mu, logvar = self.encoder(Xbatch)
            Xhat = self.decoder(z)
            nll, nll_mse, nll_kld = loss_functions.VAE_LL_loss(Xbatch, Xhat, logvar, mu)

            # compute causal effect
            if causal_obj == 'IND_UNCOND':
                causalEffect, ceDebug = causaleffect.ind_uncond(ceparams,
                    self.decoder, self.classifier, self.device)
            elif causal_obj == 'IND_COND':
                causalEffect, ceDebug = causaleffect.ind_cond(ceparams,
                    self.decoder, self.classifier, self.device)
            elif causal_obj == 'JOINT_UNCOND':
                causalEffect, ceDebug = causaleffect.joint_uncond(ceparams,
                    self.decoder, self.classifier, self.device)
            elif causal_obj == 'JOINT_COND':
                causalEffect, ceDebug = causaleffect.joint_cond(ceparams,
                    self.decoder, self.classifier, self.device)
            else:
                print('Invalid causal objective!')
            
            # compute gradient
            loss = use_ce*causalEffect + lam*nll
            loss.backward()
            self.opt.step()

            # save debug info for this step
            debug['loss'][k] = loss.item()
            debug['loss_ce'][k] = causalEffect.item()
            debug['loss_nll'][k] = (lam*nll).item()
            debug['loss_nll_mse'][k] = (lam*nll_mse).item()
            debug['loss_nll_kld'][k] = (lam*nll_kld).item()
            if self.params['debug_print']:
                print("[Step %d/%d] time: %4.2f  [CE: %g] [ML: %g] [loss: %g]" % \
                      (k, steps, time.time() - start_time, debug['loss_ce'][k],
                      debug['loss_nll'][k], debug['loss'][k]))
            if self.params['save_output'] and k % 1000 == 0:
                torch.save({
                    'step': k,
                    'model_state_dict_classifier' : self.classifier.state_dict(),
                    'model_state_dict_encoder' : self.encoder.state_dict(),
                    'model_state_dict_decoder' : self.decoder.state_dict(),
                    'optimizer_state_dict' : self.opt.state_dict(),
                    'loss' : loss,
                    }, '%s_batch_%d.pt' % \
                    (self.params['save_dir'], self.params['batch_size']))
        
        # save/return debug data from entire training run
        debug['Xbatch'] = Xbatch.detach().cpu().numpy()
        debug['Xhat'] = Xhat.detach().cpu().numpy()
        if self.params['save_output']:
            datestamp = ''.join(re.findall(r'\d+', str(datetime.datetime.now())[:10]))
            timestamp = ''.join(re.findall(r'\d+', str(datetime.datetime.now())[11:19]))
            matfilename = 'results_' + datestamp + '_' + timestamp + '.mat'
            sio.savemat(save_dir + matfilename, {'params' : params, 'data' : debug})
            if self.params['debug_print']:
                print('Finished saving data to ' + matfilename)
        return debug