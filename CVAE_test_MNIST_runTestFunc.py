from __future__ import division
import time
import datetime
import re

import numpy as np
import scipy.io as sio
import scipy as sp
import scipy.linalg

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

import loss_functions
import causaleffect
import plotting
import util

import matplotlib.pyplot as plt
#import gif
import os

from util import *
from load_mnist import *
from mnist_test_fnc import sweepLatentFactors
from informationFlow import information_flow

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


#class Decoder(nn.Module):
# 
#    def __init__(self, x_dim, z_dim):
#        """
#        Encoder initializer
#        :param x_dim: dimension of the input
#        :param z_dim: dimension of the latent representation
#        """
#        super(Decoder, self).__init__()
#        self.model_enc = nn.Linear(int(z_dim),int(x_dim))
#
#    def reparameterize(self, mu,gamma):
#        eps = torch.randn_like(mu)
#        return mu + eps*gamma
#    
#    def forward(self, x,What,gamma):
#         # Run linear forward pass
#        z = F.linear(x,What,None)
#        z = self.reparameterize(z,gamma)
#        return z
    

def CVAE_test_MNIST_runTestFunc(model = "mnist_VAE_CNN", # currently not used
         steps          = 20000,
         batch_size     = 100,
         z_dim          = 6,
         z_dim_true     = 4,
         x_dim          = 10,
         y_dim          = 1,
         alpha_dim      = 4,
         ntrain         = 5000,
         No             = 15,
         Ni             = 15,
         lam_ML         = 0.000001,
         gamma          = 0.001,
         lr             = 0.0001,
         b1             = 0.5,
         b2             = 0.999,
         use_ce         = True,
         objective      = "IND_UNCOND",
         decoder_net    = "VAE_CNN", # options are ['linGauss','nonLinGauss','VAE','VAE_CNN']
         classifier_net = "cnn", # options are ['oneHyperplane','twoHyperplane','cnn']
         data_type      = "mnist", # options are ["2dpts","mnist"]
         break_up_ce    = True, # Whether or not to break up the forward passes of the network based on alphas
         randseed       = None,
         save_output    = False,
         debug_level    = 2,
         debug_plot     = False,
         save_plot      = False,
         c_dim          = 1,
         img_size       = 28):

    # initialization
    params = {
        "steps"          : steps,
        "batch_size"     : batch_size,
        "z_dim"          : z_dim,
        "z_dim_true"     : z_dim_true,
        "x_dim"          : x_dim,
        "y_dim"          : y_dim,
        "alpha_dim"      : alpha_dim,
        "ntrain"         : ntrain,
        "No"             : No,
        "Ni"             : Ni,
        "lam_ML"         : lam_ML,
        "gamma"          : gamma,
        "lr"             : lr,
        "b1"             : b1,
        "b2"             : b2,
        "use_ce"         : use_ce,
        "objective"      : objective,
        "decoder_net"    : decoder_net,
        "classifier_net" : classifier_net,
        "data_type"      : data_type,
        "break_up_ce"    : break_up_ce,
        "randseed"       : randseed,
        "save_output"    : save_output,
        "debug_level"    : debug_level,
        'c_dim'          : c_dim,
        'img_size'       : img_size}
    params["data_std"] = 2.
    if debug_level > 0:
        print("Parameters:")
        print(params)
    
    # Initialize arrays for storing performance data
    debug = {}
    debug["loss"]              = np.zeros((steps))
    debug["loss_ce"]           = np.zeros((steps))
    debug["loss_nll"]          = np.zeros((steps))
    debug["loss_nll_logdet"]   = np.zeros((steps))
    debug["loss_nll_quadform"] = np.zeros((steps))
    debug["loss_nll_mse"]      = np.zeros((steps))
    debug["loss_nll_kld"]      = np.zeros((steps))
    if not data_type == 'mnist':
        for i in range(params["z_dim"]):
            for j in range(params["z_dim_true"]):
                debug["cossim_w%dwhat%d"%(j+1,i+1)] = np.zeros((steps))
            for j in range(i+1,params["z_dim"]):
                debug["cossim_what%dwhat%d"%(i+1,j+1)] = np.zeros((steps))
    if save_plot: frames = []
    if data_type == 'mnist':
        class_use = np.array([3,8])
        class_use_str = np.array2string(class_use)    
        y_dim = class_use.shape[0]
        newClass = range(0,y_dim)
    save_dir = '/home/mnorko/Documents/Tensorflow/causal_vae/results/class38/selectAlpha__' 
        + data_type + '_' + objective + '_zdim' + str(z_dim) + '_alpha' + str(alpha_dim) 
        + '_No' + str(No) + '_Ni' + str(Ni) + '_lam' + str(lam_ML) + '_class' 
        + class_use_str[1:(len(class_use_str)-1):2] + '/'

        
    if data_type == '2dpts':
        break_up_ce = False
        params['break_up_ce'] = False

    # seed random number generator
    if randseed is not None:
        if debug_level > 0:
            print('Setting random seed to ' + str(randseed) + '.')
        np.random.seed(randseed)
        torch.manual_seed(randseed)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    # Generate data
    if data_type == '2dpts':    
        # --- construct projection matrices ---
        # 'true' orthogonal columns used to generate data from latent factors
        #Wsquare = sp.linalg.orth(np.random.rand(x_dim,x_dim))
        Wsquare = np.identity(x_dim)
        W = Wsquare[:,:z_dim_true]
        # 1st column of W
        w1 = np.expand_dims(W[:,0], axis=1)
        # 2nd column of W
        w2 = np.expand_dims(W[:,1], axis=1)
        # form projection matrices
        Pw1 = util.formProjMat(w1)
        Pw2 = util.formProjMat(w2)
        # convert to torch matrices
        Pw1_torch = torch.from_numpy(Pw1).float()
        Pw2_torch = torch.from_numpy(Pw2).float()
        # --- construct data ---
        # ntrain instances of alpha and x
        Alpha = params["data_std"]*np.random.randn(ntrain, z_dim_true)
        X = np.matmul(Alpha, W.T)
    elif data_type == 'mnist':
        test_size = 64
        X, Y, tridx = load_mnist_classSelect('train',class_use,newClass)
        vaX, vaY, vaidx = load_mnist_classSelect('val',class_use,newClass)
        sample_inputs = vaX[0:test_size]
        sample_labels = vaY[0:test_size]
        sample_inputs_torch = torch.from_numpy(sample_inputs)
        sample_inputs_torch = sample_inputs_torch.permute(0,3,1,2).float().to(device)        
        ntrain = X.shape[0]
    
    checkpoint = torch.load(save_dir + 'network_batch' + str(batch_size) + '.pt')
    # --- initialize decoder ---
    if decoder_net == 'linGauss':
        from linGaussModel import Decoder
        decoder = Decoder(x_dim, z_dim).to(device)
    elif decoder_net == 'nonLinGauss':
        from VAEModel import Decoder_samp
        z_num_samp = No
        decoder = Decoder_samp(x_dim,z_dim).to(device)
    elif decoder_net == 'VAE':
        from VAEModel import Decoder, Encoder
        encoder = Encoder(x_dim,z_dim).to(device)
        encoder.apply(weights_init_normal)
        decoder = Decoder(x_dim, z_dim).to(device)
        decoder.apply(weights_init_normal)
    elif decoder_net == 'VAE_CNN':
        from VAEModel_CNN import Decoder, Encoder
        encoder = Encoder(z_dim,c_dim,img_size).to(device)
        decoder = Decoder(z_dim,c_dim,img_size).to(device)
        encoder.load_state_dict(checkpoint['model_state_dict_encoder'])
        decoder.load_state_dict(checkpoint['model_state_dict_decoder'])
    else:
        print("Error: decoder_net should be one of: linGauss nonLinGauss VAE")
    
    # --- initialize classifier ---
    if classifier_net == 'oneHyperplane':        
        from hyperplaneClassifierModel import OneHyperplaneClassifier
        classifier = OneHyperplaneClassifier(x_dim, y_dim, Pw1_torch, ksig=5.).to(device)
        classifier.apply(weights_init_normal)
    elif classifier_net == 'twoHyperplane':
        from hyperplaneClassifierModel import TwoHyperplaneClassifier
        classifier = TwoHyperplaneClassifier(x_dim, y_dim, Pw1_torch, Pw2_torch, ksig=5.).to(device)
        classifier.apply(weights_init_normal)
    elif classifier_net == 'cnn':
        from cnnClassifierModel import CNN
        classifier = CNN(y_dim).to(device)
        batch_orig = 64
        checkpoint = torch.load('/home/mnorko/Documents/Tensorflow/causal_vae/results/mnist_batch64_lr0.1_class38/network_batch' + str(batch_orig) + '.pt')
        classifier.load_state_dict(checkpoint['model_state_dict_classifier'])
    else:
        print("Error: classifier should be one of: oneHyperplane twoHyperplane")
        
    ## 
    
    
    if decoder_net == 'linGauss':
    	What = Variable(torch.mul(torch.randn(x_dim, z_dim, dtype=torch.float),0.5), requires_grad=True)
    else:
        What = None
            

    sample_latent,mu,var = encoder(sample_inputs_torch)
    # Set up sample_latent small so the first 10 are class 1 and the second 10 are from another class
    class0_idx = np.where(sample_labels == 0)[0]
    class1_idx = np.where(sample_labels == 1)[0]
    sample_class0 = sample_latent[class0_idx,:]
    sample_class1 = sample_latent[class1_idx,:]
    sample_latent_small = torch.cat((sample_class0[0:10,:],sample_class1[0:10,:]),0)
    imgOut_real,probOut_real,latentOut_real = sweepLatentFactors(sample_latent_small,decoder,classifier,device,img_size,c_dim,y_dim,False)
    imgOut_real_int,probOut_real_int,latentOut_real_int = sweepLatentFactors(sample_latent_small,decoder,classifier,device,img_size,c_dim,y_dim,True)
    rand_latent = torch.from_numpy(np.random.randn(10,z_dim)).float().to(device)
    imgOut_rand,probOut_rand,latentOut_rand = sweepLatentFactors(rand_latent,decoder,classifier,device,img_size,c_dim,y_dim,False)
    imgOut_rand_int,probOut_rand_int,latentOut_rand_int = sweepLatentFactors(rand_latent,decoder,classifier,device,img_size,c_dim,y_dim,True)
    
    I_flow = information_flow(params,decoder, classifier,device, What=What)
    sio.savemat(save_dir + 'sweepLatentFactors_test.mat',{'imgOut_real':imgOut_real,'probOut_real':probOut_real,'latentOut_real':latentOut_real,
                                                     'imgOut_real_int':imgOut_real_int,'probOut_real_int':probOut_real_int,'latentOut_real_int':latentOut_real_int,
                                                     'imgOut_rand':imgOut_rand,'probOut_rand':probOut_rand,'latentOut_rand':latentOut_rand,
                                                     'imgOut_rand_int':imgOut_rand_int,'probOut_rand_int':probOut_rand_int,'latentOut_rand_int':latentOut_rand_int,
                                                     'sample_latent':sample_latent.detach().cpu().numpy(),'I_flow':I_flow})

     
  

lambda_change = [0.05, 0.075, 0.1, 0.25,0.5]
obj_change = ["JOINT_UNCOND"]
alpha_change = [3]
for obj_use  in obj_change:
    for lam_use in lambda_change:
        for alpha_use in alpha_change:
            CVAE_test_MNIST_runTestFunc(
                steps = 6000,
                batch_size = 64,
                lam_ML = lam_use,
                decoder_net = "VAE_CNN",
                classifier_net = "cnn",
                use_ce = True,
                objective = obj_use,
                data_type = "mnist", 
                break_up_ce= True,
                x_dim = 3,
                z_dim = 8,
                z_dim_true = 8,
                y_dim =2,
                alpha_dim = alpha_use,
                lr = 0.0005, 
                No = 100,
                Ni = 25,
                randseed = 0,
                save_output = True,
                debug_level = 1,
                debug_plot = True,
                save_plot = False,
                c_dim= 1,
                img_size= 28)

