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
import torchvision.transforms as transforms
import torchvision.models as models
from imagenet_zebra_gorilla_dataloader import Imagenet_Gor_Zeb
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


def GenerativeCausalExplanation(
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
         decoder_net    = "VAE_CNN", # options are ['linGauss','nonLinGauss','VAE','VAE_CNN','VAE_Imagenet','VAE_fMNIST']
         classifier_net = "cnn", # options are ['oneHyperplane','twoHyperplane','cnn','cnn_imagenet','cnn_fmnist']
         data_type      = "mnist", # options are ["2dpts","mnist","imagenet","fmnist"]
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
    if decoder_net == 'linGauss':
        for i in range(params["z_dim"]):
            for j in range(params["z_dim_true"]):
                debug["cossim_w%dwhat%d"%(j+1,i+1)] = np.zeros((steps))
            for j in range(i+1,params["z_dim"]):
                debug["cossim_what%dwhat%d"%(i+1,j+1)] = np.zeros((steps))
    if save_plot: frames = []
    if data_type == 'mnist' or data_type == 'fmnist':
        class_use = np.array([0,3,4])
        class_use_str = np.array2string(class_use)    
        y_dim = class_use.shape[0]
        newClass = range(0,y_dim)
        save_dir = '/home/mnorko/Documents/Tensorflow/causal_vae/results/fmnist_class034/' + data_type + '_' + objective + '_zdim' + str(z_dim) + '_alpha' + str(alpha_dim) + '_No' + str(No) + '_Ni' + str(Ni) + '_lam' + str(lam_ML) + '_class' + class_use_str[1:(len(class_use_str)-1):2] + '/'
    else:
        save_dir = '/home/mnorko/Documents/Tensorflow/causal_vae/results/imagenet/' + data_type + '_' + objective + '_zdim' + str(z_dim) + '_alpha' + str(alpha_dim) + '_No' + str(No) + '_Ni' + str(Ni) + '_lam' + str(lam_ML)  + '_cont_kl0.1/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
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
        sample_inputs_torch = torch.from_numpy(sample_inputs)
        sample_inputs_torch = sample_inputs_torch.permute(0,3,1,2).float().to(device)        
        ntrain = X.shape[0]



    
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
        encoder.apply(util.weights_init_normal)
        decoder = Decoder(x_dim, z_dim).to(device)
        decoder.apply(util.weights_init_normal)
    elif decoder_net == 'VAE_CNN' or 'VAE_fMNIST':
        from VAEModel_CNN import Decoder, Encoder
        encoder = Encoder(z_dim,c_dim,img_size).to(device)
        encoder.apply(weights_init_normal)
        decoder = Decoder(z_dim,c_dim,img_size).to(device)
        decoder.apply(util.weights_init_normal)
    elif decoder_net == 'VAE_Imagenet':
        checkpoint = torch.load('/home/mnorko/Documents/Tensorflow/causal_vae/results/imagenet/imagenet_JOINT_UNCOND_zdim40_alpha0_No20_Ni1_lam0.001/' + 'network_batch' + str(batch_size) + '.pt')
        from VAEModel_CNN_imagenet import Decoder, Encoder
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
        classifier.apply(util.weights_init_normal)
    elif classifier_net == 'twoHyperplane':
        from hyperplaneClassifierModel import TwoHyperplaneClassifier
        classifier = TwoHyperplaneClassifier(x_dim, y_dim, Pw1_torch, Pw2_torch, ksig=5.).to(device)
        classifier.apply(util.weights_init_normal)
    elif classifier_net == 'cnn':
        from cnnClassifierModel import CNN
        classifier = CNN(y_dim).to(device)
        batch_orig = 64
        checkpoint = torch.load('./mnist_batch64_lr0.1_class38/network_batch' + str(batch_orig) + '.pt')
        #checkpoint = torch.load('/home/mnorko/Documents/Tensorflow/causal_vae/results/mnist_batch64_lr0.1_class149/network_batch' + str(batch_orig) + '.pt')
        classifier.load_state_dict(checkpoint['model_state_dict_classifier'])
    elif classifier_net == 'cnn_fmnist':
        from cnnClassifierModel import CNN
        classifier = CNN(y_dim).to(device)
        batch_orig = 64
        checkpoint = torch.load('./fmnist_batch64_lr0.1_class034/network_batch' + str(batch_orig) + '.pt')
        #checkpoint = torch.load('/home/mnorko/Documents/Tensorflow/causal_vae/results/mnist_batch64_lr0.1_class149/network_batch' + str(batch_orig) + '.pt')
        classifier.load_state_dict(checkpoint['model_state_dict_classifier'])
    elif classifier_net == 'cnn_imagenet':
        from cnnImageNetClassifierModel import CNN
        classImagenetIdx = [366,340]
        classifier_model = models.vgg16_bn(pretrained=True)
        classifier = CNN(classifier_model,classImagenetIdx).to(device)
    else:
        print("Error: classifier should be one of: oneHyperplane twoHyperplane")

            
    # --- specify optimizer ---
    # NOTE: we only include the decoder parameters in the optimizer
    # because we don't want to update the classifier parameters
    if decoder_net == 'VAE' or decoder_net == 'VAE_CNN' or decoder_net == 'VAE_Imagenet' or decoder_net == 'VAE_fMNIST':
        params_use = list(decoder.parameters()) + list(encoder.parameters()) 
    else:
        params_use = list(decoder.parameters()) 
    optimizer_NN = torch.optim.Adam(params_use, lr=lr, betas=(b1, b2))
            
    # --- train ---
    start_time = time.time()
    for k in range(0, steps):
                
        # --- reset gradients to zero ---
        # (you always need to do this in pytorch or the gradients are
        # accumulated from one batch to the next)
        optimizer_NN.zero_grad()
        
        # --- compute negative log likelihood ---
        # randomly subsample batch_size samples of x

        randIdx = np.random.randint(0, ntrain, batch_size)
        if data_type == '2dpts':
            Xbatch = torch.from_numpy(X[randIdx,:]).float()
        elif data_type == 'mnist' or data_type == 'fmnist':
            Xbatch = torch.from_numpy(X[randIdx]).float()
            Xbatch = Xbatch.permute(0,3,1,2)
        elif data_type == 'imagenet':
            try:
                Xbatch, _ = dataiter.next() 
            except:
                dataiter = iter(trainloader)
                Xbatch, _ = dataiter.next() 
        Xbatch = Xbatch.to(device)
        if decoder_net == 'linGauss':
            nll = loss_functions.linGauss_NLL_loss(Xbatch,What,gamma)
        elif decoder_net == 'nonLinGauss':
            randBatch = torch.from_numpy(np.random.randn(z_num_samp,z_dim)).float()
            Xest,Xmu,Xlogvar = decoder(randBatch)
            Xcov = torch.exp(Xlogvar)
            nll = loss_functions.nonLinGauss_NLL_loss(Xbatch,Xmu,Xcov)
        elif decoder_net == 'VAE' or decoder_net == 'VAE_CNN' or decoder_net == 'VAE_Imagenet' or decoder_net == 'VAE_fMNIST':
            latent_out,mu,logvar = encoder(Xbatch)
            Xest = decoder(latent_out)
            nll, nll_mse, nll_kld = loss_functions.VAE_LL_loss(Xbatch,Xest,logvar,mu)
        
        # --- compute mutual information causal effect term ---
        if objective == "IND_UNCOND":
            causalEffect, ceDebug = causaleffect.ind_uncond(params, decoder, classifier, device, What=What)
        elif objective == "IND_COND":
            causalEffect, ceDebug = causaleffect.ind_cond(params, decoder, classifier, device, What=What)
        elif objective == "JOINT_UNCOND":
            causalEffect, ceDebug = causaleffect.joint_uncond(params, decoder, classifier, device, What=What)                        
        elif objective == "JOINT_COND":
            causalEffect, ceDebug = causaleffect.joint_cond(params, decoder, classifier, device, What=What)
        
        # --- compute gradients ---
        # total loss
        loss = use_ce*causalEffect + lam_ML*nll
        # backward step to compute the gradients
        loss.backward()
        if decoder_net == 'linGauss':
            # update What with the computed gradient
        	What.data.sub_(lr*What.grad.data)
        	# reset the What gradients to 0
        	What.grad.data.zero_()
        else:
            optimizer_NN.step()
    
        # --- debug information ---
        debug["loss"][k]         = loss.item()
        debug["loss_ce"][k]      = causalEffect.item()
        debug["loss_nll"][k]     = (lam_ML*nll).item()
        if decoder_net == 'VAE' or decoder_net == 'VAE_CNN' or decoder_net == 'VAE_Imagenet' or decoder_net == 'VAE_fMNIST':
            debug["loss_nll_mse"][k] = (lam_ML*nll_mse).item()
            debug["loss_nll_kld"][k] = (lam_ML*nll_kld).item()
        if debug_level > 0:
            print("[Step %d/%d] time: %4.2f  [CE: %g] [ML: %g] [loss: %g]" % \
                  (k, steps, time.time() - start_time, debug["loss_ce"][k],
                   debug["loss_nll"][k], debug["loss"][k]))
        if k % 1000 == 0:
            elif data_type == 'mnist' or data_type == 'imagenet' or data_type == 'fmnist':
                torch.save({
                    'step': k,
                    'model_state_dict_classifier': classifier.state_dict(),
                    'model_state_dict_encoder': encoder.state_dict(),
                    'model_state_dict_decoder': decoder.state_dict(),
                    'optimizer_state_dict': optimizer_NN.state_dict(),
                    'loss': loss,
                    }, save_dir + 'network_batch' + str(batch_size) + '.pt')
                sample_latent,mu,var = encoder(sample_inputs_torch)
                sample_inputs_torch_new = sample_inputs_torch.permute(0,2,3,1)
                sample_inputs_np = sample_inputs_torch_new.detach().cpu().numpy()
                sample_img = decoder(sample_latent)
                sample_latent_small = sample_latent[0:10,:]
                imgOut_real,probOut_real,latentOut_real = sweepLatentFactors(sample_latent_small,decoder,classifier,device,img_size,c_dim,y_dim,False)
                rand_latent = torch.from_numpy(np.random.randn(10,z_dim)).float().to(device)
                imgOut_rand,probOut_rand,latentOut_rand = sweepLatentFactors(rand_latent,decoder,classifier,device,img_size,c_dim,y_dim,False)
                samples = sample_img
                samples = samples.permute(0,2,3,1)
                samples = samples.detach().cpu().numpy()
                save_images(samples, [8,8],
                                    '{}train_{:04d}.png'.format(save_dir, k))
                sio.savemat(save_dir + 'sweepLatentFactors.mat',{'imgOut_real':imgOut_real,'probOut_real':probOut_real,'latentOut_real':latentOut_real,
                                                                 'imgOut_rand':imgOut_rand,'probOut_rand':probOut_rand,'latentOut_rand':latentOut_rand,'loss_total':debug["loss"][:k],'loss_ce':debug["loss_ce"][:k],'loss_nll':debug['loss_nll'][:k],'samples_out':samples,'sample_inputs':sample_inputs_np})
    
    # --- save all debug data ---
    debug["X"] = Xbatch.detach().cpu().numpy()
    if not decoder_net == 'linGauss':
        debug["Xest"] = Xest.detach().cpu().numpy()
    if save_output:
        datestamp = ''.join(re.findall(r'\d+', str(datetime.datetime.now())[:10]))
        timestamp = ''.join(re.findall(r'\d+', str(datetime.datetime.now())[11:19]))
        results_folder = save_dir
        matfilename = 'results_' + datestamp + '_' + timestamp + '.mat'
        sio.savemat(results_folder + matfilename, {'params' : params, 'data' : debug})
        if debug_level > 0:
            print('Finished saving data to ' + matfilename)
    
    if save_plot:
        print('Saving plot...')
        gif.save(frames, "results.gif", duration=100)
        print('Done!')
    
    return debug


lambda_change = [0.001]
obj_change = ["JOINT_UNCOND"]
alpha_change = [0]
z_dim_change = [1,2,3,4,5,6,7,8,9]
for obj_use  in obj_change:
    for z_use in z_dim_change:
        for lam_use in lambda_change:
            for alpha_use in alpha_change:
                trial_results = GenerativeCausalExplanation(
                    steps = 8000,
                    batch_size = 32,
                    lam_ML = lam_use,
                    decoder_net = "VAE_fMNIST",
                    classifier_net = "cnn_fmnist",
                    use_ce = False,
                    objective = obj_use,
                    data_type = "fmnist", 
                    break_up_ce= True,
                    x_dim = 3,
                    z_dim = z_use,
                    z_dim_true = z_use,
                    y_dim =3,
                    alpha_dim = alpha_use,
                    lr = 0.0001, 
                    No = 100,
                    Ni = 25,
                    randseed = 0,
                    save_output = True,
                    debug_level = 1,
                    debug_plot = True,
                    save_plot = False,
                    c_dim= 1,
                    img_size= 28)

