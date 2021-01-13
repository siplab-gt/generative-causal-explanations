from __future__ import division
import time
import datetime
import re

import numpy as np
import scipy.io as sio

from torch.autograd import Variable
import torch

import loss_functions
import old.causaleffect
import util

import os

from util import *
from load_mnist import *

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def CVAE(steps           = 20000,
         batch_size      = 100,
         z_dim           = 6,
         z_dim_true      = 4,  # used for linear/gaussian data model
         x_dim           = 10, # used for linear/gaussian data model
         alpha_dim       = 4,
         ntrain          = 5000, # used for linear/gaussian data model
         No              = 15, #TODO
         Ni              = 15,
         lam_ML          = 0.000001,
         gamma           = 0.001,
         lr              = 0.0001,
         b1              = 0.5,
         b2              = 0.999,
         use_ce          = True,
         objective       = 'JOINT_UNCOND',
         gen_model       = 'VAE_CNN', # one of 'linGauss' 'VAE' 'VAE_CNN'
         classifier      = 'CNN', # one of 'oneHyperplane' 'twoHyperplane' 'CNN'
         classifier_path = './pretrained_models/mnist_38_classifier/model.pt',
         dataset         = 'mnist', # one of 'linGauss', 'mnist', 'fmnist'
         class_use       = np.array([3,8]),
         break_up_ce     = True, # if true, break up computation of network forward passes by causal factors (alphas)
         randseed        = None,
         save_output     = False,
         debug_level     = 2,
         c_dim           = 1,
         img_size        = 28):


    # --- initialization ---
    # save parameters (for easy saving and input to causal effect calculation)
    params = {'steps' : steps, 'batch_size' : batch_size, 'z_dim' : z_dim, 'z_dim_true' : z_dim_true,
              'x_dim' : x_dim, 'alpha_dim' : alpha_dim, 'ntrain' : ntrain, 'No' : No,
              'Ni' : Ni, 'lam_ML' : lam_ML, 'gamma' : gamma, 'lr' : lr, 'b1' : b1, 'b2' : b2, 'use_ce' : use_ce,
              'objective' : objective, 'decoder_net' : gen_model, 'classifier_net' : classifier,
              'dataset' : dataset, 'class_use' : class_use, 'break_up_ce' : break_up_ce,
              'randseed' : randseed, 'save_output' : save_output, 'debug_level' : debug_level,
              'c_dim' : c_dim, 'img_size' : img_size}
    data_std = 2.0 # standard deviation of generated linear/gaussian case
    if dataset == 'linGauss':
        break_up_ce = False
    if debug_level > 0:
        print("Parameters:")
        print(params)
    # initialize arrays
    debug = {}
    debug["loss"]              = np.zeros((steps))
    debug["loss_ce"]           = np.zeros((steps))
    debug["loss_nll"]          = np.zeros((steps))
    debug["loss_nll_logdet"]   = np.zeros((steps))
    debug["loss_nll_quadform"] = np.zeros((steps))
    debug["loss_nll_mse"]      = np.zeros((steps))
    debug["loss_nll_kld"]      = np.zeros((steps))
    if gen_model == 'linGauss':
        for i in range(z_dim):
            for j in range(z_dim_true):
                debug["cossim_w%dwhat%d"%(j+1,i+1)] = np.zeros((steps))
            for j in range(i+1,z_dim):
                debug["cossim_what%dwhat%d"%(i+1,j+1)] = np.zeros((steps))
    # output directory
    class_use_str = np.array2string(class_use)    
    y_dim = class_use.shape[0]
    params['y_dim'] = y_dim
    newClass = range(0,y_dim)
    save_folder_root = './pretrained_models'
    save_dir = os.path.join(save_folder_root, dataset + '_' + class_use_str[1:(len(class_use_str)-1):2]
        + '_vae' + '_zdim' + str(z_dim) + '_alphadim' + str(alpha_dim) + '_lambda' + str(lam_ML))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # seed random number generator
    if randseed is not None:
        if debug_level > 0:
            print('Setting random seed to ' + str(randseed) + '.')
        np.random.seed(randseed)
        torch.manual_seed(randseed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # --- generate data ---
    if dataset == 'linGauss':    
        # construct projection matrices
        # 'true' orthogonal columns used to generate data from latent factors
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
        # construct data (ntrain instances of alpha and x)
        Alpha = data_std*np.random.randn(ntrain, z_dim_true) #TODO
        X = np.matmul(Alpha, W.T)
    elif dataset == 'mnist':
        test_size = 64
        X, Y, tridx = load_mnist_classSelect('train',class_use,newClass)
        vaX, vaY, vaidx = load_mnist_classSelect('val',class_use,newClass)
        sample_inputs = vaX[0:test_size]
        sample_inputs_torch = torch.from_numpy(sample_inputs)
        sample_inputs_torch = sample_inputs_torch.permute(0,3,1,2).float().to(device)        
        ntrain = X.shape[0]
    elif dataset == 'fmnist':
        test_size = 64
        X, Y, tridx = load_fashion_mnist_classSelect('train',class_use,newClass)
        vaX, vaY, vaidx = load_fashion_mnist_classSelect('val',class_use,newClass)
        sample_inputs = vaX[0:test_size]
        sample_inputs_torch = torch.from_numpy(sample_inputs)
        sample_inputs_torch = sample_inputs_torch.permute(0,3,1,2).float().to(device)        
        ntrain = X.shape[0]
    

    # --- initialize VAE ---
    if gen_model == 'linGauss':
        from linGaussModel import Decoder
        decoder = Decoder(x_dim, z_dim).to(device)
        What = Variable(torch.mul(torch.randn(x_dim, z_dim, dtype=torch.float),0.5), requires_grad=True)
    elif gen_model == 'VAE':
        from models.VAE import Decoder, Encoder
        encoder = Encoder(x_dim,z_dim).to(device)
        encoder.apply(weights_init_normal)
        decoder = Decoder(x_dim, z_dim).to(device)
        decoder.apply(weights_init_normal)
        What = None
    elif gen_model == 'VAE_CNN':
        from models.CVAE import Decoder, Encoder
        encoder = Encoder(z_dim,c_dim,img_size**2).to(device)
        encoder.apply(weights_init_normal)
        decoder = Decoder(z_dim,c_dim,img_size**2).to(device)
        decoder.apply(weights_init_normal)
        What = None
    else:
        print("Error: decoder_net must be one of 'linGauss' 'VAE' 'VAE_CNN'!")


    # --- initialize classifier ---
    if classifier == 'oneHyperplane':        
        from hyperplaneClassifierModel import OneHyperplaneClassifier
        classifier = OneHyperplaneClassifier(x_dim, y_dim, Pw1_torch, ksig=5.0).to(device)
        classifier.apply(weights_init_normal)
    elif classifier == 'twoHyperplane':
        from hyperplaneClassifierModel import TwoHyperplaneClassifier
        classifier = TwoHyperplaneClassifier(x_dim, y_dim, Pw1_torch, Pw2_torch, ksig=5.0).to(device)
        classifier.apply(weights_init_normal)
    elif classifier == 'CNN':
        from models.CNN_classifier import CNN
        classifier = CNN(y_dim).to(device)
        checkpoint = torch.load(classifier_path)
        classifier.load_state_dict(checkpoint['model_state_dict_classifier'])
    else:
        print("Error: classifier should be one of 'oneHyperplane' 'twoHyperplane' 'CNN'!")
    

    # --- specify optimizer ---
    # NOTE: we only include the decoder parameters in the optimizer
    # because we don't want to update the classifier parameters
    if gen_model == 'VAE' or gen_model == 'VAE_CNN':
        params_use = list(decoder.parameters()) + list(encoder.parameters()) 
    else:
        params_use = list(decoder.parameters()) 
    optimizer = torch.optim.Adam(params_use, lr=lr, betas=(b1, b2))
    

    # --- train ---
    start_time = time.time()
    for k in range(0, steps):        
        # reset gradients to zero
        optimizer.zero_grad()
        # compute negative log likelihood
        # (randomly subsample batch_size samples of x)
        randIdx = np.random.randint(0, ntrain, batch_size)
        if dataset == 'linGauss':
            Xbatch = torch.from_numpy(X[randIdx,:]).float()
        elif dataset == 'mnist' or dataset == 'fmnist':
            Xbatch = torch.from_numpy(X[randIdx]).float()
            Xbatch = Xbatch.permute(0,3,1,2)
        elif dataset == 'imagenet':
            try:
                Xbatch, _ = dataiter.next() 
            except:
                dataiter = iter(trainloader)
                Xbatch, _ = dataiter.next() 
        Xbatch = Xbatch.to(device)
        if gen_model == 'linGauss':
            nll = loss_functions.linGauss_NLL_loss(Xbatch,What,gamma)
        elif gen_model == 'VAE' or gen_model == 'VAE_CNN':
            latent_out, mu, logvar = encoder(Xbatch)
            Xest = decoder(latent_out)
            nll, nll_mse, nll_kld = loss_functions.VAE_LL_loss(Xbatch,Xest,logvar,mu)
        # compute causal effect [(conditional) mutual information]
        if objective == 'IND_UNCOND':
            causalEffect, ceDebug = old.causaleffect.ind_uncond(params, decoder, classifier, device, What=What)
        elif objective == 'IND_COND':
            causalEffect, ceDebug = old.causaleffect.ind_cond(params, decoder, classifier, device, What=What)
        elif objective == 'JOINT_UNCOND':
            causalEffect, ceDebug = old.causaleffect.joint_uncond(params, decoder, classifier,device, What=What)
        elif objective == 'JOINT_COND':
            causalEffect, ceDebug = old.causaleffect.joint_cond(params, decoder, classifier, device, What=What)
        # compute gradients
        loss = use_ce*causalEffect + lam_ML*nll
        loss.backward()
        if gen_model == 'linGauss':
        	What.data.sub_(lr*What.grad.data)
        	What.grad.data.zero_()
        else:
            optimizer.step()
        # save debug info for this step
        debug["loss"][k]         = loss.item()
        debug["loss_ce"][k]      = causalEffect.item()
        debug["loss_nll"][k]     = (lam_ML*nll).item()
        if gen_model == 'VAE' or gen_model == 'VAE_CNN':
            debug["loss_nll_mse"][k] = (lam_ML*nll_mse).item()
            debug["loss_nll_kld"][k] = (lam_ML*nll_kld).item()
        if debug_level > 1:
            if gen_model == 'linGauss':
                Wnorm = W / np.linalg.norm(W, axis=0, keepdims=True)
                Whatnorm = What.detach().numpy() / np.linalg.norm(What.detach().numpy(), axis=0, keepdims=True)
                # cosine similarities between columns of W and What
                for i in range(z_dim):
                    for j in range(z_dim_true):
                        debug["cossim_w%dwhat%d"%(j+1,i+1)][k] = np.matmul(Wnorm[:,j],Whatnorm[:,i])
                    for j in range(i+1,z_dim):
                        debug["cossim_what%dwhat%d"%(i+1,j+1)][k] = np.matmul(Whatnorm[:,i],Whatnorm[:,j])
        # save model
        torch.save({
            'step': k,
            'model_state_dict_encoder' : encoder.state_dict(),
            'model_state_dict_decoder' : decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            }, os.path.join(save_dir, 'model.pt'))
        # print debug info for this step
        if debug_level > 0:
            print("[Step %d/%d] time: %4.2f  [CE: %g] [ML: %g] [loss: %g]" % \
                  (k, steps, time.time() - start_time, debug["loss_ce"][k],
                   debug["loss_nll"][k], debug["loss"][k]))
    

    # --- save debug data ---
    debug["X"] = Xbatch.detach().cpu().numpy()
    if not gen_model == 'linGauss':
        debug["Xest"] = Xest.detach().cpu().numpy()
    if save_output:
        datestamp = ''.join(re.findall(r'\d+', str(datetime.datetime.now())[:10]))
        timestamp = ''.join(re.findall(r'\d+', str(datetime.datetime.now())[11:19]))
        matfilename = 'results_' + datestamp + '_' + timestamp + '.mat'
        sio.savemat(os.path.join(save_dir, matfilename), {'params' : params, 'data' : debug})
        if debug_level > 0:
            print('Finished saving data to ' + matfilename)
    return debug


lambda_sweep = [0.05]
obj_sweep = ['JOINT_UNCOND']
alpha_sweep = [2]
z_dim_sweep = [6]
for obj_use  in obj_sweep:
    for z_use in z_dim_sweep:
        for lam_use in lambda_sweep:
            for alpha_use in alpha_sweep:
                trial_results = CVAE(
                    steps = 8000,
                    batch_size = 64,
                    lam_ML = lam_use,
                    gen_model = 'VAE_CNN',
                    classifier = 'CNN',
                    classifier_path = './pretrained_models/fmnist_034_classifier/model.pt',
                    class_use = np.array([0,3,4]),
                    use_ce = True,
                    gamma = 0.001,
                    lr = 5e-4,
                    b1 = 0.5,
                    b2 = 0.999,
                    objective = obj_use,
                    dataset = 'fmnist', 
                    break_up_ce = True,
                    z_dim = z_use,
                    alpha_dim = alpha_use,
                    No = 100,
                    Ni = 25,
                    randseed = 0,
                    save_output = True,
                    debug_level = 1,
                    c_dim = 1,
                    img_size = 28)
