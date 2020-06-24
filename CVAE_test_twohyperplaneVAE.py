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
import gif
import os

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Decoder(nn.Module):
 
    def __init__(self, x_dim, z_dim):
        """
        Encoder initializer
        :param x_dim: dimension of the input
        :param z_dim: dimension of the latent representation
        """
        super(Decoder, self).__init__()
        self.model_enc = nn.Linear(int(z_dim),int(x_dim))

    def reparameterize(self, mu,gamma):
        eps = torch.randn_like(mu)
        return mu + eps*gamma
    
    def forward(self, x,What,gamma):
         # Run linear forward pass
        z = F.linear(x,What,None)
        z = self.reparameterize(z,gamma)
        return z
    

def CVAE_test_twohyperplaneVAE(
        model = "linGauss_multiHP", # currently not used
        steps          = 6000,
        batch_size     = 100,
        z_dim          = 4,
        z_dim_true     = 4,
        x_dim          = 10,
        y_dim          = 1,
        alpha_dim      = 2,
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
        decoder_net    = "linGauss",
        classifier_net = "hyperplane",
        randseed       = None,
        save_output    = False,
        debug_level    = 2,
        debug_plot     = False,
        save_plot      = False):

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
        "randseed"       : randseed,
        "save_output"    : save_output,
        "debug_level"    : debug_level}
    params["plot_batchsize"] = 500
    params["data_std"] = 2.
    if debug_level > 0:
        print("Parameters:")
        print(params)
    
    # Initialize arrays for storing performance data
    debug = {}
    vis_samples = 35
    debug["yhat_min"]          = np.zeros((steps))
    debug["loss"]              = np.zeros((steps))
    debug["loss_ce"]           = np.zeros((steps))
    debug["loss_nll"]          = np.zeros((steps))
    debug["loss_nll_logdet"]   = np.zeros((steps))
    debug["loss_nll_quadform"] = np.zeros((steps))
    debug["loss_nll_mse"]      = np.zeros((steps))
    debug["loss_nll_kld"]      = np.zeros((steps))
    debug["What"]              = np.zeros((z_dim,x_dim,steps))
    debug["xhat_a1"]           = np.zeros((x_dim,vis_samples,steps))
    debug["xhat_a2"]           = np.zeros((x_dim,vis_samples,steps))
    debug["Yhat_a1"]           = np.zeros((x_dim,vis_samples,steps))
    debug["Yhat_a2"]           = np.zeros((x_dim,vis_samples,steps))
            
    # seed random number generator
    if randseed is not None:
        if debug_level > 0:
            print('Setting random seed to ' + str(randseed) + '.')
        np.random.seed(randseed)
        torch.manual_seed(randseed)
        
    # --- construct projection matrices ---
    # 'true' orthogonal columns used to generate data from latent factors
    #Wsquare = sp.linalg.orth(np.random.rand(x_dim,x_dim))
    Wsquare = np.identity(x_dim)
    W = Wsquare
    w1 = np.expand_dims(W[:,0], axis=1)
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
    
    # --- initialize decoder ---
    if decoder_net == 'linGauss':
        from linGaussModel import Decoder
        decoder = Decoder(x_dim, z_dim)
    elif decoder_net == 'nonLinGauss':
        from VAEModel import Decoder_samp
        z_num_samp = No
        decoder = Decoder_samp(x_dim,z_dim)
    elif decoder_net == 'VAE':
        from VAEModel import Decoder, Encoder
        encoder = Encoder(x_dim,z_dim)
        encoder.apply(weights_init_normal)
        decoder = Decoder(x_dim, z_dim)
    else:
        print("Error: decoder_net should be one of: linGauss nonLinGauss VAE")
    decoder.apply(weights_init_normal)
    
    # --- initialize classifier ---
    if classifier_net == 'oneHyperplane':        
        from hyperplaneClassifierModel import OneHyperplaneClassifier
        classifier = OneHyperplaneClassifier(x_dim, y_dim, Pw1_torch,
                                             ksig=100.,
                                             a1 = w1.reshape((1,2)))
    elif classifier_net == 'twoHyperplane':
        from hyperplaneClassifierModel import TwoHyperplaneClassifier
        classifier = TwoHyperplaneClassifier(x_dim, y_dim, Pw1_torch, Pw2_torch,
                                             ksig=100.,
                                             a1 = w1.reshape((1,2)),
                                             a2 = w2.reshape((1,2)))
    else:
        print("Error: classifier should be one of: oneHyperplane twoHyperplane")
    classifier.apply(weights_init_normal)
    
    if decoder_net == 'linGauss':
    	What = Variable(torch.mul(torch.randn(x_dim, z_dim, dtype=torch.float),0.5), requires_grad=True)
    else:
        What = None
            
    # --- specify optimizer ---
    # NOTE: we only include the decoder parameters in the optimizer
    # because we don't want to update the classifier parameters
    if decoder_net == 'VAE':
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
        Xbatch = torch.from_numpy(X[randIdx,:]).float()
        if decoder_net == 'linGauss':
            nll = loss_functions.linGauss_NLL_loss(Xbatch,What,gamma)
        elif decoder_net == 'nonLinGauss':
            randBatch = torch.from_numpy(np.random.randn(z_num_samp,z_dim)).float()
            Xest,Xmu,Xlogvar = decoder(randBatch)
            Xcov = torch.exp(Xlogvar)
            nll = loss_functions.nonLinGauss_NLL_loss(Xbatch,Xmu,Xcov)
        elif decoder_net == 'VAE':
            latent_out,mu,logvar = encoder(Xbatch)
            Xest = decoder(latent_out)
            nll, nll_mse, nll_kld = loss_functions.VAE_LL_loss(Xbatch,Xest,logvar,mu)
        
        # --- compute mutual information causal effect term ---
        if objective == "IND_UNCOND":
            causalEffect, ceDebug = causaleffect.ind_uncond(params, decoder, classifier, What=What)
        elif objective == "IND_COND":
            causalEffect, ceDebug = causaleffect.ind_cond(params, decoder, classifier, What=What)
        elif objective == "JOINT_UNCOND":
            causalEffect, ceDebug = causaleffect.joint_uncond(params, decoder, classifier, What=What)                        
        elif objective == "JOINT_COND":
            causalEffect, ceDebug = causaleffect.joint_cond(params, decoder, classifier, What=What)
        yhat_np = ceDebug["yhat"].detach().numpy()
        
        # --- compute gradients ---
        # total loss
        if use_ce:
            loss = causalEffect + lam_ML*nll
        else:
            loss = lam_ML*nll
        # backward step to compute the gradients
        loss.backward()
        if decoder_net == 'linGauss':
            # update What with the computed gradient
            What.data.sub_(lr*What.grad.data)
        	# reset the What gradients to 0
            What.grad.data.zero_()
        else:
            optimizer_NN.step()
    
        # --- save debug info for this step ---
        debug["yhat_min"][k]     = yhat_np.min()
        # components of objective
        debug["loss"][k]         = loss.detach().numpy()
        debug["loss_ce"][k]      = causalEffect.detach().numpy()
        debug["loss_nll"][k]     = (lam_ML*nll).detach().numpy()
        if debug_level > 1:# and k == steps-1:
            if decoder_net == 'linGauss':
                debug["What"][:,:,k] = What.detach().numpy()
            elif decoder_net == 'VAE':
                debug["loss_nll_mse"][k] = (lam_ML*nll_mse).detach().numpy()
                debug["loss_nll_kld"][k] = (lam_ML*nll_kld).detach().numpy()
                v_sweep = np.linspace(-5.,5.,vis_samples)
                # samples Yhat | x[1], x[2], x[3]
                for ix1, x1 in enumerate(v_sweep):
                    for ix2, x2 in enumerate(v_sweep):
                        xs = np.zeros((250,3))
                        xs[:,0] = x1
                        xs[:,1] = x2
                        xs[:,2] = params["data_std"]*np.random.randn(250)
                        yhats = classifier(torch.from_numpy(xs).float())[0]
                        debug["yhat_x1x2"][ix1,ix2,k] = np.mean(yhats.detach().numpy()[:,0])
                for ix1, x1 in enumerate(v_sweep):
                    for ix3, x3 in enumerate(v_sweep):
                        xs = np.zeros((250,3))
                        xs[:,0] = x1
                        xs[:,1] = params["data_std"]*np.random.randn(250)
                        xs[:,2] = x3
                        yhats = classifier(torch.from_numpy(xs).float())[0]
                        debug["yhat_x1x3"][ix1,ix3,k] = np.mean(yhats.detach().numpy()[:,0])
                for ix2, x2 in enumerate(v_sweep):
                    for ix3, x3 in enumerate(v_sweep):
                        xs = np.zeros((250,3))
                        xs[:,0] = params["data_std"]*np.random.randn(250)
                        xs[:,1] = x2
                        xs[:,2] = x3
                        yhats = classifier(torch.from_numpy(xs).float())[0]
                        debug["yhat_x2x3"][ix2,ix3,k] = np.mean(yhats.detach().numpy()[:,0])
                # samples x | alpha[1], alpha[2], beta
                for ia1, a1 in enumerate(v_sweep):
                    for ia2, a2 in enumerate(v_sweep):
                        zs = np.zeros((250,3))
                        zs[:,0] = a1
                        zs[:,1] = a2
                        zs[:,2] = np.random.randn(250)
                        xs = decoder(torch.from_numpy(zs).float()).detach().numpy()
                        debug["x1_a1a2"][ia1,ia2,k] = np.mean(xs[:,0])
                        debug["x2_a1a2"][ia1,ia2,k] = np.mean(xs[:,1])
                        debug["x3_a1a2"][ia1,ia2,k] = np.mean(xs[:,2])
                for ia1, a1 in enumerate(v_sweep):
                    for ib, b in enumerate(v_sweep):
                        zs = np.zeros((250,3))
                        zs[:,0] = a1
                        zs[:,1] = np.random.randn(250)
                        zs[:,2] = b
                        xs = decoder(torch.from_numpy(zs).float()).detach().numpy()
                        debug["x1_a1b"][ia1,ib,k] = np.mean(xs[:,0])
                        debug["x2_a1b"][ia1,ib,k] = np.mean(xs[:,1])
                        debug["x3_a1b"][ia1,ib,k] = np.mean(xs[:,2])
                for ia2, a2 in enumerate(v_sweep):
                    for ib, b in enumerate(v_sweep):
                        zs = np.zeros((250,3))
                        zs[:,0] = np.random.randn(250)
                        zs[:,1] = a2
                        zs[:,2] = b
                        xs = decoder(torch.from_numpy(zs).float()).detach().numpy()
                        debug["x1_a2b"][ia2,ib,k] = np.mean(xs[:,0])
                        debug["x2_a2b"][ia2,ib,k] = np.mean(xs[:,1])
                        debug["x3_a2b"][ia2,ib,k] = np.mean(xs[:,2])
                # samples yhat | alpha[1], alpha[2], beta
                for ia1, a1 in enumerate(v_sweep):
                    for ia2, a2 in enumerate(v_sweep):
                        zs = np.zeros((250,3))
                        zs[:,0] = a1
                        zs[:,1] = a2
                        zs[:,2] = params["data_std"]*np.random.randn(250)
                        xs = decoder(torch.from_numpy(zs).float())
                        yhats = classifier(xs)[0]
                        debug["yhat_a1a2"][ia1,ia2,k] = np.mean(yhats.detach().numpy()[:,0])
                for ia1, a1 in enumerate(v_sweep):
                    for ib, b in enumerate(v_sweep):
                        zs = np.zeros((250,3))
                        zs[:,0] = a1
                        zs[:,1] = np.random.randn(250)
                        zs[:,2] = b
                        xs = decoder(torch.from_numpy(zs).float())
                        yhats = classifier(xs)[0]
                        debug["yhat_a1b"][ia1,ib,k] = np.mean(yhats.detach().numpy()[:,0])
                for ia2, a2 in enumerate(v_sweep):
                    for ib, b in enumerate(v_sweep):
                        zs = np.zeros((250,3))
                        zs[:,0] = np.random.randn(250)
                        zs[:,1] = a2
                        zs[:,2] = b
                        xs = decoder(torch.from_numpy(zs).float())
                        yhats = classifier(xs)[0]
                        debug["yhat_a2b"][ia2,ib,k] = np.mean(yhats.detach().numpy()[:,0])
                # samples x | alpha[i]
                for ia1, a1 in enumerate(range(-3,4)):
                    zs = np.zeros((250,3))
                    zs[:,0] = a1
                    zs[:,1] = np.random.randn(250)
                    zs[:,2] = np.random.randn(250)
                    xs = decoder(torch.from_numpy(zs).float())
                    debug["xhat_a1"][ia1,:,:,k] = xs.detach().numpy().transpose()
                for ia2, a2 in enumerate(range(-3,4)):
                    zs = np.zeros((250,3))
                    zs[:,0] = np.random.randn(250)
                    zs[:,1] = a2
                    zs[:,2] = np.random.randn(250)
                    xs = decoder(torch.from_numpy(zs).float())
                    debug["xhat_a2"][ia2,:,:,k] = xs.detach().numpy().transpose()
                
        
        # --- print step information ---
        if debug_level > 0:
            print("[Step %d/%d] time: %4.2f  [CE: %g] [ML: %g] [loss: %g]" % \
                  (k, steps, time.time() - start_time, debug["loss_ce"][k],
                   debug["loss_nll"][k], debug["loss"][k]))
        
        # --- debug plot ---
        if debug_plot and k % 500 == 0:
            print('Generating plot frame...')
            # generate samples of p(x | alpha_i = alphahat_i)
            decoded_points = {}
            decoded_points["ai_vals"] = lfplot_aihat_vals
            decoded_points["samples"] = np.zeros((2,lfplot_nsamp,len(lfplot_aihat_vals),params["z_dim"]))
            for l in range(params["z_dim"]): # loop over latent dimensions
                for i, aihat in enumerate(lfplot_aihat_vals): # loop over fixed aihat values
                    for m in range(lfplot_nsamp): # loop over samples to generate
                        z = np.random.randn(params["z_dim"])
                        z[l] = aihat
                        x = decoder(torch.from_numpy(z).float(), What, gamma)
                        decoded_points["samples"][:,m,i,l] = x.detach().numpy()
            frame = plotting.debugPlot_frame(X, ceDebug["Xhat"], W, What, k,
                                             steps, debug, params, classifier,
                                             decoded_points)
            if save_plot:
                frames.append(frame)
    
    # --- save all debug data ---
    debug["X"] = X
    if save_output:
        datestamp = ''.join(re.findall(r'\d+', str(datetime.datetime.now())[:10]))
        timestamp = ''.join(re.findall(r'\d+', str(datetime.datetime.now())[11:19]))
        results_folder = './results/tests_kSig5_lr0001_' + objective + '_lam' \
            + str(lam_ML) + '_No' + str(No) + '_Ni' + str(Ni) + '_' \
            + datestamp + '_' + timestamp + '/'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        matfilename = 'results_' + datestamp + '_' + timestamp + '.mat'
        sio.savemat(results_folder + matfilename, {'params' : params, 'data' : debug})
        if debug_level > 0:
            print('Finished saving data to ' + matfilename)
    
    if save_plot:
        print('Saving plot...')
        gif.save(frames, "results.gif", duration=100)
        print('Done!')
    
    return debug, params
