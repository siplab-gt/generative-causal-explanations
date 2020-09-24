from __future__ import division
import numpy as np
import torch


"""
information_flow:
    Compute the information flow between each latent factor and the classifier output.
Inputs:
    - params["alpha_dim"]
    - params["No"]
    - params["Ni"]
    - params["decoder_net"]
    - params["gamma"] (if decoder_net == 'linGauss')
    - decoder
    - classifier
    - What (if decoder_net = 'linGauss')
Outputs:
    - I_flow - information flow for each of the z dimensions
"""
def information_flow(params, decoder, classifier, device, What=None):
    #print('computing causal effect with No=%d, Ni=%d' % (params["No"],params["Ni"]))
    latent_vec = np.zeros((params["z_dim"]*params["No"]*params["Ni"],params["z_dim"]))
    count = 0
    I_flow = np.zeros((params["z_dim"]))
    for kk in range(params["z_dim"]):
        for m in range(params["No"]):
            ind_sample_val = np.random.randn(1)
            if params["break_up_ce"] == True:
                latent_vec = np.zeros((params["Ni"],params["z_dim"]))
                count =0
            for n in range(params["Ni"]):
                latent_vec_temp  = np.random.randn(params["z_dim"])
                latent_vec_temp[kk] = ind_sample_val
                latent_vec[count,:] = latent_vec_temp                    
                count += 1
            if params["break_up_ce"] == True:   
                latent_vec_torch = torch.from_numpy(latent_vec).float().to(device)
                Xhat_single = decoder(latent_vec_torch)
                yhat_single = classifier(Xhat_single)[0]
                if m == 0:
                    #Xhat = Xhat_single
                    yhat = yhat_single
                else:
                    #Xhat = torch.cat((Xhat,Xhat_single),0)
                    yhat = torch.cat((yhat,yhat_single),0)
        if params["break_up_ce"] == False:                
            latent_vec_torch = torch.from_numpy(latent_vec).float().to(device)
            if params["decoder_net"] == 'linGauss':
                Xhat = decoder(latent_vec_torch, What, params["gamma"])
            elif params["decoder_net"] == 'nonLinGauss':
                Xhat,Xmu,Xstd = decoder(latent_vec_torch)
            elif params["decoder_net"] == 'VAE' or params["decoder_net"] == 'VAE_CNN': 
                Xhat = decoder(latent_vec_torch)
            # This classifier outputs the label and the hyperplane classifier weights
            yhat = classifier(Xhat)[0]

        # Note that latent_vec is alpha_dim*No*Ni in length. The following
        # for loop runs through the loops over K' and N_o from our write up
        eps_add = 1e-8
        I_sum_p = 0.0
        qo_vec = torch.zeros(yhat.shape[1]).to(device)
        for m in range(0, params["No"]):
            y_use = yhat[int((m)*params["Ni"]):int((m+1)*params["Ni"]),:]
            q_vec =  1/float(params["Ni"])*torch.sum(y_use,0)
            q_log = torch.log(q_vec+eps_add*torch.ones_like(q_vec))
            I_sum_p += 1/float(params["No"])*torch.sum(torch.mul(q_vec,q_log))
            qo_vec = qo_vec + 1/float(params["No"])*q_vec
        qo_log = torch.log(qo_vec+eps_add*torch.ones_like(qo_vec))
        I_sum_p -= torch.sum(torch.mul(qo_vec,qo_log))
        I_flow[kk] = -I_sum_p.detach().cpu().numpy()
 
    return I_flow
        

"""
information_flow:
    Compute the information flow between a latent factor and the classifier output.
Inputs:
    - dim (latent factor index to compute information flow from)
    - params["alpha_dim"]
    - params["No"]
    - params["Ni"]
    - params["decoder_net"]
    - params["gamma"] (if decoder_net == 'linGauss')
    - decoder
    - classifier
    - What (if decoder_net = 'linGauss')
Outputs:
    - I_flow - information flow for each of the z dimensions
"""
def information_flow_singledim(dim, params, decoder, classifier, device, What=None):
    latent_vec = np.zeros((1*params["No"]*params["Ni"],params["z_dim"]))
    count = 0
    for m in range(params["No"]):
        ind_sample_val = np.random.randn(1)
        if params["break_up_ce"] == True:
            latent_vec = np.zeros((params["Ni"],params["z_dim"]))
            count = 0
        for n in range(params["Ni"]):
            latent_vec_temp  = np.random.randn(params["z_dim"])
            latent_vec_temp[dim] = ind_sample_val
            latent_vec[count,:] = latent_vec_temp                    
            count += 1
        if params["break_up_ce"] == True:   
            latent_vec_torch = torch.from_numpy(latent_vec).float().to(device)
            Xhat_single = decoder(latent_vec_torch)
            yhat_single = classifier(Xhat_single)[0]
            if m == 0:
                #Xhat = Xhat_single
                yhat = yhat_single
            else:
                #Xhat = torch.cat((Xhat,Xhat_single),0)
                yhat = torch.cat((yhat,yhat_single),0)
    if params["break_up_ce"] == False:                
        latent_vec_torch = torch.from_numpy(latent_vec).float().to(device)
        if params["decoder_net"] == 'linGauss':
            Xhat = decoder(latent_vec_torch, What, params["gamma"])
        elif params["decoder_net"] == 'nonLinGauss':
            Xhat,Xmu,Xstd = decoder(latent_vec_torch)
        elif params["decoder_net"] == 'VAE' or params["decoder_net"] == 'VAE_CNN': 
            Xhat = decoder(latent_vec_torch)
        # This classifier outputs the label and the hyperplane classifier weights
        yhat = classifier(Xhat)[0]
    # Note that latent_vec is alpha_dim*No*Ni in length. The following
    # for loop runs through the loops over K' and N_o from our write up
    eps_add = 1e-8
    I_sum_p = 0.0
    qo_vec = torch.zeros(yhat.shape[1]).to(device)
    for m in range(0, params["No"]):
        y_use = yhat[int((m)*params["Ni"]):int((m+1)*params["Ni"]),:]
        q_vec =  1/float(params["Ni"])*torch.sum(y_use,0)
        q_log = torch.log(q_vec+eps_add*torch.ones_like(q_vec))
        I_sum_p += 1/float(params["No"])*torch.sum(torch.mul(q_vec,q_log))
        qo_vec = qo_vec + 1/float(params["No"])*q_vec
    qo_log = torch.log(qo_vec+eps_add*torch.ones_like(qo_vec))
    I_sum_p -= torch.sum(torch.mul(qo_vec,qo_log))
    I_flow = I_sum_p.detach().cpu().numpy()
 
    return I_flow
        
