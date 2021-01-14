import numpy as np
import torch

"""
joint_uncond:
    Sample-based estimate of "joint, unconditional" causal effect, -I(alpha; Yhat).
Inputs:
    - params['Nalpha'] monte-carlo samples per causal factor
    - params['Nbeta']  monte-carlo samples per noncausal factor
    - params['K']      number of causal factors
    - params['L']      number of noncausal factors
    - params['M']      number of classes (dimensionality of classifier output)
    - decoder
    - classifier
    - device
Outputs:
    - negCausalEffect (sample-based estimate of -I(alpha; Yhat))
    - info['xhat']
    - info['yhat']
"""
def joint_uncond(params, decoder, classifier, device):
    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['M']).to(device)
    zs = np.zeros((params['Nalpha']*params['Nbeta'], params['z_dim']))
    for i in range(0, params['Nalpha']):
        alpha = np.random.randn(params['K'])
        zs = np.zeros((params['Nbeta'],params['z_dim']))  
        for j in range(0, params['Nbeta']):
            beta = np.random.randn(params['L'])
            zs[j,:params['K']] = alpha
            zs[j,params['K']:] = beta
        # decode and classify batch of Nbeta samples with same alpha
        xhat = decoder(torch.from_numpy(zs).float().to(device))
        yhat = classifier(xhat)[0]
        p = 1./float(params['Nbeta']) * torch.sum(yhat,0) # estimate of p(y|alpha)
        I = I + 1./float(params['Nalpha']) * torch.sum(torch.mul(p, torch.log(p+eps)))
        q = q + 1./float(params['Nalpha']) * p # accumulate estimate of p(y)
    I = I - torch.sum(torch.mul(q, torch.log(q+eps)))
    negCausalEffect = -I
    info = {"xhat" : xhat, "yhat" : yhat}
    return negCausalEffect, info


"""
joint_uncond_singledim:
    Sample-based estimate of "joint, unconditional" causal effect
    for single latent factor, -I(z_i; Yhat). Note the interpretation
    of params['Nalpha'] and params['Nbeta'] here: Nalpha is the number
    of samples of z_i, and Nbeta is the number of samples of the other
    latent factors.
Inputs:
    - params['Nalpha']
    - params['Nbeta']
    - params['K']
    - params['L']
    - params['M']
    - decoder
    - classifier
    - device
    - dim (i : compute -I(z_i; Yhat) **note: i is zero-indexed!**)
Outputs:
    - negCausalEffect (sample-based estimate of -I(z_i; Yhat))
    - info['xhat']
    - info['yhat']
"""
def joint_uncond_singledim(params, decoder, classifier, device, dim):
    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['M']).to(device)
    zs = np.zeros((params['Nalpha']*params['Nbeta'], params['z_dim']))
    for i in range(0, params['Nalpha']):
        z_fix = np.random.randn(1)
        zs = np.zeros((params['Nbeta'],params['z_dim']))  
        for j in range(0, params['Nbeta']):
            zs[j,:] = np.random.randn(params['K']+params['L'])
            zs[j,dim] = z_fix
        # decode and classify batch of Nbeta samples with same alpha
        xhat = decoder(torch.from_numpy(zs).float().to(device))
        yhat = classifier(xhat)[0]
        p = 1./float(params['Nbeta']) * torch.sum(yhat,0) # estimate of p(y|alpha)
        I = I + 1./float(params['Nalpha']) * torch.sum(torch.mul(p, torch.log(p+eps)))
        q = q + 1./float(params['Nalpha']) * p # accumulate estimate of p(y)
    I = I - torch.sum(torch.mul(q, torch.log(q+eps)))
    negCausalEffect = -I
    info = {"xhat" : xhat, "yhat" : yhat}
    return negCausalEffect, info