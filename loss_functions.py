import torch

def linGauss_NLL_loss(Xbatch,What,gamma):
    x_dim = What.size()[0]
    batch_size = Xbatch.size()[0]
    # regularized covariance estimate: What*What^T + gamma*I
    Sigmahat = torch.matmul(What,torch.t(What)) + gamma*torch.eye(x_dim)
    # regularized inverse covariance estimate: (What*What^T + gamma*I)^-1
    Sigmainvhat = torch.inverse(torch.matmul(What,torch.t(What)) + gamma*torch.eye(x_dim))
    # negative log likelihood: nll \propto log|Sigma| + x^T Sigma^-1 x
    nll_logdet = torch.log(torch.det(Sigmahat))
    # NOTE: the multiplication here may seem strange but remember that
    # X_torch is (batch_size,x_dim), so each row contains an x_i. This
    # is equal to 1/batch_size sum_i x_i^T Sigma^-1 x_i
    XSigmainvhatXT = torch.matmul(Xbatch,torch.matmul(Sigmainvhat,torch.t(Xbatch)))
    nll_quadform = 1/float(batch_size) * torch.sum(torch.diag(XSigmainvhatXT))
    nll = nll_logdet + nll_quadform
    
    return nll
    
def nonLinGauss_NLL_loss(Xbatch,Xmu,Xcov_vec):
    num_samp = Xmu.size()[0]
    batch_size = Xbatch.size()[0]
    nll_sum = 0.0
    Xcov = torch.diag_embed(Xcov_vec)
    covInv = torch.inverse(Xcov)
    for k in range(0,batch_size):
        meanDiff = Xbatch[k,:]-Xmu
        expPow = -0.5*torch.matmul(torch.matmul(torch.unsqueeze(meanDiff,1), covInv),torch.unsqueeze(meanDiff,2))[:,0,0]
        likelihood = 1/float(num_samp)*torch.sum(torch.mul(torch.pow(torch.prod(Xcov_vec,1),-0.5),torch.exp(expPow)))
        nll_sum = nll_sum + torch.log(likelihood)
    nll = nll_sum/float(batch_size)
    return nll
    
def VAE_LL_loss(Xbatch,Xest,logvar,mu):
    batch_size = Xbatch.shape[0]
    sse_loss = torch.nn.MSELoss(reduction = 'sum') # sum of squared errors
    KLD = 1./batch_size * -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    mse = 1./batch_size * sse_loss(Xest,Xbatch)
    auto_loss = mse + KLD
    return auto_loss, mse, KLD