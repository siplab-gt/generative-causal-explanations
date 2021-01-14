from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch


class OneHyperplaneClassifier(nn.Module):
 
    """
        Initialize classifier
        Inputs:
         - x_dim : dimension of the input
         - y_dim : number of classes in the classifier
         - P    : projection matrix, projects onto subspace
        Optional inputs:
         - a1    : slope of hyperplane (y_dim,x_dim) np.array
         - b1    : bias of hyperplane (y_dim,) np.array
         - ksig : 'slope' of sigmoid function
    """
    def __init__(self, x_dim, y_dim, P, a=None, b=None, ksig=5):
        super(OneHyperplaneClassifier, self).__init__()
        
        if a is None:
            self.a = Parameter(torch.matmul(torch.randn(int(y_dim),int(x_dim)),torch.t(P)))
        else:
            assert a.shape == (int(y_dim), int(x_dim))
            self.a = Parameter(torch.matmul(torch.Tensor(a),torch.t(P)))
                
        if b is None:
            self.b = Parameter(torch.Tensor(int(y_dim)))
            nn.init.constant_(self.b,0.0)
        else:
            assert b.shape == (int(y_dim))
            self.b = Parameter(torch.Tensor(b))
            
        self.ksig = ksig
    
    """
        Perform classification: yhat = sig(k*(a1^Tx-b1))*sig(k*(a2^Tx-b2))
        
        Inputs:
         - x : input data sample
         
        Outputs:
         - yhat : ( p(yhat=0), p(yhat=1) )
         - a   : slope of hyperplane
    """
    def forward(self, x):
        # Project x into subspace defined by W_1
        # F.linear(x, A, b) performs x A^T + b. We compute x weight^T - bias
        z = F.linear(x, self.a, -1*self.b)
        # Take the sigmoid to make the result between 0 and 1
        yhat_class0 = torch.sigmoid(self.ksig*z)
        yhat_class1 = 1. - yhat_class0
        yhat = torch.cat((yhat_class0,yhat_class1),1)
        return yhat, self.a


class TwoHyperplaneClassifier(nn.Module):
 
    """
        Initialize classifier
        Inputs:
         - x_dim : dimension of the input
         - y_dim : number of classes in the classifier
         - P1    : projection matrix, projects onto first subspace
         - P2    : projection matrix, projects onto second subspace
        Optional inputs:
         - a1    : slope of 1st hyperplane (1,x_dim) np.array
         - a2    : slope of 2nd hyperplane (1,x_dim) np.array
         - b1    : bias of 1st hyperplane np.array
         - b2    : bias of 2nd hyperplane np.array
         - ksig  : 'slope' of sigmoids
    """
    def __init__(self, x_dim, y_dim, P1, P2, a1=None, a2=None, b1=None, b2=None, ksig=5):

        super(TwoHyperplaneClassifier, self).__init__()
        
        if a1 is None:
            self.a1 = Parameter(torch.matmul(torch.randn(1,int(x_dim)),torch.t(P1)))
        else:
            self.a1 = Parameter(torch.matmul(torch.Tensor(a1),torch.t(P1)))
            
        if a2 is None:
            self.a2 = Parameter(torch.matmul(torch.randn(1,int(x_dim)),torch.t(P2)))
        else:
            self.a2 = Parameter(torch.matmul(torch.Tensor(a2),torch.t(P2)))
        
        if b1 is None:
            self.b1 = Parameter(torch.Tensor(1))
            nn.init.constant_(self.b1,0.0)
        else:
            assert b1.shape == (int(y_dim))
            self.b1 = Parameter(torch.Tensor(b1))
            
        if b2 is None:
            self.b2 = Parameter(torch.Tensor(1))
            nn.init.constant_(self.b2,0.0)
        else:
            assert b2.shape == (int(y_dim))
            self.b2 = Parameter(torch.Tensor(b2))
            
        self.ksig = ksig
    
    """
        Perform classification: yhat = sig(k*(a1^Tx-b1))*sig(k*(a2^Tx-b2))
        Inputs:
        - x : input data sample
        Outputs:
        - yhat : ( p(yhat=0), p(yhat=1) )
        - a1   : slope of 1st hyperplane
        - a2   : slope of 2nd hyperplane
    """
    def forward(self, x):
        # project x into subspace defined by W_1
        # F.linear(x, A, b) performs x A^T + b. We compute x weight^T - bias
        z1 = F.linear(x, self.a1, -1*self.b1)
        z2 = F.linear(x, self.a2, -1*self.b2)
        # take the sigmoid to make the result between 0 and 1
        yhat_class0 = torch.sigmoid(self.ksig*z1)*torch.sigmoid(self.ksig*z2)
        yhat_class1 = 1. - yhat_class0
        yhat = torch.cat((yhat_class0,yhat_class1),1)
        return yhat, self.a1, self.a2


class XORHyperplaneClassifier(nn.Module):
 
    """
        Initialize classifier
        Inputs:
         - x_dim : dimension of the input
         - y_dim : number of classes in the classifier
         - P1    : projection matrix, projects onto first subspace
         - P2    : projection matrix, projects onto second subspace 
        Optional inputs:
         - a1    : slope of 1st hyperplane (y_dim,x_dim) np.array
         - a2    : slope of 2nd hyperplane (y_dim,x_dim) np.array
         - b1    : bias of 1st hyperplane (y_dim,) np.array
         - b2    : bias of 2nd hyperplane (y_dim,) np.array
    """
    def __init__(self, x_dim, y_dim, P1, P2, a1=None, a2=None, b1=None, b2=None, ksig=5):

        super(XORHyperplaneClassifier, self).__init__()
        
        if a1 is None:
            self.a1 = Parameter(torch.matmul(torch.randn(int(y_dim),int(x_dim)),torch.t(P1)))
        else:
            assert a1.shape == (int(y_dim), int(x_dim))
            self.a1 = Parameter(torch.matmul(torch.Tensor(a1),torch.t(P1)))
            
        if a2 is None:
            self.a2 = Parameter(torch.matmul(torch.randn(int(y_dim),int(x_dim)),torch.t(P2)))
        else:
            assert a2.shape == (int(y_dim), int(x_dim))
            self.a2 = Parameter(torch.matmul(torch.Tensor(a2),torch.t(P2)))
        
        if b1 is None:
            self.b1 = Parameter(torch.Tensor(int(y_dim)))
            nn.init.constant_(self.b1,0.0)
        else:
            assert b1.shape == (int(y_dim))
            self.b1 = Parameter(torch.Tensor(b1))
            
        if b2 is None:
            self.b2 = Parameter(torch.Tensor(int(y_dim)))
            nn.init.constant_(self.b2,0.0)
        else:
            assert b2.shape == (int(y_dim))
            self.b2 = Parameter(torch.Tensor(b2))
            
        self.ksig = ksig
    
    """
        Perform classification: yhat = (a1^Tx > b1) XOR (a2^Tx > b2)
        Inputs:
         - x : input data sample     
        Outputs:
         - yhat : ( p(yhat=0), p(yhat=1) )
         - a1   : slope of 1st hyperplane
         - a2   : slope of 2nd hyperplane
    """
    def forward(self, x):
        # project x into subspace defined by W_1
        # F.linear(x, A, b) performs x A^T + b. We compute x weight^T - bias
        z1 = F.linear(x, self.a1, -1*self.b1)
        z2 = F.linear(x, self.a2, -1*self.b2)
        # take the sigmoid to make the result between 0 and 1
        yhat_class0 = ((z1 > 0) ^ (z2 > 0)).float()
        yhat_class1 = 1. - yhat_class0
        yhat = torch.cat((yhat_class0,yhat_class1),1)
        return yhat, self.a1, self.a2