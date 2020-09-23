
import torch.nn as nn
import torch

class Encoder(nn.Module):
 
    def __init__(self,x_dim,z_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(x_dim),512),
            nn.ReLU()
        )
        
        self.f_mu = nn.Linear(512, z_dim)
        self.f_var = nn.Linear(512, z_dim)


    def encode(self, img):
        h1 = self.model(img)
        return self.f_mu(h1),self.f_var(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Decoder_samp(nn.Module):

    def __init__(self,x_dim,z_dim):
        super(Decoder_samp, self).__init__()
       
        self.model = nn.Sequential(
            nn.Linear(z_dim,512),
        )
        
        self.f_mu = nn.Linear(512, x_dim)
        self.f_std = nn.Linear(512, x_dim)
        self.x_dim = x_dim
    def decode(self,z):
        h1 = self.model(z)
        return self.f_mu(h1),self.f_std(h1)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, z):
        mu,logvar = self.decode(z)
        output = self.reparameterize(mu,logvar)
        return output,mu,logvar   


class Decoder(nn.Module):

    def __init__(self,x_dim,z_dim):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim,512),
            nn.ReLU(),
            nn.Linear(512,x_dim)
        )

    def forward(self, z):
        img= self.model(z)
        return img  
