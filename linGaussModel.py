from __future__ import division

import torch.nn as nn
import torch.nn.functional as F
import torch

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
    
    def forward(self, x, What, gamma):
         # Run linear forward pass
        z = F.linear(x, What, None)
        z = self.reparameterize(z, gamma)
        return z