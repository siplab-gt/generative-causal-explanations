import torch.nn as nn
import torch.nn.functional as F
import torch

class Decoder(nn.Module):

    """
        Encoder initializer
        :param x_dim: dimension of the input
        :param z_dim: dimension of the latent representation
    """
    def __init__(self, x_dim, z_dim):
        super(Decoder, self).__init__()
        self.model_enc = nn.Linear(int(z_dim),int(x_dim))

    def reparameterize(self, mu,gamma):
        eps = torch.randn_like(mu)
        return mu + eps*gamma
    
    # run linear forward pass
    def forward(self, x, What, gamma):
        z = F.linear(x, What, None)
        z = self.reparameterize(z, gamma)
        return z