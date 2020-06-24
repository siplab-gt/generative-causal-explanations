
import torch.nn as nn
import torch

class Encoder(nn.Module):
 
    def __init__(self, z_dim, c_dim,img_size):
        """
        Encoder initializer
        :param x_dim: dimension of the input
        :param z_dim: dimension of the latent representation
        :param M: number of transport operators
        """
        super(Encoder, self).__init__()
        
        self.model_enc = nn.Sequential(
            nn.Conv2d(int(c_dim), 64, 4, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ZeroPad2d((1,2,1,2)),
            nn.Conv2d(64, 64, 4, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(int(64*img_size*img_size/16), z_dim)
        self.fc_var = nn.Linear(int(64*img_size*img_size/16), z_dim)
        

    def encode(self, img):
        out = self.model_enc(img)
        out = out.view(out.size(0),-1)
      
        return self.fc_mu(out),self.fc_var(out)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
class Decoder(nn.Module):

    def __init__(self,z_dim,c_dim,img_size):
        super(Decoder, self).__init__()
        self.img_4 = img_size/4
        self.fc = nn.Sequential(
                nn.Linear(z_dim,int(self.img_4*self.img_4*64)),
                nn.ReLU(),
                )
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d( 64, 64, 4, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d( 64, 64, 4, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d( 64, int(c_dim), 4, stride=2, padding=1),
            nn.BatchNorm2d(int(c_dim)),
            nn.Sigmoid()
        )
        

    def forward(self, z):

        batch_size = z.shape[0]
        temp_var = self.fc(z)
        temp_var = temp_var.view(batch_size,64,int(self.img_4),int(self.img_4))
        img= self.model(temp_var)
        return img 
