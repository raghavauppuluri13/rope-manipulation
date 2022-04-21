from torch import nn
import torch

class FCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim ):
        super(FCN,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim,out_dim),
            nn.ReLU(True)
        )

    def forward(self,x):
        return self.main(x)


class G(nn.Module):
    def __init__(self,s_dim, z_dim,channel_dim=3):
        super(G,self).__init__()
        self.input_dim = 2 * s_dim + z_dim
        self.channel_dim = channel_dim 
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, 512, 4, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 2 * channel_dim, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, s, s_next, z):
        x = torch.cat((s,s_next,z),dim=1)
        return self.main(x)

class D(nn.Module):
    def __init__(self,channel_dim):
        super(D,self).__init__()
        self.main = nn.Sequential(
            # input size (2 or 6) x 64 x64
            nn.Conv2d(2 * channel_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4),
            nn.Sigmoid()
        )

    def forward(self, o, o_next):
        x = torch.cat((o, o_next),dim=1)
        return self.main(x)
        
class Prior(nn.Module):
    def __init__(self):
        super(Prior,self).__init__()
        self.dist = torch.distributions.Uniform(-1,1)

    def forward(self):
        return self.dist.sample()


class T(nn.Module):
    """
    Gaussian transition function
    N(0,T_theta)
    """

    def __init__(self, ):
        super(T, self).__init__()
        

    def forward(self, x):
        x = self.main(x)

class Q(nn.Module):
    """
    Gaussian Posterior Estimator on observations 
    """
    def __init__(self, channel_dim,):
        super(Q,self).__init__()
        self.mu_conv = nn.Conv2d(128,channel_dim,1)
        self.std_conv = nn.Conv2d(128,channel_dim,1)
        self.main = nn.Sequential(
            # input size (2 or 6) x 64 x64
            nn.Conv2d(2 * channel_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4),
            nn.Sigmoid()
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(512,128,4,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self,x):
        x = self.main(x)
        x = self.last_layer(x)
        mu = self.mu_conv(x)
        std = self.std_conv(x)
        return mu,std
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)