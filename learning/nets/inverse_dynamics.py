'''

2 FC layers

Operates on grayscale, segmented images of environment
'''
import torch
from torch import nn

import torch.nn.functional as F
from torchvision.models import resnet18,alexnet

class InverseDynamics(nn.Module):
    def __init__(self,p_out_dim, th_out_dim, len_out_dim,latent_dim=200):
        super(InverseDynamics,self).__init__()
        self.latent_dim = latent_dim 
        self.alexnet = alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,self.latent_dim)
        )

        self.p_out_dim = p_out_dim
        self.th_out_dim = th_out_dim
        self.len_out_dim = len_out_dim
        self.p_fc = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ELU(True),
            nn.Linear(latent_dim, p_out_dim),
        )
        self.th_fc = nn.Sequential(
            nn.Linear(latent_dim * 2 + p_out_dim, latent_dim),
            nn.ELU(True),
            nn.Linear(latent_dim, th_out_dim),
        )
        self.len_fc = nn.Sequential(
            nn.Linear(latent_dim * 2 + p_out_dim + th_out_dim, latent_dim),
            nn.ELU(True),
            nn.Linear(latent_dim, len_out_dim),
        )
        

    def forward(self, o, o_next):
        o = self.alexnet(o)
        o_next = self.alexnet(o_next)
        o.view(-1)
        o_next.view(-1)
        x = torch.cat((o,o_next),dim=-1)
        # p
        p = torch.argmax(self.p_fc(x),dim=-1)
        p = F.one_hot(torch.argmax(self.p_fc(x),dim=-1),self.p_out_dim)

        # theta
        th = torch.cat((p,x),dim=-1)
        th = F.one_hot(torch.argmax(self.th_fc(th),dim=-1),self.th_out_dim)

        # length
        len = torch.cat((p,th,x),dim=-1)
        len = F.one_hot(torch.argmax(self.len_fc(len),dim=-1),self.len_out_dim)

        return p,th,len
