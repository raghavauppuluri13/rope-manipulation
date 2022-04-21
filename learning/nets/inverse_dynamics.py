'''

2 FC layers

Operates on grayscale, segmented images of environment
'''
from torch import nn

class FCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCN,self).__init__()
    
    def forward(self)

class InverseDynamics(nn.Module):
    def __init__(self,s_dim, z_dim,channel_dim=3):
        super(InverseDynamics,self).__init__()
        self.input_dim = 2 * s_dim + z_dim
        self.channel_dim = channel_dim 
        self.main = nn.Sequential(
            nn.Conv2d(self.input_dim, 96, 4, 1, bias=False),
            nn.ELU(True),
            nn.Linear(96,)
        )

    def forward(self, s, s_next, z):
        x = torch.cat((s,s_next,z),dim=1)
        return self.main(x)