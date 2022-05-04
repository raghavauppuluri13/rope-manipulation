from sre_parse import State
from torch import nn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class G(nn.Module):
    def __init__(self, s_dim, z_dim, channel_dim=3):
        super(G, self).__init__()
        self.input_dim = 2 * s_dim + z_dim
        self.channel_dim = channel_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, 512, 4, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 2 * channel_dim, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, s, s_next, z):
        x = torch.cat((s, s_next, z), dim=1).view(-1, self.input_dim, 1, 1)
        x = self.main(x)
        return x[:, : self.channel_dim, :, :], x[:, self.channel_dim :, :, :]


class D(nn.Module):
    def __init__(self, channel_dim=3):
        super(D, self).__init__()
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
        x = torch.cat((o, o_next), dim=1)
        return self.main(x)


class Uniform(nn.Module):
    def __init__(self, dim):
        super(Uniform, self).__init__()
        self.dim = dim
        self.dist = torch.distributions.Uniform(-1.0, 1.0)

    def forward(self):
        return self.dist.sample(self.dim)

    def log_prob(self, s):
        return self.dist.log_prob(s).sum(1)


class Normal(nn.Module):
    def __init__(self, dim):
        super(Normal, self).__init__()
        self.dim = dim
        self.dist = torch.distributions.Normal(0.0, 1.0)

    def forward(self):
        return self.dist.sample(self.dim)


class T(nn.Module):
    """
    Gaussian transition function
    N(0,T_theta)
    """

    def __init__(self, s_dim, default_var=0.1):
        # input: vector of state_dim
        super(T, self).__init__()

        self.state_dim = s_dim

        self.main = nn.Sequential(
            nn.Linear(s_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, s_dim),
        )
        self.var = default_var**2

    def forward(self, x):
        dist = self.ret_dist(x)
        return dist.sample()

    def ret_var(self, x):
        return self.main(x).exp()

    def ret_dist(self, s):
        mu = s
        var = self.ret_var(s)
        dist = MultivariateNormal(mu, covariance_matrix=torch.diag_embed(var))
        return dist

    def log_post_prob(self, s, s_next):
        dist = self.ret_dist(s)
        return dist.log_prob(s_next)


class Q(nn.Module):
    """
    Gaussian Posterior Estimator on observations
    """

    def __init__(self, s_dim, channel_dim=3):
        super(Q, self).__init__()
        self.mu_conv = nn.Conv2d(128, s_dim, 1)
        self.std_conv = nn.Conv2d(128, s_dim, 1)
        self.main = nn.Sequential(
            nn.Conv2d(channel_dim, 64, 4, 2, 1),
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
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(512, 128, 4, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.main(x)
        x = self.last_layer(x)
        mu = self.mu_conv(x).squeeze()
        var = self.std_conv(x).squeeze().exp()
        return mu, var

    def log_post_prob(self, x, s):
        mu, var = self.forward(x)
        dist = torch.distributions.Normal(mu, var)
        log_prob = dist.log_prob(s)
        return log_prob.sum(1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class EvalClassifier(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
