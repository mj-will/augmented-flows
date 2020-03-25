"""
Pytorch implementation of augmented flows

Author: Michael Williams
"""
import numpy as np
import torch
import torch.nn as nn

def swish(x):
    """Swish activation function"""
    return torch.mul(x, torch.sigmoid(x))


class Transform(nn.Module):
    """
    Basic neural net to perform encoding or decoding
    """
    def __init__(self, in_dim, out_dim, n_neurons):
        super(Transform, self).__init__()

        self.linear1 = nn.Linear(in_dim, n_neurons)
        self.linear2 = nn.Linear(n_neurons, n_neurons)
        self.linear3 = nn.Linear(n_neurons, 2 * out_dim)

        with torch.no_grad():
            self.linear3.weight.data.fill_(0.)
            self.linear3.bias.data.fill_(0.)

    def forward(self, x):
        """
        Forward pass
        """
        x = self.linear1(x)
        x = swish(x)
        x = self.linear2(x)
        x = swish(x)
        x = self.linear3(x)
        x_s, x_m = torch.split(x, x.shape[-1] // 2, dim=1)
        x_s = torch.nn.functional.logsigmoid(x_s)
        x_s = torch.clamp(x_s, min=-2.5, max=2.5)
        return x_s, x_m


class AugmentedBlock(nn.Module):
    """
    An implementation on augmented flow blocks

    See: https://arxiv.org/abs/2002.07101
    """
    def __init__(self, x_dim, e_dim, n_layers=2, n_neurons=64):
        """
        Intialise the block
        """
        super(AugmentedBlock, self).__init__()
        self.encoder = Transform(x_dim, e_dim, n_neurons)
        self.decoder = Transform(e_dim, x_dim, n_neurons)

        def init(m):
          if isinstance(m, nn.Linear):
            m.bias.data.fill_(0)
            nn.init.orthogonal_(m.weight.data)

    def forward(self, feature, augment, mode='forward'):
        """
        Forward or backwards pass
        """
        log_J = 0.
        if mode == 'forward':
            # encode e -> z | x
            log_s, m = self.encoder(feature)
            s = torch.exp(log_s)
            z = s * augment + m
            log_J += log_s.sum(-1, keepdim=True)
            # decode x -> y | z
            log_s, m = self.decoder(z)
            s = torch.exp(log_s)
            y = s * feature + m
            log_J += log_s.sum(-1, keepdim=True)
            return y, z, log_J
        else:
            # decode y -> x | z
            log_s, m = self.decoder(augment)
            s = torch.exp(-log_s)
            x = s * (feature - m)
            log_J -= log_s.sum(-1, keepdim=True)
            # encode z -> e | z
            log_s, m = self.encoder(x)
            s = torch.exp(-log_s)
            e = s * (augment - m)
            log_J -= log_s.sum(-1, keepdim=True)
            return x, e, log_J


class AugmentedSequential(nn.Sequential):
    """
    A sequential container for augmented flows
    """
    def forward(self, feature, augment, mode='forward'):
        """
        Forward or backward pass through the flows
        """
        log_dets = torch.zeros(feature.size(0), 1, device=feature.device)
        if mode == 'forward':
            for module in self._modules.values():
                feature, augment, log_J = module(feature, augment, mode)
                log_dets += log_J
        else:
            for module in reversed(self._modules.values()):
                feature, augment, log_J = module(feature, augment, mode)
                log_dets += log_J

        return feature, augment, log_dets

    def log_N(self, x):
        """
        Calculate of the log probability of an N-dimensional gaussian
        """
        return (-0.5 * x.pow(2) - 0.5 * np.log(2 * np.pi)).sum(
            -1, keepdim=True)

    def log_p_xe(self, feature, augment):
        """
        Calculate the joint log probability p(x, e)
        """
        # get transformed features
        y, z, log_J = self(feature, augment)
        # y & z should be gaussian
        y_prob = self.log_N(y)
        z_prob = self.log_N(z)
        return (y_prob + z_prob + log_J).sum(-1, keepdim=True)

    def log_p_x(self, feature, e_dim, K=1000):
        """
        Calculate the lower bound of the marginalised log probability p(x)
        """
        log_p_x = torch.zeros(feature.size(0), 1, device=feature.device)
        # get log p(x, e)
        for i,f in enumerate(feature):
            e = torch.Tensor(K, e_dim).normal_().to(feature.device)
            log_q = self.log_N(e)
            # need to pass the same feature K times (for each e)
            f_repeated = f * torch.ones(K, f.size(0))
            log_p_xe = self.log_p_xe(f_repeated, e)
            # compute sum of ratio
            lpx = -np.log(K) + torch.logsumexp(log_p_xe - log_q, (0))
            log_p_x[i] = lpx
        return log_p_x
