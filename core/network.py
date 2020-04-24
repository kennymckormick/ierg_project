"""
This file implement neural network for you.

Nothing you need to do in this file.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class mtmlp(nn.Module):
    def __init__(self, sizes, activation, output_activation=nn.Identity):
        super(mtmlp, self).__init__()
        layers = []
        num_layers = len(sizes)
        self.num_layers = num_layers
        for j in range(self.num_layers - 2):
            layers += [nn.Linear(sizes[j], sizes[j + 1]), activation()]

        self.backbone = nn.Sequential(*layers)
        self.heada = nn.Sequential(
            *[nn.Linear(sizes[num_layers - 2], sizes[num_layers - 1]), output_activation()])
        self.headb = nn.Sequential(
            *[nn.Linear(sizes[num_layers - 2], sizes[num_layers - 1]), output_activation()])

    def forward(self, x, branch='a'):
        feat = self.backbone(x)
        assert branch in ['a', 'b']
        if branch == 'a':
            return self.heada(feat)
        else:
            return self.headb(feat)


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh,
                 output_activation=nn.Identity, act_coeff=1.0):
        super(MLPGaussianActor, self).__init__()
        self.act_coeff = act_coeff
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32) * self.act_coeff
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) +
                          [act_dim], activation, output_activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs) * self.act_coeff
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1, keepdim=True)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MTMLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh,
                 output_activation=nn.Identity, act_coeff=1.0):
        super(MTMLPGaussianActor, self).__init__()
        self.act_coeff = act_coeff
        log_std_a = -0.5 * np.ones(act_dim, dtype=np.float32) * self.act_coeff
        self.log_std_a = torch.nn.Parameter(torch.as_tensor(log_std_a))

        log_std_b = -0.5 * np.ones(act_dim, dtype=np.float32) * self.act_coeff
        self.log_std_b = torch.nn.Parameter(torch.as_tensor(log_std_b))

        self.mu_net = mtmlp([obs_dim] + list(hidden_sizes) +
                            [act_dim], activation, output_activation)

    def _distribution(self, obs, branch='a'):
        assert branch in ['a', 'b']
        mu = self.mu_net(obs, branch) * self.act_coeff
        std = torch.exp(self.log_std_a if branch == 'a' else self.log_std_b)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1, keepdim=True)

    def forward(self, obs, act=None, branch='a'):
        pi = self._distribution(obs, branch)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # not squeezing here, follow original setting
        return self.v_net(obs)


class MTMLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super(MTMLPCritic, self).__init__()
        self.v_net = mtmlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, branch='a'):
        # not squeezing here, follow original setting
        return self.v_net(obs, branch)


class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64),
                 activation=nn.Tanh, output_activation=nn.Identity,
                 act_coeff=1.0, pretrain_pth=None):
        super().__init__()
        # policy builder depends on action space
        self.pi = MLPGaussianActor(
            obs_dim, act_dim, hidden_sizes, activation, output_activation, act_coeff)

        self.v = MLPCritic(obs_dim, hidden_sizes, activation)
        if pretrain_pth is not None:
            wt = torch.load(pretrain_pth)
            if 'model' in wt:
                wt = wt['model']
            self.load_state_dict(wt)

    def step(self, obs, deterministic=False):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            if deterministic:
                a = pi.mean
            else:
                a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return v, a, logp_a


class MTMLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64),
                 activation=nn.Tanh, output_activation=nn.Identity,
                 act_coeff=1.0, pretrain_pth=None):
        super().__init__()
        # policy builder depends on action space
        self.pi = MTMLPGaussianActor(
            obs_dim, act_dim, hidden_sizes, activation, output_activation, act_coeff)

        self.v = MTMLPCritic(obs_dim, hidden_sizes, activation)
        if pretrain_pth is not None:
            wt = torch.load(pretrain_pth)
            if 'model' in wt:
                wt = wt['model']
            self.load_state_dict(wt)

    def step(self, obs, deterministic=False, branch='a'):
        with torch.no_grad():
            pi = self.pi._distribution(obs, branch)
            if deterministic:
                a = pi.mean
            else:
                a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs, branch)
        return v, a, logp_a
