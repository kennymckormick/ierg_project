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


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_coeff):
        super().__init__()
        self.act_coeff = act_coeff
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32) * self.act_coeff
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) +
                          [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs) * self.act_coeff
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # not squeezing here, follow original setting
        return self.v_net(obs)


class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64, 64), activation=nn.Tanh, act_coeff=1.0):
        super().__init__()
        # policy builder depends on action space
        self.pi = MLPGaussianActor(
            obs_dim, act_dim, hidden_sizes, activation, act_coeff)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, eval=True):
        if eval:
            with torch.no_grad():
                pi = self.pi._distribution(obs)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                v = self.v(obs)
        else:
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return v, a, logp_a

    def act(self, obs):
        return self.step(obs)[0]
