"""
This file implement a base trainer class for both A2C and PPO trainers.
-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import os
import numpy as np
import torch
from torch import nn

from .network import MLPActorCritic


# updated to be consistent with new network
class BaseTrainer:
    def __init__(self, env, config, myconfig={}):

        act_funcs = {'Sigmoid': nn.Sigmoid, 'ReLU': nn.ReLU,
                     'Tanh': nn.Tanh, 'Identity': nn.Identity}
        if 'activation' in myconfig:
            config.activation = act_funcs[myconfig['activation']]
            myconfig.pop('activation')
        if 'output_activation' in myconfig:
            config.output_activation = act_funcs[myconfig['output_activation']]
            myconfig.pop('output_activation')
        # can be real_obs_dim, real_act_dim, pretrain_pth, etc.
        for k, v in myconfig.items():
            setattr(config, k, v)

        self.device = config.device
        self.config = config
        self.lr = config.LR
        self.num_envs = config.num_envs
        self.value_loss_weight = config.value_loss_weight
        self.num_steps = config.num_steps
        self.grad_norm_max = config.grad_norm_max
        self.eps = 1e-6

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_high = env.action_space.high
        self.act_low = env.action_space.low
        # should be in config
        if not hasattr(config, 'real_obs_dim'):
            config.real_obs_dim = self.obs_dim
        if not hasattr(config, 'real_act_dim'):
            config.real_act_dim = self.act_dim

        assert sum(self.act_high == self.act_high[0]) == self.act_dim
        assert sum(self.act_low == self.act_low[0]) == self.act_dim
        assert self.act_high[0] == -self.act_low[0]
        config.act_coeff = self.act_high[0]

        self.model = MLPActorCritic(
            config.real_obs_dim, config.real_act_dim, hidden_sizes=config.hidden_sizes,
            activation=config.activation, output_activation=config.output_activation,
            act_coeff=config.act_coeff, pretrain_pth=config.pretrain_pth, std_init_offset=config.std_init_offset)

        self.model = self.model.to(self.device)
        self.model.train()

        self.setup_optimizer()
        self.setup_rollouts()

    def setup_optimizer(self):
        raise NotImplementedError()

    def setup_rollouts(self):
        raise NotImplementedError()

    def compute_loss(self, rollouts):
        raise NotImplementedError()

    def update(self, rollout):
        raise NotImplementedError()

    def evaluate_actions(self, obs, act):
        """Run models to get the values, log probability of the action in
        current state"""
        values = self.model.v(obs)
        pi = self.model.pi._distribution(obs)
        action_log_probs = self.model.pi._log_prob_from_distribution(pi, act)

        return values.view(-1, 1), action_log_probs.view(-1, 1)

    def compute_values(self, obs):
        """Compute the values corresponding to current policy at current
        state"""
        values = self.model.v(obs)
        return values

    def save_w(self, log_dir="", suffix=""):
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, "checkpoint-{}.pkl".format(suffix))
        torch.save(dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict()
        ), save_path)
        return save_path

    def load_w(self, log_dir="", suffix=""):
        save_path = os.path.join(log_dir, "checkpoint-{}.pkl".format(suffix))
        if os.path.isfile(save_path):
            state_dict = torch.load(
                save_path,
                torch.device('cpu') if not torch.cuda.is_available() else None
            )
            self.model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
