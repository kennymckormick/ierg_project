"""
This file implement a base trainer class for both A2C and PPO trainers.

You should finish `evaluate_actions` and `compute_action`

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import os
import numpy as np
import torch
from torch import nn

from .network import MTMTMLPActorCritic


# updated to be consistent with new network
# dim_dict should look like: {'obs_a': num, 'act_a': num ...}
class BaseTrainerMTMT:
    def __init__(self, config, dim_dict):

        self.device = config.device
        self.config = config
        self.lr = config.LR
        self.num_envs = config.num_envs
        self.value_loss_weight = config.value_loss_weight
        self.num_steps = config.num_steps
        self.grad_norm_max = config.grad_norm_max
        self.eps = 1e-6

        self.obs_a = dim_dict['obs_a']
        self.obs_b = dim_dict['obs_b']
        self.act_a = dim_dict['act_a']
        self.act_a = dim_dict['act_b']
        self.coeff_a = dim_dict['coeff_a']
        self.coeff_b = dim_dict['coeff_b']
        self.obs_dim = [self.obs_a, self.obs_b]
        self.act_dim = [self.act_a, self.act_b]
        self.coeff = [self.coeff_a, self.coeff_b]

        self.model = MTMTMLPActorCritic(
            self.obs_dim, self.act_dim, hidden_sizes=config.hidden_sizes,
            activation=config.activation, output_activation=config.output_activation,
            act_coeff=self.coeff, pretrain_pth=config.pretrain_pth)

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

    # should given branch
    def evaluate_actions(self, obs, act, branch='a'):
        """Run models to get the values, log probability of the action in
        current state"""
        values = self.model.v(obs, branch)
        pi = self.model.pi._distribution(obs, branch)
        action_log_probs = self.model.pi._log_prob_from_distribution(pi, act)

        return values.view(-1, 1), action_log_probs.view(-1, 1)

    # should given branch
    def compute_values(self, obs, branch='a'):
        """Compute the values corresponding to current policy at current
        state"""
        values = self.model.v(obs, branch)
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
