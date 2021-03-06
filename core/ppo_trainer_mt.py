"""
This file implement PPO algorithm.

You need to implement `update` and `compute_loss` function.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import numpy as np
import torch
import torch.optim as optim
from torch import nn

from .base_trainer_mt import BaseTrainerMT
from .buffer import PPORolloutStorage


class PPOConfig:
    """Not like previous assignment where we use a dict as config, here we
    build a class to represent config."""

    def __init__(self):
        # Common
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.eval_freq = 10
        self.save_freq = 200
        self.log_freq = 10
        self.num_envs = 1

        # Sample
        self.num_steps = 200  # num_steps * num_envs = sample_batch_size

        # Learning
        self.GAMMA = 0.99
        self.LR = 5e-4
        self.grad_norm_max = 10.0
        self.ppo_epoch = 10
        self.mini_batch_size = 500
        self.ppo_clip_param = 0.1
        self.USE_GAE = True
        self.gae_lambda = 0.95
        self.value_loss_weight = 1.0
        self.hidden_sizes = (64, 64)
        self.activation = nn.Tanh
        self.output_activation = nn.Identity
        self.pretrain_pth = None


ppo_config = PPOConfig()


class PPOTrainerMT(BaseTrainerMT):
    def __init__(self, enva, envb, config, myconfig={}):
        super(PPOTrainerMT, self).__init__(enva, envb, config, myconfig)

        self.num_sgd_steps = config.ppo_epoch
        self.mini_batch_size = config.mini_batch_size
        self.clip_param = config.ppo_clip_param

    def setup_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                                    eps=1e-5)

    def setup_rollouts(self):
        self.rollouts = [PPORolloutStorage(self.num_steps, self.num_envs,
                                           self.config.real_obs_dim, self.config.real_act_dim,
                                           self.device, self.config.USE_GAE,
                                           self.config.gae_lambda),
                         PPORolloutStorage(self.num_steps, self.num_envs,
                                           self.config.real_obs_dim, self.config.real_act_dim,
                                           self.device, self.config.USE_GAE,
                                           self.config.gae_lambda)]

    def compute_loss(self, sample, branch='a'):
        """Compute the loss of PPO"""
        observations_batch, actions_batch, return_batch, masks_batch, \
            old_action_log_probs_batch, adv_targ = sample

        assert old_action_log_probs_batch.shape == (self.mini_batch_size, 1)
        assert adv_targ.shape == (self.mini_batch_size, 1)
        assert return_batch.shape == (self.mini_batch_size, 1)

        values, action_log_probs = self.evaluate_actions(
            observations_batch, actions_batch, branch)

        assert values.shape == (self.mini_batch_size, 1)
        assert action_log_probs.shape == (self.mini_batch_size, 1)
        assert values.requires_grad
        assert action_log_probs.requires_grad

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        clip_adv = torch.clamp(ratio, 1 - self.clip_param,
                               1 + self.clip_param) * adv_targ
        policy_loss = torch.min(ratio * adv_targ, clip_adv)
        policy_loss = - policy_loss.mean()

        value_loss = torch.nn.MSELoss()(values, return_batch)

        loss = policy_loss + self.value_loss_weight * value_loss

        return loss, policy_loss, value_loss

    def update(self, rollout_a, rollout_b):
        # Get the normalized advantages
        advantage_a = rollout_a.returns[:-1] - rollout_a.value_preds[:-1]
        advantage_a = (advantage_a - advantage_a.mean()) / \
            (advantage_a.std() + 1e-5)

        advantage_b = rollout_b.returns[:-1] - rollout_b.value_preds[:-1]
        advantage_b = (advantage_b - advantage_b.mean()) / \
            (advantage_b.std() + 1e-5)

        value_loss_epoch_a = []
        policy_loss_epoch_a = []
        total_loss_epoch_a = []

        value_loss_epoch_b = []
        policy_loss_epoch_b = []
        total_loss_epoch_b = []

        for e in range(self.num_sgd_steps):
            data_generator_a = rollout_a.feed_forward_generator(
                advantage_a, self.mini_batch_size)
            data_generator_b = rollout_b.feed_forward_generator(
                advantage_b, self.mini_batch_size)

            for sample_a, sample_b in zip(data_generator_a, data_generator_b):
                total_loss, policy_loss, value_loss = self.compute_loss(
                    sample_a, 'a')

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_norm_max)
                self.optimizer.step()

                value_loss_epoch_a.append(value_loss.item())
                policy_loss_epoch_a.append(policy_loss.item())
                total_loss_epoch_a.append(total_loss.item())

                total_loss, policy_loss, value_loss = self.compute_loss(
                    sample_b, 'b')

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_norm_max)
                self.optimizer.step()

                value_loss_epoch_b.append(value_loss.item())
                policy_loss_epoch_b.append(policy_loss.item())
                total_loss_epoch_b.append(total_loss.item())

        return [[np.mean(policy_loss_epoch_a), np.mean(value_loss_epoch_a), np.mean(total_loss_epoch_a)],
                [np.mean(policy_loss_epoch_b), np.mean(value_loss_epoch_b), np.mean(total_loss_epoch_b)]]
