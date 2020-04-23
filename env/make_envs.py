"""
This file defines a helper function to build gym environment.
Usages:
    make_envs(
        env_id="CUHKRLPong-v0", # Environment name, must in [CUHKRLPong-v0,
                                # CUHKRLPongDouble-v0, CartPole-v0].
        seed=0,                 # Random seed
        log_dir="data",         # Which directory to store data and checkpoints
        num_envs=5,             # How many concurrent environments to run
        asynchronous=True,      # Whether to use asynchronous envrionments.
                                # This can extremely accelerate the system
    )
Notes:
1. If you wish to use asynchronous environments, you should run it in python
scripts under "if __name__ == '__main__'" line.
2. CartPole-v0 environment can be used for testing algorithms.
-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import os
import shutil

import gym

from .utils import DummyVecEnv, SubprocVecEnv


__all__ = ["make_envs"]

msg = """
Multiprocessing vectorized environments need to be created under
"if __name__ == '__main__'" line due to the limitation of multiprocessing
module.
If you are testing codes within interactive interface like jupyter
notebook, please set the num_envs to 1, i.e. make_envs(num_envs=1) to avoid
such error. We return envs = None now.
"""


class Walker2d_wrapper(gym.Wrapper):
    # options should be a list
    def __init__(self, env, options={}):
        super(Walker2d_wrapper, self).__init__(env)
        self.options = options

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'reward_threshold' in self.options:
            reward = (reward >= self.options['reward_threshold'])
        return observation, reward, done, info


def make_envs(env_id="Walker2d-v3", seed=0, log_dir="data", num_envs=5,
              asynchronous=True, options={}):
    asynchronous = asynchronous and num_envs > 1

    if env_id == "Walker2d-v3":
        healthy_z_range = (0.8, 2.0)
        if 'healthy_z_range' in options:
            healthy_z_range = options.pop('healthy_z_range')
        envs = [lambda: Walker2d_wrapper(gym.make(
            env_id, healthy_z_range=healthy_z_range, healthy_reward=0), options) for i in range(num_envs)]
        envs = SubprocVecEnv(envs) if asynchronous else DummyVecEnv(envs)
        return envs

    if env_id == 'Humanoid-v3':
        healthy_z_range = (1.0, 2.0)
        if 'healthy_z_range' in options:
            healthy_z_range = options.pop('healthy_z_range')
        healthy_reward = 0
        if 'healthy_reward' in options:
            healthy_reward = options.pop('healthy_reward')
        envs = [lambda: gym.make(
            env_id, healthy_z_range=healthy_z_range, healthy_reward=healthy_reward) for i in range(num_envs)]
        envs = SubprocVecEnv(envs) if asynchronous else DummyVecEnv(envs)
        return envs

    raise NotImplementedError
