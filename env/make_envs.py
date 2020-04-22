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
        resized_dim=42          # Resized the observation to a 42x42 image
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


def make_envs(env_id="CompetitivePong-v0", seed=0, log_dir="data", num_envs=5,
              asynchronous=True, resized_dim=42):
    """
    Create CUHKPong-v0, CUHKPongDouble-v0 or CartPole-v0 environments. If
    num_envs > 1, put them into different processes.
    :param env_id: The name of environment you want to create
    :param seed: The random seed
    :param log_dir: The path to store the learning stats
    :param num_envs: How many environments you want to run concurrently (Too
        large number will block your program.)
    :param asynchronous: whether to use multiprocessing
    :param resized_dim: resize the observation to image with shape (1,
        resized_dim, resized_dim)
    :return: A vectorized environment
    """
    asynchronous = asynchronous and num_envs > 1

    if env_id in ["CartPole-v0", "Walker-2d", "Humanoid-v2", "HumanoidStandup-v2"]:
        print("Setup easy environment CartPole-v0 for testing.")
        envs = [lambda: gym.make(env_id) for i in range(num_envs)]
        envs = SubprocVecEnv(envs) if asynchronous else DummyVecEnv(envs)
        return envs
