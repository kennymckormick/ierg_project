"""
This file implements the train scripts for both A2C and PPO

You need to implement all TODOs in this script.

Note that you may find this file is completely compatible for both A2C and PPO.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""

# Modification: Now it should deal with continuous control problems
import argparse
from collections import deque

import gym
import numpy as np
import torch
import json
from env import make_envs

from core.ppo_trainer import PPOTrainer, ppo_config
from core.utils import verify_log_dir, pretty_print, Timer, evaluate, \
    summary, save_progress, FrameStackTensor, step_envs, reduce_shape, enlarge_shape
from env.make_envs import Walker2d_wrapper

gym.logger.set_level(40)
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--algo",
    default="PPO",
    type=str,
    help="(Required) The algorithm you want to run. Must in [PPO]."
)
parser.add_argument(
    "--log-dir",
    default="default",
    type=str,
    help="The path of directory that you want to store the data to. "
)
parser.add_argument(
    "--num-envs",
    default=15,
    type=int,
    help="The number of parallel environments. Default: 15"
)
parser.add_argument(
    "--learning-rate", "-LR",
    default=5e-4,
    type=float,
    help="The learning rate. Default: 5e-4"
)
parser.add_argument(
    "--seed",
    default=100,
    type=int,
    help="The random seed. Default: 100"
)
parser.add_argument(
    "--max-steps",
    "-N",
    default=1e7,
    type=float,
    help="The random seed. Default: 1e7"
)
parser.add_argument(
    "--env-id",
    default="Walker2d-v3",
    type=str,
    help="The environment id"
)
parser.add_argument(
    "--envopt",
    default=None,
    type=str,
)
parser.add_argument(
    "--trainopt",
    default=None,
    type=str,
)
args = parser.parse_args()

env_options = {}
trainer_options = {}


def train(args):
    # Verify algorithm and config
    global env_options, trainer_options
    algo = args.algo
    if algo == "PPO":
        config = ppo_config
    else:
        raise ValueError("args.algo must in [PPO]")
    config.num_envs = args.num_envs
    if args.envopt is not None:
        f = open(args.envopt)
        env_options = json.load(f)
    if args.trainopt is not None:
        f = open(args.trainopt)
        trainer_options = json.load(f)

    for k, v in trainer_options.items():
        setattr(config, k, v)

    # Seed the environments and setup torch
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)

    # Clean log directory
    log_dir = verify_log_dir('work_dirs', args.log_dir)

    # Create vectorized environments
    num_envs = args.num_envs
    env_id = args.env_id
    envs = make_envs(
        env_id=env_id,
        seed=seed,
        log_dir=log_dir,
        num_envs=num_envs,
        asynchronous=True,
        options=env_options,
    )

    if env_id == "Walker2d-v3":
        healthy_z_range = (0.8, 2.0)
    elif env_id == 'Humanoid-v3':
        healthy_z_range = (1.0, 2.0)
    if 'healthy_z_range' in env_options:
        healthy_z_range = env_options['healthy_z_range']
    eval_env = gym.make(
        env_id, healthy_z_range=healthy_z_range, healthy_reward=0)
    if env_id == "Walker2d-v3":
        eval_env = Walker2d_wrapper(env, env_options)

    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]
    real_obs_dim = obs_dim
    real_act_dim = act_dim
    if 'real_obs_dim' in trainer_options:
        real_obs_dim = trainer_options['real_obs_dim']
    if 'real_act_dim' in trainer_options:
        real_act_dim = trainer_options['real_act_dim']
    dim_dict = dict(obs_dim=obs_dim, act_dim=act_dim,
                    real_obs_dim=real_obs_dim, real_act_dim=real_act_dim)

    # Setup trainer
    if algo == "PPO":
        trainer = PPOTrainer(envs, config)
    else:
        raise NotImplementedError

    # Create a placeholder tensor to help stack frames in 2nd dimension
    # That is turn the observation from shape [num_envs, 1, 84, 84] to
    # [num_envs, 4, 84, 84].
    frame_stack_tensor = FrameStackTensor(
        num_envs, envs.observation_space.shape, config.device)

    # Setup some stats helpers
    episode_rewards = np.zeros([num_envs, 1], dtype=np.float)
    total_episodes = total_steps = iteration = 0
    reward_recorder = deque(maxlen=100)
    episode_length_recorder = deque(maxlen=100)
    sample_timer = Timer()
    process_timer = Timer()
    update_timer = Timer()
    total_timer = Timer()
    progress = []
    evaluate_stat = {}

    # Start training
    print("Start training!")
    obs = envs.reset()
    frame_stack_tensor.update(obs)
    trainer.rollouts.observations[0].copy_(
        reduce_shape(frame_stack_tensor.get(), real_obs_dim))
    while True:  # Break when total_steps exceeds maximum value
        with sample_timer:
            for index in range(config.num_steps):

                trainer.model.eval()
                values, actions, action_log_prob = trainer.model.step(
                    reduce_shape(frame_stack_tensor.get(), real_obs_dim), eval=True)

                cpu_actions = actions.cpu().numpy()
                cpu_actions = enlarge_shape(cpu_actions, act_dim)

                obs, reward, done, info, masks, total_episodes, \
                    total_steps, episode_rewards = step_envs(
                        cpu_actions, envs, episode_rewards, frame_stack_tensor,
                        reward_recorder, episode_length_recorder, total_steps,
                        total_episodes, config.device)

                rewards = torch.from_numpy(
                    reward.astype(np.float32)).view(-1, 1).to(config.device)

                # Store samples
                trainer.rollouts.insert(
                    reduce_shape(frame_stack_tensor.get(),
                                 real_obs_dim), actions,
                    action_log_prob, values, rewards, masks)

        # ===== Process Samples =====
        with process_timer:
            with torch.no_grad():
                next_value = trainer.compute_values(
                    trainer.rollouts.observations[-1])
            trainer.rollouts.compute_returns(next_value, config.GAMMA)

        trainer.model.train()
        # ===== Update Policy =====
        with update_timer:
            policy_loss, value_loss, total_loss = trainer.update(
                trainer.rollouts)
            trainer.rollouts.after_update()

        # ===== Evaluate Current Policy =====
        if iteration % config.eval_freq == 0:
            eval_timer = Timer()
            rewards, eplens = evaluate(
                trainer, eval_env, 1, dim_dict=dim_dict)
            evaluate_stat = summary(rewards, "episode_reward")
            evaluate_stat.update(summary(eplens, "episode_length"))
            evaluate_stat.update(dict(
                evaluate_time=eval_timer.now,
                evaluate_iteration=iteration
            ))

        # ===== Log information =====
        if iteration % config.log_freq == 0:
            stats = dict(
                log_dir=log_dir,
                frame_per_second=int(total_steps / total_timer.now),
                training_episode_reward=summary(reward_recorder,
                                                "episode_reward"),
                training_episode_length=summary(episode_length_recorder,
                                                "episode_length"),
                evaluate_stats=evaluate_stat,
                learning_stats=dict(
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    total_loss=total_loss
                ),
                total_steps=total_steps,
                total_episodes=total_episodes,
                time_stats=dict(
                    sample_time=sample_timer.avg,
                    process_time=process_timer.avg,
                    update_time=update_timer.avg,
                    total_time=total_timer.now,
                    episode_time=sample_timer.avg + process_timer.avg +
                    update_timer.avg
                ),
                iteration=iteration
            )

            progress.append(stats)
            pretty_print({
                "===== {} Training Iteration {} =====".format(
                    algo, iteration): stats
            })

        if iteration % config.save_freq == 0:
            trainer_path = trainer.save_w(log_dir, "iter{}".format(iteration))
            progress_path = save_progress(log_dir, progress)
            print("Saved trainer state at <{}>. Saved progress at <{}>.".format(
                trainer_path, progress_path
            ))

        # [TODO] Stop training when total_steps is greater than args.max_steps
        if total_steps > args.max_steps:
            break
        pass

        iteration += 1

    trainer.save_w(log_dir, "final")
    envs.close()


if __name__ == '__main__':
    train(args)
