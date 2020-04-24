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

from core.ppo_trainer_mtmt import PPOTrainerMTMT, ppo_config
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
    default="Humanoid-v3",
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

parser.add_argument(
    "--opt",
    default=None,
    type=str,
)
args = parser.parse_args()

env_options = {}
trainer_options = {}


# Only env option exists in the case
def train(args):
    # Verify algorithm and config
    global env_options, trainer_options
    algo = args.algo
    if algo == "PPO":
        config = ppo_config
    else:
        raise ValueError("args.algo must in [PPO]")
    config.num_envs = args.num_envs
    if args.trainopt is not None:
        f = open(args.trainopt)
        trainer_options = json.load(f)
    if args.opt is not None:
        opt = json.load(open(args.opt))
        env_options = opt['env']
        trainer_options = opt['trainer']

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

    main_envs = make_envs(
        env_id=env_id,
        seed=seed,
        log_dir=log_dir,
        num_envs=num_envs,
        asynchronous=True,
    )

    aux_envs = make_envs(
        env_id=env_id,
        seed=seed,
        log_dir=log_dir,
        num_envs=num_envs,
        asynchronous=True,
    )

    envs = [main_envs, aux_envs]

    # eval_env is main_env
    healthy_z_range = (1.0, 2.0)
    if 'healthy_z_range' in env_options:
        healthy_z_range = env_options['healthy_z_range']
    eval_env = gym.make(
        env_id, healthy_z_range=healthy_z_range, healthy_reward=0)

    obs_dim = main_envs.observation_space.shape[0]
    act_dim = main_envs.action_space.shape[0]
    reduce_obs_dim = 46
    reduce_act_dim = 11
    obs_dims = [obs_dim, reduce_obs_dim]
    act_dims = [act_dim, reduce_act_dim]

    dim_dict = dict(obs_a=obs_dim, act_a=act_dim, obs_b=46,
                    act_b=11, coeff_a=0.4, coeff_b=0.4)
    dim_dict['act_dim'] = act_dim
    dim_dict['real_obs_dim'] = obs_dim

    # Setup trainer
    if algo == "PPO":
        trainer = PPOTrainerMTMT(config, dim_dict)
    else:
        raise NotImplementedError

    frame_stack_tensors = [FrameStackTensor(num_envs, main_envs.observation_space.shape, config.device),
                           FrameStackTensor(num_envs, aux_envs.observation_space.shape, config.device)]

    # Setup some stats helpers
    episode_rewards = [np.zeros([num_envs, 1], dtype=np.float), np.zeros([
        num_envs, 1], dtype=np.float)]

    total_episodes = total_steps = iteration = 0

    reward_recorders = [deque(maxlen=100), deque(maxlen=100)]
    episode_length_recorders = [deque(maxlen=100), deque(maxlen=100)]

    sample_timer = Timer()
    process_timer = Timer()
    update_timer = Timer()
    total_timer = Timer()
    progress = []
    evaluate_stat = {}

    # Start training
    print("Start training!")
    obs = [envs[i].reset() for i in range(2)]
    _ = [frame_stack_tensors[i].update(obs[i]) for i in range(2)]

    # first update
    for i in range(2):
        trainer.rollouts[i].observations[0].copy_(
            reduce_shape(frame_stack_tensors[i].get(), obs_dims[i]))

    branch_names = ['a', 'b']

    while True:  # Break when total_steps exceeds maximum value
        with sample_timer:
            # prepare rollout a
            for ind in range(2):
                for index in range(config.num_steps):
                    trainer.model.eval()
                    values, actions, action_log_prob = trainer.model.step(
                        reduce_shape(
                            frame_stack_tensors[ind].get(), obs_dims[ind]),
                        deterministic=False, branch=branch_names[ind])
                    cpu_actions = actions.cpu().numpy()
                    cpu_actions = enlarge_shape(cpu_actions, 17)

                    # obs, done, info not needed, we have masks & obs in frame_stack_tensors
                    _, reward, _, _, masks, new_total_episodes, new_total_steps, episode_rewards[ind] = \
                        step_envs(cpu_actions, envs[ind], episode_rewards[ind], frame_stack_tensors[ind],
                                  reward_recorders[ind], episode_length_recorders[ind],
                                  total_steps, total_episodes, config.device)

                    if ind == 0:
                        total_episodes = new_total_episodes
                        total_steps = new_total_steps

                    rewards = torch.from_numpy(
                        reward.astype(np.float32)).view(-1, 1).to(config.device)

                    trainer.rollouts[ind].insert(
                        reduce_shape(frame_stack_tensors[ind].get(),
                                     obs_dims[ind]), actions,
                        action_log_prob, values, rewards, masks)

        # ===== Process Samples =====
        with process_timer:
            with torch.no_grad():
                for i in range(2):
                    next_value = trainer.compute_values(
                        trainer.rollouts[i].observations[-1], branch_names[i])
                    trainer.rollouts[i].compute_returns(
                        next_value, config.GAMMA)

        trainer.model.train()
        # ===== Update Policy =====
        with update_timer:
            losses = trainer.update(trainer.rollouts[0], trainer.rollouts[1])
            policy_loss, value_loss, total_loss = list(zip(*losses))
            trainer.rollouts[0].after_update()
            trainer.rollouts[1].after_update()

        # ===== Evaluate Current Policy =====
        if iteration % config.eval_freq == 0:
            eval_timer = Timer()
            # seems ok, by default model is dealing with task1
            rewards, eplens = evaluate(trainer, eval_env, 1, dim_dict=dim_dict)
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
                training_episode_reward_a=summary(reward_recorders[0],
                                                  "episode_reward"),
                training_episode_length_a=summary(episode_length_recorders[0],
                                                  "episode_length"),
                training_episode_reward_b=summary(reward_recorders[1],
                                                  "episode_reward"),
                training_episode_length_b=summary(episode_length_recorders[1],
                                                  "episode_length"),
                evaluate_stats=evaluate_stat,
                learning_stats_a=dict(
                    policy_loss=policy_loss[0],
                    value_loss=value_loss[0],
                    total_loss=total_loss[0]
                ),
                learning_stats_b=dict(
                    policy_loss=policy_loss[1],
                    value_loss=value_loss[1],
                    total_loss=total_loss[1]
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
