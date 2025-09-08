import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
# from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'
from Close_Env import RLEnv
import wandb


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, step, datanum):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(1, 9):
		obs, done, ep_reward, t = env.reset(i, 'test'), False, 0, 0
		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			ep_reward += reward
			t += 1
		episode_rewards.append(ep_reward)
		env.save_test(datanum)
	return np.nanmean(episode_rewards)


def train(cfg):
	"""Training script for TD-MPC. Requires a CUDA-enabled device."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	env, agent, buffer = RLEnv(), TDMPC(cfg), ReplayBuffer(cfg)
	
	# Run training
	L = logger.Logger(work_dir, cfg)
	wandb.init(project='tdmpc', name='close')
	episode_idx, start_time = 0, time.time()
	data_total_num = 0
	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):

		# Collect trajectory
		obs = env.reset()
		episode = Episode(cfg, obs)
		while not episode.done:
			action = agent.plan(obs, step=step, t0=episode.first)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			episode += (obs, action, reward, done)
			data_total_num += 1
		# assert len(episode) == cfg.episode_length
		buffer += episode

		# Update model
		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i))

		# Log training episode
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward}
		train_metrics.update(common_metrics)
		L.log(train_metrics, category='train')
		wandb.log(train_metrics)

		# Evaluate agent periodically
		if env_step < 36500:
			if env_step % cfg.eval_freq == 0:
				common_metrics['episode_reward'] = evaluate(env, agent, step, data_total_num)
				L.log(common_metrics, category='eval')
		else:
			if env_step % (cfg.eval_freq * 5) == 0:
				common_metrics['episode_reward'] = evaluate(env, agent, step, data_total_num)
				L.log(common_metrics, category='eval')

	L.finish(agent)
	wandb.finish()
	print('Training completed successfully')


if __name__ == '__main__':
	train(parse_cfg(Path().cwd() / __CONFIG__))
