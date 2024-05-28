import hydra
import utils
import torch
import logging
import gymnasium as gym
from omegaconf import OmegaConf
from dotmap import DotMap
from hydra.utils import instantiate
from gymnasium.wrappers import RecordEpisodeStatistics
from core import train
from buffer import get_buffer
from metaworld_env import metaworld_env
import os
os.environ["MUJOCO_GL"] = "egl"
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    	#  coffee-button-v2-goal-observable
	# if (config.env_name).endswith("goal-observable"):   
	# 	from .metaworld_env import metaworld_env
	# 	env = metaworld_env(config.env_name, config.seed, episode_length=200, reward_type=config.reward_type)
    if cfg.env_name.endswith("goal-observable"):
        env = RecordEpisodeStatistics(metaworld_env(cfg.env_name, 3407, episode_length=200))
    else:
        env = RecordEpisodeStatistics(gym.make(cfg.env_name, render_mode=None))
    device = f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu"
    state_size = utils.get_space_shape(env.observation_space)
    action_size = utils.get_space_shape(env.action_space)
    log_dict = utils.get_log_dict(cfg.agent._target_)
    for seed in cfg.seeds:
        utils.set_seed_everywhere(env, seed)
        # def reward_func_wrapper():
        #     reward_env = gym.make(cfg.env_name, render_mode=None)
        #     reward_env.reset(seed=seed)
        #     def compute_reward(state, action):
        #         rewards = []
        #         state = state.cpu().numpy()
        #         action = action.cpu().numpy()
        #         for i in range(state.shape[0]):
        #             reward_env.unwrapped.state = state[i]
        #             _, reward, _, _, _ = reward_env.step(action[i])
        #             rewards.append(reward)
        #         return rewards
        #     return compute_reward
        # buffer = get_buffer(cfg.buffer, state_size=state_size, action_size=action_size, device=device, seed=seed, compute_reward=reward_func_wrapper())
        buffer = get_buffer(cfg.buffer, state_size=state_size, action_size=action_size, device=device, seed=seed, compute_reward=None)
        agent = instantiate(cfg.agent, state_size=state_size, action_size=action_size, action_space=env.action_space, device=device)
        logger.info(f"Training seed {seed} for {cfg.train.timesteps} timesteps with {agent} and {buffer}")
        # get_attr of omega_conf is slow, so we convert it to dotmap
        train_cfg = DotMap(OmegaConf.to_container(cfg.train, resolve=True))
        eval_mean = train(train_cfg, env, agent, buffer, seed, log_dict)
        logger.info(f"Finish training seed {seed} with everage eval mean: {eval_mean}")


if __name__ == "__main__":
    main()
