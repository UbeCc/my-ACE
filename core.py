import logging
import numpy as np
from buffer import ReplayBuffer, PrioritizedReplayBuffer
from copy import deepcopy
from utils import merge_videos, visualize
from gymnasium.wrappers import RecordVideo
logger = logging.getLogger(__name__)

def eval(env, agent, episodes, seed):
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=episode + seed)
        done, truncated = False, False

        while not (done or truncated):
            state, _, done, truncated, info = env.step(agent.get_action(state))
        returns.append(info['episode']['r'].item())
    return np.mean(returns), np.std(returns)

def train(cfg, env, agent, buffer, seed, log_dict):
    eval_env = deepcopy(env)
    for key in log_dict.keys():
        log_dict[key].append([])
    
    done, truncated, best_reward = False, False, -np.inf
    state, _ = env.reset(seed=seed)
    for step in range(1, cfg.timesteps + 1):
        if not step % cfg.update_causal_weight_interval or step == 1:
            if buffer.size < cfg.update_cw_batch_size:
                state_weight, action_weight = buffer.default_state_weight, buffer.default_action_weight
            else:
                cw_state, cw_action, cw_reward, _, _ = buffer.sample(cfg.update_cw_batch_size)
                state_weight, action_weight = agent.get_causal_weights(cw_state, cw_action, cw_reward)
        if done or truncated:
            state, _ = env.reset()
            done, truncated = False, False
            log_dict['train_returns'][-1].append(info['episode']['r'].item())
            log_dict['train_steps'][-1].append(step - 1)

        action = agent.get_action(state, sample=True)
        next_state, reward, done, truncated, info = env.step(action)
        buffer.add((state, action, reward, next_state, int(done)))
        state = next_state

        if step > cfg.batch_size + cfg.nstep:
            if isinstance(buffer, PrioritizedReplayBuffer):
                batch, weights, tree_idxs = buffer.sample(cfg.batch_size)
                ret_dict = agent.update(batch, weights=weights, state_weight=state_weight, action_weight=action_weight)
                buffer.update_priorities(tree_idxs, ret_dict['td_error'])

            elif isinstance(buffer, ReplayBuffer):
                batch = buffer.sample(cfg.batch_size)
                ret_dict = agent.update(batch, state_weight=state_weight, action_weight=action_weight)
            else:
                raise RuntimeError("Unknown buffer")

            for key in ret_dict.keys():
                log_dict[key][-1].append(ret_dict[key])

        if step % cfg.eval_interval == 0:
            eval_mean, eval_std = eval(eval_env, agent=agent, episodes=cfg.eval_episodes, seed=seed)
            log_dict['eval_steps'][-1].append(step - 1)
            log_dict['eval_returns'][-1].append(eval_mean)
            logger.info(f"Seed: {seed}, Step: {step}, Eval mean: {eval_mean}, Eval std: {eval_std}")
            if eval_mean > best_reward:
                best_reward = eval_mean
                agent.save(f'best_model_seed_{seed}')

        if step % cfg.plot_interval == 0:
            visualize(step, f'{agent} with {buffer}', log_dict)

    agent.save(f'final_model_seed_{seed}')
    visualize(step, f'{agent} with {buffer}', log_dict)

    env = RecordVideo(eval_env, f'final_videos_seed_{seed}', name_prefix='eval', episode_trigger=lambda x: x % 3 == 0 and x < cfg.eval_episodes, disable_logger=True)
    eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes, seed=seed)

    agent.load(f'best_model_seed_{seed}')  # use best model for visualization
    env = RecordVideo(eval_env, f'best_videos_seed_{seed}', name_prefix='eval', episode_trigger=lambda x: x % 3 == 0 and x < cfg.eval_episodes, disable_logger=True)
    eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes, seed=seed)
    merge_videos(f'final_videos_seed_{seed}')
    merge_videos(f'best_videos_seed_{seed}')
    env.close()
    return eval_mean
