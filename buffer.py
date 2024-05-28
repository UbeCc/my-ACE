import torch
import numpy as np
from collections import deque
import copy

def get_buffer(cfg, **args):
    assert type(cfg.nstep) == int and cfg.nstep > 0, 'nstep must be a positive integer'
    her_args = {'goal_size': cfg.goal_size}
    if not cfg.use_per:
        if cfg.nstep == 1:
            # if cfg.use_her:
                # return HindsightReplayBuffer(cfg.capacity, **args, **her_args)
            # else:
                # return ReplayBuffer(cfg.capacity, **args)
            return ReplayBuffer(cfg.capacity, **args)
        else:
            return NStepReplayBuffer(cfg.capacity, cfg.nstep, cfg.gamma, **args)
    else:
        if cfg.nstep == 1:
            return PrioritizedHindsightReplayBuffer(cfg.capacity, cfg.per_eps, cfg.per_alpha, cfg.per_beta, **args, **her_args)
        else:
            raise NotImplementedError

class ReplayBuffer:
    def __init__(self, capacity, state_size, action_size, device, seed, **args):
        self.device = device
        self.states = torch.zeros(capacity, state_size, dtype=torch.float).contiguous()
        self.actions = torch.zeros(capacity, action_size, dtype=torch.float).contiguous()
        self.rewards = torch.zeros(capacity, dtype=torch.float).contiguous()
        self.next_states = torch.zeros(capacity, state_size, dtype=torch.float).contiguous()
        self.dones = torch.zeros(capacity, dtype=torch.int).contiguous()
        self.rng = np.random.default_rng(seed)
        self.idx = 0
        self.size = 0
        self.capacity = capacity
        self.default_state_weight = torch.ones(state_size, dtype=torch.float32)
        self.default_action_weight = torch.ones(action_size, dtype=torch.float32)

    def __repr__(self) -> str:
        return 'NormalReplayBuffer'

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition in the buffer
        self.states[self.idx] = torch.as_tensor(state)
        self.actions[self.idx] = torch.as_tensor(action)
        self.rewards[self.idx] = torch.as_tensor(reward)
        self.next_states[self.idx] = torch.as_tensor(next_state)
        self.dones[self.idx] = torch.as_tensor(done)

        # update counters
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size):
        # using np.random.default_rng().choice is faster https://ymd_h.gitlab.io/ymd_blog/posts/numpy_random_choice/
        sample_idxs = self.rng.choice(self.size, batch_size, replace=False)
        batch = (
            self.states[sample_idxs].to(self.device),
            self.actions[sample_idxs].to(self.device),
            self.rewards[sample_idxs].to(self.device),
            self.next_states[sample_idxs].to(self.device),
            self.dones[sample_idxs].to(self.device)
        )
        return batch


class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, n_step, gamma, state_size, action_size, device, seed):
        super().__init__(capacity, state_size, action_size, device, seed)
        self.n_step = n_step
        self.n_step_buffer = deque([], maxlen=n_step)
        self.gamma = gamma

    def __repr__(self) -> str:
        return f'{self.n_step}StepReplayBuffer'

    def n_step_handler(self):
        """Get n-step state, action, reward and done for the transition, discard those rewards after done=True"""
        state, action, reward, done = self.n_step_buffer[0]
        for i in range(1, self.n_step):
            _, _, cur_reward, cur_done = self.n_step_buffer[i]
            reward += self.gamma ** i * cur_reward
            done = done or cur_done
            if done:
                break
        return state, action, reward, done

    def add(self, transition):
        state, action, reward, next_state, done = transition
        self.n_step_buffer.append((state, action, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        state, action, reward, done = self.n_step_handler()
        super().add((state, action, reward, next_state, done))


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, state_size, action_size, device, seed):
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.eps = eps  # minimal priority for stability
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps
        super().__init__(capacity, state_size, action_size, device, seed)

    def add(self, transition):
        self.priorities[self.idx] = self.max_priority
        super().add(transition)

    def sample(self, batch_size):
        sample_idxs = self.rng.choice(self.capacity, batch_size, p=self.priorities / self.priorities.sum(), replace=True)
        
        # Get the importance sampling weights for the sampled batch using the prioity values
        # For stability reasons, we always normalize weights by max(w_i) so that they only scale the
        # update downwards, whenever importance sampling is used, all weights w_i were scaled so that max_i w_i = 1.
        
        weights = np.power(self.size * self.priorities[sample_idxs], -self.beta)        
        max_weight = weights.max()
        weights = torch.tensor(weights / max_weight).to(self.device)
    
        batch = (
            self.states[sample_idxs].to(self.device, non_blocking=True),
            self.actions[sample_idxs].to(self.device, non_blocking=True),
            self.rewards[sample_idxs].to(self.device, non_blocking=True),
            self.next_states[sample_idxs].to(self.device, non_blocking=True),
            self.dones[sample_idxs].to(self.device, non_blocking=True)
        )
        return batch, weights, sample_idxs

    def update_priorities(self, data_idxs, priorities: np.ndarray):
        priorities = (priorities + self.eps) ** self.alpha

        self.priorities[data_idxs] = priorities
        self.max_priority = np.max(self.priorities)

    def __repr__(self) -> str:
        return 'PrioritizedReplayBuffer'

class HindsightReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, state_size, action_size, device, seed, goal_size, compute_reward):
        super().__init__(capacity, state_size, action_size, device, seed)
        self.goal_size = goal_size
        self.goals = torch.zeros(capacity, goal_size, dtype=torch.float).contiguous()
        self.compute_reward = compute_reward

    def add(self, transition, goal):
        state, action, reward, next_state, done = transition
        self.goals[self.idx] = torch.as_tensor(goal, dtype=torch.float)
        # Compute reward based on the new goal if necessary
        # modified_reward = self.compute_reward(state, next_state, goal)
        # Store the modified transition
        super().add((state, action, reward, next_state, done))
        if done:
            self.store_episode()
            self.goals = []

    def sample(self, batch_size):
        batch = super().sample(batch_size)
        goals = self.goals[batch[0].indices].to(self.device)
        return (*batch, goals)
    
    def _sample_achieved_goal(self, goals, transition_idx):
        # here, we use `future` goal selection strategy
        selected_idx = np.random.choice(np.arange(transition_idx + 1, len(goals)))
        selected_transition = goals[selected_idx]
        return self.env.convert_obs_to_dict(selected_transition[0])['achieved_goal']

    def _sample_achieved_goals(self, episode_transitions, transition_idx):
        """
        Sample a batch of achieved goals according to the sampling strategy.
        :param episode_transitions: ([tuple]) list of the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        return [
            self._sample_achieved_goal(episode_transitions, transition_idx)
            for _ in range(self.n_sampled_goal)
        ]

    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions
        for transition_idx, transition in enumerate(self.episode_transitions):

            obs_t, action, reward, obs_tp1, done, info = transition

            # Add to the replay buffer
            self.replay_buffer.add(obs_t, action, reward, obs_tp1, done)

            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx == len(self.episode_transitions) - 1):
                break

            # Sampled n goals per transition, where n is `n_sampled_goal`
            # this is called k in the paper
            sampled_goals = self._sample_achieved_goals(self.episode_transitions, transition_idx)
            # For each sampled goals, store a new transition
            for goal in sampled_goals:
                # Copy transition to avoid modifying the original one
                obs, action, reward, next_obs, done, info = copy.deepcopy(transition)

                # Convert concatenated obs to dict, so we can update the goals
                obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs, next_obs))

                # Update the desired goal in the transition
                obs_dict['desired_goal'] = goal
                next_obs_dict['desired_goal'] = goal

                # Update the reward according to the new desired goal
                reward = self.env.compute_reward(next_obs_dict['achieved_goal'], goal, info)
                # Can we use achieved_goal == desired_goal?
                done = False

                # Transform back to ndarrays
                obs, next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict))

                # Add artificial transition to the replay buffer
                self.replay_buffer.add(obs, action, reward, next_obs, done)
                
class PrioritizedHindsightReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, state_size, action_size, device, seed, goal_size, compute_reward):
        super().__init__(capacity, eps, alpha, beta, state_size, action_size, device, seed)
        self.goal_size = goal_size
        self.goals = torch.zeros(capacity, goal_size, dtype=torch.float).contiguous()
        self.compute_reward = compute_reward

    def add(self, transition, goal):
        state, action, reward, next_state, done = transition
        self.goals[self.idx] = torch.as_tensor(goal, dtype=torch.float)
        
        # Compute reward based on the new goal
        modified_reward = self.compute_reward(state, next_state, goal)
        
        # Store the modified transition
        super().add((state, action, modified_reward, next_state, done))

    def sample(self, batch_size):
        batch, weights, sample_idxs = super().sample(batch_size)
        goals = self.goals[sample_idxs].to(self.device)
        return (*batch, goals, weights, sample_idxs)