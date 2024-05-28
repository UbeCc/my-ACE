import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.transforms import TanhTransform
from torch.distributions import Normal, TransformedDistribution

import numpy as np
from typing import Optional
from jaxtyping import Float, jaxtyped
from beartype import beartype

def mlp(input_size, layer_sizes, output_size, output_activation=nn.Identity, activation=nn.ELU):
    sizes = [input_size] + list(layer_sizes) + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, num_states, num_actions, action_space, hidden_dims = [400, 300], output_activation=nn.Tanh):
        super(Actor, self).__init__()
        self.action_space = action_space
        self.action_space.low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
        self.action_space.high = torch.as_tensor(self.action_space.high, dtype=torch.float32)
        self.fcs = mlp(num_states, hidden_dims, num_actions, output_activation=output_activation)
    
    def _normalize(self, action):
        return (action + 1) * (self.action_space.high - self.action_space.low) / 2 + self.action_space.low
    
    def to(self, device):
        self.action_space.low = self.action_space.low.to(device)
        self.action_space.high = self.action_space.high.to(device)
        return super().to(device)

    def forward(self, x):
        # use tanh as output activation
        return self._normalize(self.fcs(x))

class SoftActor(Actor):
    def __init__(self, num_states, num_actions, hidden_size, action_space, log_std_min, log_std_max):
        super().__init__(num_states, num_actions * 2, action_space, hidden_dims=hidden_size, output_activation=nn.Identity)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    @jaxtyped(typechecker=beartype)
    def forward(self, 
            state: Float[Tensor, "*batch_size state_dim"]
        ) -> tuple[Float[Tensor, "*batch_size action_dim"], Float[Tensor, "*batch_size action_dim"]]:
        """
        Obtain mean and log(std) from the fully-connected network.
        Crop the value of log_std to the specified range.
        """
        # Forward pass through the neural network to obtain mean and log_std

        output = self.fcs(state) 
        action_dim = output.shape[-1] // 2
        mean, log_std = torch.split(output, action_dim, dim=-1)
        normalized_mean = self._normalize(mean)

        # Clip the log_std to the specified range
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return normalized_mean, log_std

    @jaxtyped(typechecker=beartype)
    def evaluate(self, 
            state: Float[Tensor, "*batch_size state_dim"],
            sample: bool = True
        ) -> tuple[Float[Tensor, "*batch_size action_dim"], Optional[Float[Tensor, "*batch_size"]]]:
        
        mean, log_std = self.forward(state)
        
        # Create a normal distribution with the obtained mean and standard deviation
        std = log_std.exp()  # Calculate standard deviation from log_std
        normal_dist = torch.distributions.Normal(mean, std)
        
        if sample:
            # Reparameterization trick: sample from the normal distribution and use tanh
            z = normal_dist.rsample()  # Reparameterization trick
            action = torch.tanh(z)
            log_prob = normal_dist.log_prob(z).sum(dim=-1)
            
            # Account for the transformation through tanh
            log_prob -= torch.sum(torch.log(1 - action**2 + 1e-6), dim=-1)
            
        else:
            # Return the mean action if not sampling
            action = torch.tanh(mean)
            log_prob = None
        
        return self._normalize(action), log_prob

class ACEActor(Actor):
    def __init__(self, num_states, num_actions, hidden_size, action_space, log_std_min, log_std_max):
        super().__init__(num_states, num_actions * 2, action_space, hidden_dims=hidden_size, output_activation=nn.Identity)
        
        self.causal_default_weight = torch.ones(num_actions, dtype=torch.float32)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    @jaxtyped(typechecker=beartype)
    def forward(self, 
            state: Float[Tensor, "*batch_size state_dim"]
        ) -> tuple[Float[Tensor, "*batch_size action_dim"], Float[Tensor, "*batch_size action_dim"]]:
        """
        Obtain mean and log(std) from the fully-connected network.
        Crop the value of log_std to the specified range.
        """
        # Forward pass through the neural network to obtain mean and log_std

        output = self.fcs(state) 
        action_dim = output.shape[-1] // 2
        mean, log_std = torch.split(output, action_dim, dim=-1)
        normalized_mean = self._normalize(mean)

        # Clip the log_std to the specified range
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return normalized_mean, log_std

    def evaluate(self, state, action_weight=None):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
        #* compute causal weighted entropy
        if action_weight is None:
            action_weight = self.causal_default_weight
        if type(action_weight) == np.ndarray:
            action_weight = torch.from_numpy(action_weight)
        action_weight = action_weight.to(log_prob.device).clone().detach()
        log_prob = log_prob * action_weight.unsqueeze(0)
        mean = torch.tanh(mean)
        # return action, log_prob, mean
        # mean is for evaluation
        return action, log_prob.squeeze(-1)

class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dims):
        super().__init__()
        self.fcs = mlp(num_states + num_actions, hidden_dims, 1)

    def forward(self, state, action):
        return self.fcs(torch.cat([state, action], dim=1)).squeeze()