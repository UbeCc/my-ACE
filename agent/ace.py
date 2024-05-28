import time
from copy import deepcopy
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import random
import itertools
import numpy as np
import torch.nn.functional as F
import pandas as pd
import lingam 
# from lingam import DirectLiNGAM
from causallearnmain.causallearn.search.FCMBased import lingam
import torch
from torch import Tensor
from agent.td3 import TD3Agent
from models import Critic, ACEActor
from copy import deepcopy

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

class ACEAgent(TD3Agent):
    def __init__(self, state_size, action_size, action_space, hidden_dim, lr_actor, lr_critic, gamma, tau, nstep, target_update_interval, log_std_min, log_std_max, device, threshold, reset_update_interval, lr_alpha):
        self.threshold = threshold
        self.model = lingam.DirectLiNGAM()
        
        self.critic_net = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic_net).to(device)
        self.critic_optimizer = torch.optim.AdamW(self.critic_net.parameters(), lr=lr_critic)
        self.critic_net_2 = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic_target_2 = deepcopy(self.critic_net_2).to(device)
        self.critic_optimizer_2 = torch.optim.AdamW(self.critic_net_2.parameters(), lr=lr_critic)
        self.actor_net = ACEActor(state_size, action_size, hidden_dim, deepcopy(action_space), log_std_min, log_std_max).to(device)
        self.actor_optimizer = torch.optim.AdamW(self.actor_net.parameters(), lr=lr_actor)

        self.tau = tau
        self.device = device
        self.gamma = gamma ** nstep
        self.target_update_interval = target_update_interval
        self.reset_update_interval = reset_update_interval
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device))

        self.train_step = 0        
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=lr_alpha)

    def __repr__(self):
        return 'ACEAgent'

    def get_causal_weights(self, state, action, reward, sample_size=512, causal_method="DirectLiNGAM"):
        indices = torch.randperm(state.shape[0], device=state.device)[:sample_size]
        states, actions, rewards = state[indices].cpu(), action[indices].cpu(), reward[indices].cpu()
        rewards = np.squeeze(rewards.numpy())
        rewards = np.reshape(rewards, (sample_size, 1))
        states = states.numpy()
        actions = actions.numpy()
        
        X_ori = np.hstack((states, actions, rewards))
        X = pd.DataFrame(X_ori, columns=list(range(X_ori.shape[1])))
        
        if causal_method == "DirectLiNGAM":
            # self.model.fit(X)
            # weight_r = self.model.adjacency_matrix_[-1,np.shape(states)[1]:(np.shape(states)[1]+np.shape(actions)[1])]

            start_time = time.time()  
            self.model.fit(X)
            weight_r = self.model.adjacency_matrix_[-1, :states.shape[1] + actions.shape[1]]
            end_time = time.time()
            print("TIMEIS", end_time - start_time)
        else:
            raise NotImplementedError
        
        weight = F.softmax(torch.tensor(weight_r, dtype=torch.float32), dim=0).numpy()
        # multiply by action shape
        weight *= weight.shape[0]

        state_weight, action_weight = weight[:states.shape[1]], weight[states.shape[1]:]
        return torch.tensor(state_weight, device=state.device), torch.tensor(action_weight, device=state.device)

    def get_causal_entropy(self, log_action_prob, action_weights):
        """
        Calculate the causal entropy based on the policy and causal weights.
        """
        # 根据论文中的公式 (2) 实现因果熵的计算
        action_weights = action_weights.to(log_action_prob.device)
        causal_entropy = -torch.sum(action_weights * log_action_prob.exp() * log_action_prob, dim=-1)
        return causal_entropy
    
    def get_dormant_degree(self):
        """
        Calculate the gradient dormancy degree.
        """
        dormant_count = 0
        total_count = 0

        # 遍历actor网络和两个critic网络的所有参数
        for param in itertools.chain(self.actor_net.parameters(), self.critic_net.parameters(), self.critic_net_2.parameters()):
            if param.grad is not None:
                # 计算参数梯度的L2范数
                l2_norm = torch.norm(param.grad, p=2)
                # 如果梯度的L2范数小于阈值，则认为该神经元休眠
                dormant_count += l2_norm < self.threshold
                total_count += 1

        # 计算休眠度，即休眠神经元的比例
        dormant_degree = dormant_count / total_count if total_count > 0 else 0
        return dormant_degree
    
    def update(self, batch, weights=None, state_weight=None, action_weight=None):
        state, action, reward, next_state, done = batch
        # 根据当前的state和action计算causal entropy，然后修改reward
        # reward = torch.tensor(self.compute_reward(state_weights * state, action_weights * action), device=state.device)
        actor_loss, alpha, log_action_prob, alpha_loss = self.update_actor(state, action_weight)
        critic_loss, critic_loss_2, td_error = self.update_critic(state, action, reward, next_state, done, weights, state_weight, action_weight, log_action_prob)
        
        if not self.train_step % self.target_update_interval:
            self.soft_update(self.critic_target, self.critic_net)
            self.soft_update(self.critic_target_2, self.critic_net_2)
        
        # 检查是否需要重置网络
        dormant_degree = self.get_dormant_degree()
        if self.train_step % self.reset_update_interval == 0 and dormant_degree > self.threshold:
            dormant_metrics = cal_dormant_grad(self.actor_net, type='policy', percentage=self.tau)
            if dormant_metrics:
                factor = perturb_factor(dormant_metrics['policy_grad_dormant_ratio'])
            else:
                factor = 1
            causal_diff = torch.max(action_weight, dim=0)[0] - torch.min(action_weight, dim=0)[0]
            dormant_metrics["causal_diff"] = causal_diff
            causal_factor = np.exp(-8 * causal_diff) - 0.5
            factor = perturb_factor(causal_factor)
            dormant_metrics["factor"] =factor
            if factor < 1: 
                perturb(self.actor_net, self.actor_optimizer, factor)
                perturb(self.critic_net, self.critic_optimizer, factor)
        self.train_step += 1
        return {'critic_loss': critic_loss, 'critic_loss_2': critic_loss_2, 'actor_loss': actor_loss, 'td_error': td_error, 'alpha_loss': alpha_loss}

    def reset_networks(self):
        """
        Reset the networks with a soft reset based on the current weights and random weights.
        """
        tau = self.tau  # 使用预设的tau值

        # 为actor和critic网络生成随机权重
        random_actor_weights = self.actor_net.state_dict()
        for key in random_actor_weights.keys():
            random_actor_weights[key] = torch.randn_like(self.actor_net.state_dict()[key])

        random_critic_weights = self.critic_net.state_dict()
        random_critic_weights_2 = self.critic_net_2.state_dict()
        
        # 软重置actor网络
        for key in self.actor_net.state_dict().keys():
            self.actor_net.state_dict()[key] = (1 - tau) * self.actor_net.state_dict()[key] + tau * random_actor_weights[key]
        
        # 软重置critic网络
        for key in self.critic_net.state_dict().keys():
            self.critic_net.state_dict()[key] = (1 - tau) * self.critic_net.state_dict()[key] + tau * random_critic_weights[key]
        
        for key in self.critic_net_2.state_dict().keys():
            self.critic_net_2.state_dict()[key] = (1 - tau) * self.critic_net_2.state_dict()[key] + tau * random_critic_weights_2[key]

    def update_alpha(self, action_log_prob):
        action_log_prob = action_log_prob.clone().detach()
        alpha_loss = self.get_alpha_loss(action_log_prob)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return self.log_alpha.exp(), alpha_loss

    def get_alpha_loss(self, action_log_prob):
        """
        Calculate alpha loss. You should view `log_alpha` as alpha in the loss calculation here to pass the autotest.
        """
        alpha_loss = -(self.log_alpha * (action_log_prob + self.target_entropy).detach()).mean()
        return alpha_loss

            
    def get_Qs(self, 
            state: Float[Tensor, "batch_size state_dim"], 
            action: Float[Tensor, "batch_size action_dim"], 
            reward: Float[Tensor, "batch_size"], 
            next_state: Float[Tensor, "batch_size state_dim"], 
            done: Int[Tensor, "batch_size"],
            state_weight: Float[Tensor, "state_size"],
            action_weight: Float[Tensor, "action_size"],
            log_action_prob
        ) -> tuple[Float[Tensor, "batch_size"], Float[Tensor, "batch_size"], Float[Tensor, "batch_size"]]:
        """
        Obtain the two Q value estimates and the target Q value from the twin Q networks.
        """
        with torch.no_grad():
            next_action, _ = self.actor_net.evaluate(next_state, action_weight.cpu().numpy())
            # 使用新的reward
            causal_entropy = self.get_causal_entropy(log_action_prob, action_weight)
            target_Q1 = self.critic_target(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            Q_target = reward + self.gamma * (1 - done) * (target_Q - self.log_alpha.exp() * causal_entropy + self.target_entropy)
        
        Q1 = self.critic_net(state, action)
        Q2 = self.critic_net_2(state, action)
        return Q1, Q2, Q_target
    
    def update_critic(self, state, action, reward, next_state, done, weights=None, state_weight=None, action_weight=None, log_action_prob=None):
        Q, Q2, Q_target = self.get_Qs(state, action, reward, next_state, done, state_weight=state_weight, action_weight=action_weight, log_action_prob=log_action_prob)
        with torch.no_grad():
            td_error = torch.abs(Q - Q_target)
    
        if weights is None:
            critic_loss = torch.mean((Q - Q_target)**2)
            critic_loss_2 = torch.mean((Q2 - Q_target)**2)
        else:
            critic_loss = torch.mean((Q - Q_target)**2 * weights)
            critic_loss_2 = torch.mean((Q2 - Q_target)**2 * weights)

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward(retain_graph=True)
        self.critic_optimizer_2.step()
        return critic_loss.item(), critic_loss_2.item(), td_error.mean().item()

    def get_actor_loss(self, state, action_weight):
        """
        Calculate actor loss and log prob using policy network.
        """
        action, log_action_prob = self.actor_net.evaluate(state, action_weight)
        Q1 = self.critic_net(state, action)
        actor_loss = -Q1.mean()
        return actor_loss, log_action_prob

    @jaxtyped(typechecker=beartype)
    def update_actor(self, state, action_weight):
        """
        Update the actor network using the critic's Q values.
        """
        # 根据论文中的描述，使用因果熵来更新actor
        actor_loss, log_action_prob = self.get_actor_loss(state, action_weight)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)      
        self.actor_optimizer.step()

        alpha, alpha_loss = self.update_alpha(log_action_prob)

        return actor_loss.item(), alpha.item(), log_action_prob, alpha_loss
    
    @torch.no_grad()
    def get_action(self, state, sample=False):
        action, _ = self.actor_net.evaluate(torch.as_tensor(state).to(self.device))
        return action.cpu().numpy()
    
class LinearOutputHook:
    def __init__(self):
        self.outputs = []
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

def cal_dormant_ratio(model, *inputs, type='policy', percentage=0.1):
    metrics = dict()
    hooks = []
    hook_handlers = []
    total_neurons = 0
    dormant_neurons = 0
    dormant_indices = dict()
    active_indices = dict()

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = LinearOutputHook()
            hooks.append(hook)
            hook_handlers.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(*inputs)

    count = 0
    for module, hook in zip(
        (module
         for module in model.modules() if isinstance(module, nn.Linear)),
            hooks):
        with torch.no_grad():
            for output_data in hook.outputs:
                mean_output = output_data.abs().mean(0)
                avg_neuron_output = mean_output.mean()
                dormant_indice = (mean_output < avg_neuron_output *
                                   percentage).nonzero(as_tuple=True)[0]
                all_indice = list(range(module.weight.shape[0]))
                active_indice = [index for index in all_indice if index not in dormant_indice]
                total_neurons += module.weight.shape[0]
                dormant_neurons += len(dormant_indice)
                module_dormant_ratio = len(dormant_indices) / module.weight.shape[0]
                if module_dormant_ratio > 0.1:
                    dormant_indices[str(count)] = dormant_indice
                    active_indices[str(count)] = active_indice
                count += 1

    for hook in hooks:
        hook.outputs.clear()

    for hook_handler in hook_handlers:
        hook_handler.remove()

    metrics[type + "_output_dormant_ratio"] = dormant_neurons / total_neurons

    return metrics, dormant_indices, active_indices

def cal_dormant_grad(model, type = 'policy', percentage=0.025):
    metrics = dict()
    total_neurons = 0
    dormant_neurons = 0
    count = 0
    for module in (module for module in model.modules() if isinstance(module, nn.Linear)):
        grad_norm = module.weight.grad.norm(dim=1)  
        avg_grad_norm = grad_norm.mean()
        dormant_indice = (grad_norm < avg_grad_norm * percentage).nonzero(as_tuple=True)[0]
        total_neurons += module.weight.shape[0]
        dormant_neurons += len(dormant_indice)
        module_dormant_grad = len(dormant_indice) / module.weight.shape[0]
        metrics[
                type + '_' + str(count) +
                '_grad_dormant'] = module_dormant_grad
        count += 1
    metrics[type + "_grad_dormant_ratio"] = dormant_neurons / total_neurons
    return metrics

def perturb(net, optimizer, perturb_factor):
    linear_keys = [
        name for name, mod in net.named_modules()
        if isinstance(mod, torch.nn.Linear)
    ]
    new_net = deepcopy(net)
    new_net.apply(weight_init)

    for name, param in net.named_parameters():
        if any(key in name for key in linear_keys):
            noise = new_net.state_dict()[name] * (1 - perturb_factor)
            param.data = param.data * perturb_factor + noise
        else:
            param.data = net.state_dict()[name]
    optimizer.state = defaultdict(dict)
    return net, optimizer

def dormant_perturb(model, optimizer, dormant_indices, perturb_factor=0.2):
    random_model = deepcopy(model)
    random_model.apply(weight_init)
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    random_layers = [module for module in random_model.modules() if isinstance(module, nn.Linear)]

    for key in dormant_indices:
        perturb_layer = linear_layers[key]
        random_layer = random_layers[key]
        with torch.no_grad():
            for index in dormant_indices[key]:
                noise = (random_layer.weight[index, :] * (1 - perturb_factor)).clone()
                perturb_layer.weight[index, :] = perturb_layer.weight[index, :] * perturb_factor + noise

    optimizer.state = defaultdict(dict)
    return model, optimizer
    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def perturb_factor(dormant_ratio, max_perturb_factor=0.9, min_perturb_factor=0.2):
    return min(max(min_perturb_factor, 1 - dormant_ratio), max_perturb_factor)