import torch
from torch import Tensor
from agent.td3 import TD3Agent
from models import SoftActor, Critic
from copy import deepcopy

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

class SACAgent(TD3Agent):
    def __init__(self, state_size, action_size, action_space, hidden_dim, lr_actor, lr_critic, lr_alpha, gamma, tau, nstep, target_update_interval, log_std_min, log_std_max, device, **args):
        self.critic_net = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic_net).to(device)
        self.critic_optimizer = torch.optim.AdamW(self.critic_net.parameters(), lr=lr_critic)
        self.critic_net_2 = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic_target_2 = deepcopy(self.critic_net_2).to(device)
        self.critic_optimizer_2 = torch.optim.AdamW(self.critic_net_2.parameters(), lr=lr_critic)
        self.actor_net = SoftActor(state_size, action_size, hidden_dim, deepcopy(action_space), log_std_min, log_std_max).to(device)
        self.actor_optimizer = torch.optim.AdamW(self.actor_net.parameters(), lr=lr_actor)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=lr_alpha)

        self.tau = tau
        self.device = device
        self.gamma = gamma ** nstep
        self.target_update_interval = target_update_interval
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device))

        self.train_step = 0

    def __repr__(self):
        return 'SACAgent'

    def update(self, batch, weights=None):
        state, action, reward, next_state, done = batch
        critic_loss, critic_loss_2, td_error = self.update_critic(state, action, reward, next_state, done, weights)
        actor_loss, alpha = self.update_actor(state)
        if not self.train_step % self.target_update_interval:
            self.soft_update(self.critic_target, self.critic_net)
            self.soft_update(self.critic_target_2, self.critic_net_2)
        self.train_step += 1
        return {'critic_loss': critic_loss, 'critic_loss_2': critic_loss_2, 'actor_loss': actor_loss, 'alpha': alpha, 'td_error': td_error}

    @jaxtyped(typechecker=beartype)
    def get_Qs(self, 
            state: Float[Tensor, "batch_size state_dim"], 
            action: Float[Tensor, "batch_size action_dim"], 
            reward: Float[Tensor, "batch_size"], 
            next_state: Float[Tensor, "batch_size state_dim"], 
            done: Int[Tensor, "batch_size"]
        ) -> tuple[Float[Tensor, "batch_size"], Float[Tensor, "batch_size"], Float[Tensor, "batch_size"]]:
        """
        Obtain the two Q value estimates and the target Q value from the twin Q networks.
        """
        with torch.no_grad():
            next_action, log_prob = self.actor_net.evaluate(next_state, sample=True)
            target_Q1 = self.critic_target(next_state, next_action)
            target_Q2 = self.critic_target_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            alpha = self.log_alpha.exp()
            Q_target = reward + self.gamma * (1 - done) * (target_Q - alpha * log_prob)
        
        Q1 = self.critic_net(state, action)
        Q2 = self.critic_net_2(state, action)
        
        return Q1, Q2, Q_target
    
    def update_critic(self, state, action, reward, next_state, done, weights=None):
        Q, Q2, Q_target = self.get_Qs(state, action, reward, next_state, done)
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

    @jaxtyped(typechecker=beartype)
    def get_actor_loss(self, 
            state: Float[Tensor, "batch_size state_dim"]
        ) -> tuple[Float[Tensor, ""], Float[Tensor, "batch_size"]]:
        """
        Calculate actor loss and log prob using policy network.
        """
        action, log_prob = self.actor_net.evaluate(state, sample=True)
        Q1 = self.critic_net(state, action)
        actor_loss = (self.log_alpha.exp() * log_prob - Q1).mean()
        return actor_loss, log_prob
    
    def update_actor(self, state):
        actor_loss, action_log_prob = self.get_actor_loss(state)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)      
        self.actor_optimizer.step()

        alpha = self.update_alpha(action_log_prob)

        return actor_loss.item(), alpha.item()
    
    @jaxtyped(typechecker=beartype)
    def get_alpha_loss(self, 
            action_log_prob: Float[Tensor, "batch_size"]
        ) -> Float[Tensor, ""]:
        """
        Calculate alpha loss. You should view `log_alpha` as alpha in the loss calculation here to pass the autotest.
        """
        alpha_loss = -(self.log_alpha * (action_log_prob + self.target_entropy)).mean()
        return alpha_loss
    
    def update_alpha(self, action_log_prob):
        action_log_prob = action_log_prob.clone().detach()
        alpha_loss = self.get_alpha_loss(action_log_prob)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return self.log_alpha.exp()

    @torch.no_grad()
    def get_action(self, state, sample=False):
        action, _ = self.actor_net.evaluate(torch.as_tensor(state).to(self.device), sample)
        return action.cpu().numpy()
