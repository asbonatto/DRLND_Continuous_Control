import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

class BaseNet(nn.Module):
    """
    Implements the basic feed forward network with relu activation layer
    """
    def __init__(self, input_dim, hidden_units=[64, 64]):
        super(BaseNet, self).__init__()
        dims = [input_dim, ] + hidden_units
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = F.tanh(layer(x))
        return x

class ActorCriticNet(nn.Module):
    """
    Represents the Actor/Critic network with shared parameters
    """
    def __init__(self, state_size, action_size, action_scale = 1.):
        """
        Args:
            action_scale (float array with output_dim shape): the parameter range for control output
        """
        super().__init__()

        # Actor/Critic network
        self.actor_body = BaseNet(state_size)
        self.actor_head = nn.Linear(self.actor_body.feature_dim, action_size)
        self.action_scale = action_scale
        self.std = torch.ones(1, action_size)

        self.critic_body = BaseNet(state_size)
        self.critic_head = nn.Linear(self.critic_body.feature_dim, 1)
        
    def forward(self, x, action = None):
        """
        Maps state -> a , p(a|s)
        If the action is passed to the network, the probability of the action is returned
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
       
        mean = self.actor_body(x)
        mean = torch.tanh(self.actor_head(mean))
        mean =  mean * self.action_scale

        # TODO: fix this ugly code
        if mean.is_cuda:
            self.std = self.std.cuda()

        dist = torch.distributions.Normal(mean, self.std)
        if action is None:
            action = dist.sample()
        
        # NOTE: the sum of logs is the joint probability of taking all actions together
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        value = self.critic_body(x)
        value = self.critic_head(value)

        return action, log_prob, value

