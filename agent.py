import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

GAMMA = 0.99            # discount factor
LR = 5.0e-4             # learning rate 
CLIP_RANGE = 0.20          # deviation to the old policy
GRAD_CLIP = 5           # gradient clipping
NUM_EPOCHS = 10         # number of batch passes
BATCH_SIZE = 256        # number of points in batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
torch.manual_seed(1)
class PPO_Agent():
    """
    Interacts with and learns from the environment through 
    Proximal Policy Optimization
    """
    def __init__(self, policy, max_tsteps = 1000, seed = 0):
        """Initialize an Agent object.
        
        Params
        ======
            actor_net (object): Policy Network
            clip_eps (float): finish the datasets, with points
            seed (int): random seed
        """
        self.seed = random.seed(seed)
        self.max_tsteps = max_tsteps
        self.policy = policy.to(device)
        self.memory = []
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = LR, eps = 1e-5)

    def save(self, filename = 'ppo_agent.pth'):
        """Save network weights to file"""
        torch.save(self.policy.state_dict(), filename)

    def train(self, env, max_episodes = 10):
        """
        Trains the agent
        """
        self.score_history = []
        running_score = deque(maxlen=100)
        for episode in range(max_episodes):
            score, t = self.__collect_experiences___(env, self.max_tsteps)
            score = np.mean(score)
            self.score_history.append(score)
            running_score.append(score)
            msg = 'Episode {:3d} of {} Avg/Last/Min/Max Score : {:.2f}/{:.2f}/{:.2f}/{:.2f}.'
            print(msg.format(episode, max_episodes, np.mean(running_score), score, np.min(running_score), np.max(running_score)))
            self.__update__()
        torch.save(self.policy.state_dict(), 'model.pth')
        self.score_history = np.array(self.score_history)
        np.savetxt('log.txt', self.score_history)
    
    def __collect_experiences___(self, env, max_tsteps):
        """
        Runs current policy and feeds the "replay buffer"
        """
        state = env.reset()
        score = 0
        temp_memory = []
        self.memory = []
        for t in range(max_tsteps):

            state = torch.tensor(state, dtype=torch.float, device=device)
            action, log_prob, value = self.policy(state)
            next_state, reward, done = env.step(action.cpu().detach().numpy())
            score += reward
            temp_memory.append([state, value.detach(), action.detach(), log_prob.detach(), reward])

            state = next_state

        # from IPython.core.debugger import Tracer
        # Tracer()()
        state = torch.tensor(state, dtype=torch.float, device=device)

        returns = self.policy(state)[-1].detach()
        temp_memory.append([state, value, None, None, None])
        # First step is to compute the advantages ... 
        self.memory = [None] * (len(temp_memory) - 1)
        for i in reversed(range(len(temp_memory) - 1)):

            states, values, actions, log_probs, rewards = temp_memory[i]
            rewards = torch.from_numpy(rewards).unsqueeze(1).float().to(device)
            returns = rewards + GAMMA * returns
            advantages = returns - values
            self.memory[i] = [states, actions, advantages, log_probs]

        return score, t + 1

    def __update__(self):
        """
        Train the networks according to Proximal Policy Optimization algorithm
        """
        for _ in range(NUM_EPOCHS):
            np.random.shuffle(self.memory)
            idx = list(range(0, len(self.memory), BATCH_SIZE)) + [len(self.memory)]
            idx = list(zip(idx, idx[1:]))
            for batch in idx:
                
                states, actions, advantages, old_log_probs = zip(*self.memory[batch[0]:batch[1]])
                states = torch.cat(states)
                actions = torch.cat(actions)
                advantages = torch.cat(advantages)
                old_log_probs = torch.cat(old_log_probs)

                # Surrogate actor loss...
                _, log_probs, values = self.policy(states, actions)
                r = (log_probs - old_log_probs).exp()
                
                actor_loss1 = r*advantages
                actor_loss2 = r.clamp(1 - CLIP_RANGE, 1 + CLIP_RANGE) * advantages
                actor_loss = -torch.min(actor_loss1, actor_loss2).mean()
                critic_loss = 0.5 * (advantages - values).pow(2).mean()
                loss = actor_loss + critic_loss

                self.policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), GRAD_CLIP)
                self.policy_optimizer.step()

                
                

    