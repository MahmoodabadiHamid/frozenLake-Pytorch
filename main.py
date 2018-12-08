import numpy as np
import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

env = FrozenLakeEnv(map_name="8x8")
env = FrozenLakeEnv(is_slippery=False)
#env = gym.make("FrozenLake-v0").unwrapped

state_size = env.observation_space.n
action_size = env.action_space.n
lr = 0.0001

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def to_onehot(state_size,value):
  my_onehot = np.zeros((state_size))
  my_onehot[value] = 1.0
  return my_onehot

def trainIters(actor, critic, n_iters):
    scores = []
    sumScores=0
    avgScores = []
    dicIterReward={}
    dicIterScore = {}
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    sumOfi = 0
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()
        
        
        for i in count():
            #env.render()
            state = torch.FloatTensor(to_onehot(state_size,state))
            
            dist, value = actor(state), critic(state)
            
            action = dist.sample()
            
            next_state, reward, done, _ = env.step(int(action))
            
            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float))
            masks.append(torch.tensor([1-done], dtype=torch.float))
            
            state = next_state
            #if reward:
                #print('Iteration: {}, Score: {},  reward: {}'.format(iter, i, reward))
            sumOfi += i
            if done :
                if (reward):
                    dicIterReward[iter] = reward
                if(iter%9 == 0):
                    
                    dicIterScore[iter] = sumOfi / 9
                    sumOfi = 0
                break
            

            next_state = torch.FloatTensor(to_onehot(state_size, next_state))
            next_value = critic(next_state)
        
        returns = compute_returns(next_value, rewards, masks)
        
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    
    rewardLists = sorted(dicIterReward.items())
    scoreLists = sorted(dicIterScore.items())
    rx, ry = zip(*rewardLists)
    sx, sy = zip(*scoreLists)

    
    fig.add_subplot(211)
    plt.scatter(rx, ry, s=2)

    fig.add_subplot(212)
    plt.plot(sx, sy)
    #plt.figure(figsize=(70, 2))
    fig.set_figwidth(70)
    plt.show()
    
    
    #torch.save(actor, 'model/actor.pkl')
    #torch.save(critic, 'model/critic.pkl')
    env.close()


if __name__ == '__main__':
    if os.path.exists('model/actor.pkl') and False :
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size)
    if os.path.exists('model/critic.pkl') and False:
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size)
print('Starting...')
trainIters(actor, critic, n_iters=100)





