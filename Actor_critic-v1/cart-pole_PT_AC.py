import pygame
from torch.autograd import Variable
from torchvision import transforms
import gameEnv
import os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#env = gym.make("CartPole-v0").unwrapped

state_size = 12*12#env.observation_space.shape[0]
action_size = 1#env.action_space.n
lr = 0.1e-3


class Convolution(nn.Module):
    def __init__(self):
        super(Convolution, self).__init__()
        #_____________ Starting CNN Layer ___________________________
        
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=5, padding=2),
                #nn.BatchNorm2d(1),
                #nn.ReLU(),
                nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=5, padding=2),
                #nn.BatchNorm2d(1),
                #nn.ReLU(),
                nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=5, padding=2),
                #nn.BatchNorm2d(1),
                #nn.ReLU(),
                nn.MaxPool2d(2))

        self.layer4 = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=5, padding=2),
                #nn.BatchNorm2d(1),
                #nn.ReLU(),
                nn.MaxPool2d(2))
        
        #_____________ End of CNN Layer ___________________________
    def forward(self, state):
        
        #state = self.transform(state.squeeze())
        #print('here')
        conv1 = self.layer1(state)
        conv2 = self.layer2(state)
        conv3 = self.layer3(state)
        conv4 = self.layer4(state)
        state = conv4.view(1, -1)
        return state

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
        distribution = Normal(torch.Tensor([0.0]),torch.Tensor([1.0]))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
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


def main(actor, critic, convolution, env, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(n_iters):
        #state = (env.getState())
        
        log_probs = []
        values = []
        rewards = []
        masks = []
        #entropy = 0
        #env.reset()

        #for i in count():
        while True :
            
            state = (env.getState())
            
            #env.render()
            
            state = convolution(state)
            state = (torch.FloatTensor(state))#.to(device))
            #state = np.reshape()

            
            dist, value = actor(state), critic(state)
            action = dist.sample()
            
            #print('dsf ',dist.log_prob(action).unsqueeze(0))
            log_prob = dist.log_prob(action).unsqueeze(0)
            log_probs.append(log_prob )
            
            #print(actor(state))
            #action = dist.sample()
            #print(action)
            #input() 
            next_state, reward, done, _ = env.step(action[0].detach().numpy())


            #print(reward)
            #input()
            if (done):
                env.terminate()
                break
                #print('hi')
                
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float))#, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float))#, device=device))

            state = next_state
            #print(state)
            #if done:
                #print('Iteration: {}, Score: {}'.format(iter, i))
                #break
            env.updateDisplay()
                              
            env.mainClock.tick(env.FPS)
            if (env.playerHasHitBaddie()       or
              env.playerRect.top > env.winH   or
              env.playerRect.top < 0           or
              env.playerRect.left > env.winW  or
              env.playerRect.left < 0):
                break
        next_state = torch.FloatTensor(convolution(next_state))#.to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        
        advantage = returns - values
        print(log_probs)
        actor_loss = -(log_probs[:len(advantage)] * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        print('advantage ',advantage)
        #print('actorloss', actor_loss)
        #input()
        #actor_loss = Variable(actor_loss , requires_grad = True)
        #critic_loss = Variable(critic_loss , requires_grad = True)
        #grad_fn=<NegBackward>
        #actor_loss = torch.tensor(actor_loss, requires_grad=True)
        #critic_loss = torch.tensor(critic_loss, requires_grad=True)
        
        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward(retain_graph=True)
        critic_loss.backward(retain_graph=True)
        optimizerA.step()
        optimizerC.step()
        env =  gameEnv.game(actor, critic, level = 'EASY')
        
        torch.save(actor, 'actor.pkl')
        torch.save(critic, 'critic.pkl')
        for param in actor.parameters():
            print(param)
        input()
        for param in actor.parameters():
            print(param.grad)
        input()
    env.close()


if __name__ == '__main__':
    if os.path.exists('model/actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size)#.to(device)
    if os.path.exists('model/critic.pkl'):
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size)#.to(device)
    convolution = Convolution()
    env =  gameEnv.game(actor, critic, level = 'EASY')
    #pygame.init()
    main(actor, critic, convolution, env, n_iters=3)
