import matplotlib.pyplot as plt
import numpy as np
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
action_size = 2#env.action_space.n
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
        output = self.linear3(output)[0]
        
        mu = output[0]
        sigma = output[1]
        
        #print('aaa')
        #input(output)
        #distribution_distance = torch.normal(mu1, sigma1)
        #distribution_angle = torch.normal(mu2, sigma2)
        #distribution = Normal(torch.Tensor([0.0]),torch.Tensor([1.0]))
        return mu, sigma #distribution_distance, distribution_angle


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


def main(actor_distance, actor_angle, critic, convolution, env, n_iters):
    optimizerActorDistance = optim.Adam(actor_distance.parameters())
    optimizerActorAngle = optim.Adam(actor_angle.parameters())
    optimizerC = optim.Adam(critic.parameters())
    cum_rewards = []
        
    for i in range(n_iters):
        #state = (env.getState())
        
        log_probs_distance = []
        log_probs_angle = []
        values = []
        rewards = []
        masks = []
        #entropy = 0
        #env.reset()

        #for i in count():
        cum_reward = 0
        while True :
            
            state = (env.getState())
            
            #env.render()
            
            state = convolution(state)
            state = (torch.FloatTensor(state))#.to(device))
            #state = np.reshape()

            
            mu1,sigma1 = actor_distance(state)
            mu2,sigma2 = actor_angle(state)
            
            value = critic(state)
            #action = dist.sample()
            
            #print('dsf ',dist.log_prob(action).unsqueeze(0))
            normal_dist_distance = Normal(mu1, sigma1)
            normal_dist_angle = Normal(mu2, sigma2)

            distance = normal_dist_distance.sample()
            angle    = normal_dist_angle.sample()
            

            #log_prob_distance = torch.log(torch.pow( torch.sqrt(2. * sigma1 * np.pi) , -1)) - (normal_dist_distance - mu1)*(normal_dist_distance - mu1)*torch.pow((2. * sigma1), -1)
            log_prob_distance = normal_dist_distance.log_prob(distance).unsqueeze(0)
            log_prob_angle = normal_dist_angle.log_prob(angle).unsqueeze(0)

            #log_prob_angle = torch.log(torch.pow( torch.sqrt(2. * sigma2 * np.pi) , -1)) - (normal_dist_distance - mu2)*(normal_dist_angle - mu2)*torch.pow((2. * sigma2), -1)

            entropy_distance = 0.5 * (torch.log(2. * np.pi * sigma1 ) + 1.)
            entropy_angle = 0.5 * (torch.log(2. * np.pi * sigma2 ) + 1.)
            

            log_prob_distance = log_prob_distance
            log_prob_angle = log_prob_angle
            
            log_probs_distance.append(log_prob_distance )
            log_probs_angle.append(log_prob_angle )
            
            
            next_state, reward, done, _ = env.step(distance, angle)

            cum_reward += reward
            #print(cum_reward)
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
        print('cum_reward: ', cum_reward)
        cum_rewards.append(cum_reward)
        cum_reward = 0
        
        
        next_state = torch.FloatTensor(convolution(next_state))#.to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)
        #log_probs = torch.tensor(log_probs)
        #print(log_probs)
        #print(len(log_probs))
        #input()

        log_probs_distance = torch.cat(log_probs_distance)
        log_probs_angle = torch.cat(log_probs_angle)
        
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        
        advantage = returns - values
        
        actor_distance_loss = -(log_probs_distance* advantage.detach()).mean()
        print("log prob",log_probs_angle)
        #print("advantage",advantage.detach())
        actor_angle_loss = -(log_probs_angle * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        #print('actorloss', actor_loss)
        #input()
        #actor_loss = Variable(actor_loss , requires_grad = True)
        #critic_loss = Variable(critic_loss , requires_grad = True)
        #grad_fn=<NegBackward>
        #actor_loss = torch.tensor(actor_loss, requires_grad=True)
        #critic_loss = torch.tensor(critic_loss, requires_grad=True)
        
        optimizerActorDistance.zero_grad()
        optimizerActorAngle.zero_grad()
        optimizerC.zero_grad()
        
        actor_distance_loss.backward(retain_graph=True)
        actor_angle_loss.backward(retain_graph=True)
        
        critic_loss.backward(retain_graph=True)
        optimizerActorDistance.step()
        optimizerActorAngle.step()
        optimizerC.step()
        env =  gameEnv.game(actor_distance, actor_angle, critic, level = 'EASY')
        
        torch.save(actor_distance, 'actor.pkl')
        torch.save(actor_angle, 'actor.pkl')
        torch.save(critic, 'critic.pkl')
        
        print(critic_loss)
        print(actor_distance_loss)
        print(actor_angle_loss)
        #for param in actor_angle.parameters():
        #    print(param)
        #input()
        #for param in actor_distance.parameters():
        #    print(param.grad)
        #input()
    #print(len(cum_rewards))
    plt.plot(list(range(0, len(cum_rewards))),cum_rewards)
    plt.show()
    #env.close()


if __name__ == '__main__':
    if os.path.exists('model/actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor_distance = Actor(state_size, action_size)#.to(device)
        actor_angle = Actor(state_size, action_size)#.to(device)
    if os.path.exists('model/critic.pkl'):
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size)#.to(device)
    convolution = Convolution()
    env =  gameEnv.game(actor_distance, actor_angle, critic, level = 'EASY')
    #pygame.init()
    n_iters = input('number of iteration? ')
    main(actor_distance, actor_angle, critic, convolution, env, int(n_iters))
