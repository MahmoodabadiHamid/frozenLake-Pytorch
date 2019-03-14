import rrt
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
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
        
        #_____________ End of Conv Layer definition ___________________________
    def forward(self, state):
        
        #state = self.transform(state.squeeze())
        #print('here')
        #print(type(state))
        state=state.to(device)
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
    #rewards=rewards.to(device)
    #masks=masks.to(device)
    R = next_value.to(device)
    returns = []
    for step in reversed(range(len(rewards))):
        rewards[step] = rewards[step].to(device)
        #gamma = gamma.to(device)
        R = R.to(device)
        masks[step] = masks[step].to(device)
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def main(actor_distance, actor_angle, critic, convolution, env, n_iters):
    optimizerActorDistance = optim.Adam(actor_distance.parameters(),lr)
    optimizerActorAngle = optim.Adam(actor_angle.parameters(),lr)
    optimizerC = optim.Adam(critic.parameters())
    cum_rewards = [0]
    all_avg_cum_rewards = [0]
        
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
            e = pygame.event.get()
            env.state = torch.zeros([1, 36*4], dtype=torch.float32).to(device)
            state = (env.getState().to(device))
            #env.render()
            
            state = convolution(state)
            state = ((state).to(device))
            #state = np.reshape()

            
            mu1,sigma1 = actor_distance(state)
            mu2,sigma2 = actor_angle(state)
            
            value = critic(state)
            #action = dist.sample()
            if sigma1 < 0:
                sigma1 *= -1
            if sigma2 < 0:
                sigma2 *= -1
            
            if mu1 < 0:
                mu1 *= -1
          
            if mu2 < 0:
                mu2 += 360
            if mu2 > 360:
                mu2  -=  360
            
            
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
            
                
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float))#, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float))#, device=device))

            state = next_state.to(device)

            if (done):
                env.terminate()
                break
                #print('hi')
            #print(state)
            #if done:
                #print('Iteration: {}, Score: {}'.format(iter, i))
                #break
            #env.updateDisplay()
                              
            env.mainClock.tick(env.FPS)
            if (env.playerHasHitBaddie()       or
              env.playerRect.top > env.winH   or
              env.playerRect.top < 0           or
              env.playerRect.left > env.winW  or
              env.playerRect.left < 0):
                break
        print('avg_reward: ', sum(cum_rewards)/len(cum_rewards))
        cum_rewards.append(cum_reward)
        cum_reward = 0
       

        
        
        next_state = ((convolution(next_state)))
        next_value = critic(next_state.to(device))
        returns = compute_returns(next_value, rewards, masks)
        #log_probs = torch.tensor(log_probs)
        #print(log_probs)
        #print(len(log_probs))
        #input()

        log_probs_distance = torch.cat(log_probs_distance).to(device)
        log_probs_angle = torch.cat(log_probs_angle).to(device)
        
        try:
        
            returns = torch.cat(returns).detach().to(device)
            values = torch.cat(values).to(device)
            
            advantage = returns - values
            
            actor_distance_loss = -(log_probs_distance* advantage.detach()).mean().to(device)
            #print("log prob",log_probs_angle)
            #print("advantage",advantage.detach())
            actor_angle_loss = -(log_probs_angle * advantage.detach()).mean().to(device)
            critic_loss = advantage.pow(2).mean().to(device)
    
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
            env =  gameEnv.game( level = 'EASY')
            
            torch.save(actor_distance, str(path)+'actor_distance.pkl')
            torch.save(actor_angle, str(path)+'actor_angle.pkl')
            torch.save(critic, str(path)+'critic.pkl')
            torch.save(convolution.to(device), str(path)+'convolution.pkl')
            
            
            #print(critic_loss)
            #print(actor_distance_loss)
            #print(actor_angle_loss)
            #for param in actor_angle.parameters():
            #    print(param)
            #input()
            #for param in actor_distance.parameters():
            #    print(param.grad)
            #input()
        except:
            #env =  gameEnv.game(actor_distance, actor_angle, critic, level = 'EASY')
            env =  gameEnv.game(level = 'EASY')

            print("something wrong happened")
        all_avg_cum_rewards.append(sum(cum_rewards)/len(cum_rewards))
        os.system('%clear')
        plt.figure(figsize=(20,5))
        plt.plot(list(range(0, len(all_avg_cum_rewards ))),all_avg_cum_rewards , '-b', label = 'reward average')
        plt.plot(list(range(0, len(cum_rewards))),cum_rewards, '-g', label = 'reward per game')
        plt.savefig('Reward Plot')
        plt.show()



if __name__ == '__main__':
    print('version 2')
    path = ''#input('input path: ')
    NUM_OF_RRT_ITER = 0
    NUM_OF_RRT_EPOCH = 10
    
    if os.path.exists(str(path)+'actor_distance.pkl'):
        actor_distance = torch.load(str(path)+'actor_distance.pkl').to(device)
        print('actor_distance Model loaded')
    else:
        actor_distance = Actor(state_size, action_size).to(device)
        print('actor_distance Model created')
        
    if os.path.exists(str(path)+'actor_angle.pkl'):
        actor_angle = torch.load(str(path)+'actor_angle.pkl').to(device)
        print('actor_angle Model loaded')
    else:
        actor_angle = Actor(state_size, action_size).to(device)
        print('actor_angle Model created')
        
    if os.path.exists(str(path)+'critic.pkl'): 
        critic = torch.load(str(path)+'critic.pkl').to(device)
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size=1).to(device)
        print('Critic Model created')
    if os.path.exists(str(path)+'convolution.pkl'):
        convolution = torch.load(str(path)+'convolution.pkl').to(device)
        print('convolution Model loaded')
    else:
        convolution = Convolution().to(device)
        print('convolution Model created')
    
#    env =  gameEnv.game(actor_distance, actor_angle, critic)
    env =  gameEnv.game(level = 'EASY')

    #env.FPS = 200
    for k in range(NUM_OF_RRT_ITER):
        print('RRT Iteration: ',str(k))
        #rrt_obj = rrt.RRT(env, actor_distance, actor_angle, convolution)
        #actor_distance, actor_angle, convolution = rrt_obj.runRRT(NUM_OF_RRT_EPOCH, path)
        
    env.FPS = 24
    n_iters = 100000# input('number of iteration? ')
    print('RRT training has been done!')
    main(actor_distance, actor_angle, critic, convolution, env, int(n_iters))

