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


state_size = 1*36
action_size = 2

class ActorCritic(torch.nn.Module):
    def __init__(self , enc_in ):
        super(ActorCritic, self).__init__( )
        
        enc_in = 9 * 4 
        enc_out = 18
        
        dec_in = enc_out 
        dec_out = 9
        
        last_layer_fc_out = 36
        
        self.conv1 = torch.nn.Sequential(nn.Conv2d(1, 1, kernel_size=5, padding=2),
                                          torch.nn.LeakyReLU(),
                                          nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(nn.Conv2d(1, 1, kernel_size=5, padding=2),
                                          #torch.nn.LeakyReLU(),
                                          #nn.BatchNorm2d(1),
                                          nn.MaxPool2d(2))
        self.conv3 = torch.nn.Sequential(nn.Conv2d(1, 1, kernel_size=5, padding=2),
                                          torch.nn.LeakyReLU(),
                                          #nn.BatchNorm2d(1),
                                          nn.MaxPool2d(2))
       
        
        
        self.fc_enc  = nn.Linear(enc_in,enc_out) # enc_input_layer
        
        
        self.actor_mu_dec   =nn.Linear(dec_in, last_layer_fc_out)
        self.actor_mu = nn.Linear(last_layer_fc_out, 2)
        
        self.actor_sigma_dec= nn.Linear(dec_in, last_layer_fc_out)
        self.actor_sigma = nn.Linear(last_layer_fc_out, 2)
        
        
        self.critic_dec     = nn.Linear(dec_in, last_layer_fc_out)
        self.critic_linear = nn.Linear(last_layer_fc_out, 1)
        
        
        
        
        self.state = torch.zeros([1, 9*4], dtype=torch.float32).to(device)
        self.state.requires_grad_()
        self.train()  

        
    def forward(self, inputs):
        
        x = inputs
        
        x.requires_grad_()
        x = x.to(device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        #print(x.shape)
        self.state = torch.cat((self.state, x), 1)
        self.state = self.state[:,x.shape[1]:]
        #print(self.state)

        x = F.relu(self.fc_enc(self.state))
        
        mu_dec = self.actor_mu_dec(x)
        sigma_dec = self.actor_sigma_dec(x)
        critic_dec = self.critic_dec(x)
        return self.actor_mu(mu_dec)[0], self.actor_sigma(sigma_dec)[0], self.critic_linear(critic_dec) 
                  

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value.to(device)
    returns = []
    for step in reversed(range(len(rewards))):
        rewards[step] = rewards[step].to(device)
        R = R.to(device)
        masks[step] = masks[step].to(device)
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def main(n_iters):
    path = ''#input('input path: ')
    NUM_OF_RRT_ITER = 00
    NUM_OF_RRT_EPOCH = 10
    

    if os.path.exists(str(path)+'ac.pkl'):
        ac = torch.load(str(path)+'ac.pkl').to(device)
        print('ac Model loaded')
    else:
        ac = ActorCritic(state_size).to(device)
        print('ac Model created')
        
 
    env =  gameEnv.game(level = 'EASY')
    for k in range(NUM_OF_RRT_ITER):
        print('RRT Iteration: ',str(k))
        #rrt_obj = rrt.RRT(env, actor_distance, actor_angle, convolution)
        #actor_distance, actor_angle, convolution = rrt_obj.runRRT(NUM_OF_RRT_EPOCH, path)
    print('RRT training has been done!')
    lr = .01
    optimizer = optim.Adam(ac.parameters(), lr = lr)
    
    cum_rewards = []
    all_avg_cum_rewards = []
        
    for i in range(1, n_iters):
        
        if (i % 100 == 0):
            print(i)
            print(lr)
            lr = float("{0:.3f}".format(lr*0.9))
            print('lr: ', str(lr))
            optimizer = optim.Adam(ac.parameters(), lr = lr)
            
        log_probs_distance = []
        log_probs_angle = []
        values = []
        rewards = []
        masks = []


        cum_reward = 0
        done = False
        distance_avarage = 1
        while True :
            e = pygame.event.get()
            env.state = torch.zeros([1, 36*4], dtype=torch.float32).to(device)
            state = (env.getState().to(device))
            
            
            
            actor_mu, actor_sigma, critic = ac(state)
            mu1, mu2, sigma1, sigma2, value = actor_mu[0], actor_mu[1] , actor_sigma[0], actor_sigma[1], critic
            
            
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
            

            normal_dist_distance = Normal(mu1, sigma1)
            normal_dist_angle = Normal(mu2, sigma2)

            distance = normal_dist_distance.sample()
            if distance>500:
                distance=50
            angle    = normal_dist_angle.sample()
            
            log_prob_distance = normal_dist_distance.log_prob(distance).unsqueeze(0)
            log_prob_angle = normal_dist_angle.log_prob(angle).unsqueeze(0)


            entropy_distance = 0.5 * (torch.log(2. * np.pi * sigma1 ) + 1.)
            entropy_angle = 0.5 * (torch.log(2. * np.pi * sigma2 ) + 1.)
            

            log_prob_distance = log_prob_distance
            log_prob_angle = log_prob_angle
            
            log_probs_distance.append(log_prob_distance )
            log_probs_angle.append(log_prob_angle )
            
            
            next_state, reward, done, _ = env.step(distance, angle)

            cum_reward += reward
            
            if (done):
                env.terminate()
                break
                
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float))
            masks.append(torch.tensor([1-done], dtype=torch.float))

            state = next_state.to(device)
            
            if (env.playerHasHitBaddie()       or
              env.playerRect.top > env.winH   or
              env.playerRect.top < 0           or
              env.playerRect.left > env.winW  or
              env.playerRect.left < 0):
                break
        
        print('cum_reward: ', cum_reward)
        cum_rewards.append(cum_reward)
        print('all avg reward: ' + str(sum(cum_rewards)/len(cum_rewards)))
        cum_reward = 0
        
        
        _, _, next_value = ac(((next_state)))
        
        
        returns = compute_returns(next_value, rewards, masks)



        log_probs_distance = torch.cat(log_probs_distance).to(device)
        log_probs_angle = torch.cat(log_probs_angle).to(device)
        
        try:
        
            returns = torch.cat(returns).detach().to(device)
            values = torch.cat(values).to(device)
            
            advantage = returns - values
            
            actor_distance_loss = -(log_probs_distance* advantage.detach()).mean().to(device)
            
            actor_angle_loss = -(log_probs_angle * advantage.detach()).mean().to(device)
            critic_loss = advantage.pow(2).mean().to(device)
            
            loss_average = (critic_loss + actor_angle_loss + actor_distance_loss)/3

            optimizer.zero_grad()
            loss_average.backward(retain_graph=True)
            optimizer.step()


            env =  gameEnv.game( level = 'EASY')
            
            torch.save(ac, str(path)+'ac.pkl')
 
        except Exception as e:  
            env =  gameEnv.game(level = 'EASY')

        all_avg_cum_rewards.append(sum(cum_rewards)/len(cum_rewards))
                
        plt.figure(figsize=(20,5))
        plt.plot(list(range(0, len(all_avg_cum_rewards ))),all_avg_cum_rewards , '-b', label = 'reward average')
        plt.plot(list(range(0, len(cum_rewards))),cum_rewards, '-g', label = 'reward per game')
        plt.savefig('Reward Plot')
        plt.show()

    


if __name__ == '__main__':
    n_iters = 100000
    main( int(n_iters))

