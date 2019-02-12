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

state_size = 1*36
#env.observation_space.shape[0]
action_size = 2#env.action_space.n

class ActorCritic(torch.nn.Module):
    def __init__(self , enc_in ):
        super(ActorCritic, self).__init__( )
        lstm_out = 256
        lstm_in = lstm_out
        
        enc_in = 3 # for pendulum
        enc_hidden = 200
        enc_out = lstm_in
        self.layer1 = torch.nn.Sequential(nn.Conv2d(1, 1, kernel_size=5, padding=2),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(nn.Conv2d(1, 1, kernel_size=5, padding=2),
                                          torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(nn.Conv2d(1, 1, kernel_size=5, padding=2),
                                          torch.nn.ReLU())
        
        self.fc_enc_in  = nn.Linear(625,enc_hidden) # enc_input_layer
        self.fc_enc_out  = nn.Linear(enc_hidden,enc_out) # enc_output_layer
        self.lstm = nn.LSTMCell(lstm_in, lstm_out)
        self.actor_mu = nn.Linear(lstm_out, 1)
        self.actor_sigma = nn.Linear(lstm_out, 1)
        self.critic_linear = nn.Linear(lstm_out, 1)   
        self.train()  
        
    def forward(self, inputs):
        
        x, (hx, cx) = inputs
        
        x.requires_grad_()
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = F.relu(self.fc_enc_in(x))
        x = self.fc_enc_out(x)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_mu(x), self.actor_sigma(x), (hx, cx)

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


def main(n_iters):
    path = ''#input('input path: ')
    NUM_OF_RRT_ITER = 0
    NUM_OF_RRT_EPOCH = 10
    

    if os.path.exists(str(path)+'ac.pkl'):
        ac = torch.load(str(path)+'ac.pkl').to(device)
        print('ac Model loaded')
    else:
        ac = ActorCritic(state_size,2 ,1  ,device = device).to(device)
        print('ac Model created')
        
 
    env =  gameEnv.game(level = 'EASY')
    #env.FPS = 200
    for k in range(NUM_OF_RRT_ITER):
        print('RRT Iteration: ',str(k))
        #rrt_obj = rrt.RRT(env, actor_distance, actor_angle, convolution)
        #actor_distance, actor_angle, convolution = rrt_obj.runRRT(NUM_OF_RRT_EPOCH, path)
    print('RRT training has been done!')
    env.FPS = 24
    optimizer = optim.Adam(ac.parameters(), lr = 0.01)
    #optimizerActorAngle = optim.Adam(actor_angle.parameters(), lr = 0.01)
    #optimizerC = optim.Adam(critic.parameters(), lr = 0.01)
    cum_rewards = []
    all_avg_cum_rewards = []
        
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
            env.updateDisplay()
            state = (env.getState().to(device))
            
            #env.render()
            
            actor_mu, actor_sigma, critic = ac(state)
            mu1, mu2, sigma1, sigma2, value = actor_mu[0], actor_mu[1] , actor_sigma[0], actor_sigma[1], critic
            
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

            state = next_state.to(device)
            
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
        print('cum_reward: ', cum_reward)
        cum_rewards.append(cum_reward)
        cum_reward = 0
        
        
        #next_state = ((convolution(next_state)))
        #next_value = critic(next_state.to(device))
        
        _, _, next_value = ac(next_state)
        
        
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
            loss_average = (critic_loss + actor_angle_loss + actor_distance_loss)/3
            #print('actorloss', actor_loss)
            #input()
            #actor_loss = Variable(actor_loss , requires_grad = True)
            #critic_loss = Variable(critic_loss , requires_grad = True)
            #grad_fn=<NegBackward>
            #actor_loss = torch.tensor(actor_loss, requires_grad=True)
            #critic_loss = torch.tensor(critic_loss, requires_grad=True)
            
            optimizer.zero_grad()
            loss_average.backward(retain_graph=True)
            optimizer.step()
            
            env =  gameEnv.game( level = 'EASY')
            
            torch.save(ac, str(path)+'ac.pkl')
 
            
            
            #print(critic_loss)
            #print(actor_distance_loss)
            #print(actor_angle_loss)
            #for param in actor_angle.parameters():
            #    print(param)
            #input()
            #for param in actor_distance.parameters():
            #    print(param.grad)
            #input()
        except Exception as e:
            print(str(e))
            input()
            
            env =  gameEnv.game(level = 'EASY')
            print("something wrong happened")
        
        #avg_cum_rewards.append(sum(cum_rewards[-10:-1])/len(cum_rewards[-10:-1]))
        #avg_cum_rewards.append(sum(cum_rewards)/len(cum_rewards))
        all_avg_cum_rewards.append(sum(cum_rewards)/len(cum_rewards))
        
        #plt.plot(list(range(0, len(avg_cum_rewards))),avg_cum_rewards, '-r', label = 'reward average')
        plt.plot(list(range(0, len(all_avg_cum_rewards ))),all_avg_cum_rewards , '-b', label = 'reward average')
        plt.plot(list(range(0, len(cum_rewards))),cum_rewards, '-g', label = 'reward per game')
        plt.savefig('Reward Plot')
        plt.show()
    #print(len(cum_rewards))
    plt.plot(list(range(0, len(cum_rewards))),cum_rewards)
    plt.show()
    #env.close()


if __name__ == '__main__':
    n_iters = 100000# input('number of iteration? ')
    
    main( int(n_iters))

