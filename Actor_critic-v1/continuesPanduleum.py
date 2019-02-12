# Reference - Deepmind A3C's paper, https://arxiv.org/pdf/1602.01783.pdf
# See section 9 - Continuous Action Control Using the MuJoCo Physics Simulator
#
# Code based on https://github.com/pfre00/a3c

import argparse
import torch
import torch.multiprocessing as mp
import gym
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import gym
from datetime import datetime, timedelta

#from model import ActorCritic
#from train import train
#from async_rmsprop import AsyncRMSprop
#from async_adam import AsyncAdam

# Training settings
# parser = argparse.ArgumentParser(description='A3C')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
#                     help='learning rate (default: 0.0001)')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
#                     help='discount factor for rewards (default: 0.99)')
# parser.add_argument('--tau', type=float, default=0.9, metavar='T',
#                     help='parameter for GAE (default: 0.9)')
# parser.add_argument('--num-processes', type=int, default=4, metavar='N',
#                     help='how many training processes to use (default: 4)')
# parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
#                     help='number of forward steps in A3C (default: 20)')
# parser.add_argument('--env-name', default='Breakout-v0', metavar='ENV',
#                     help='environment to train on (default: Breakout-v0)')
# parser.add_argument('--render', default=False, action='store_true',
#                     help='render the environment')
import easydict
args = easydict.EasyDict({
#     "batch_size": 100,
#     "train_steps": 1000,
    "lr":0.0001,
    "gamma":0.99,
    "tau":0.9,
    "seed":1.00,
    "num-processes":4,
    "num_steps":20,
    "render":False,
    "num-steps":20,
    "max-episode-length":10000,
    "env-name":'Breakout-v0',
#     "shared-optimizer":True,
#     "amsgrad":True,
#     "gpu_ids":-1,
#     "stack-frames":1,
#     "model":'MLP',
#     "log-dir":'logs/',
#     "save-model-dir":'trained_models/',
#     "load-model-dir":'trained_models/',
#     "optimizer":'Adam',
#     "save-max":True,
#     "load":False
})
#if __name__ == '__main__':

#args = parser.parse_args()

#torch.manual_seed(args.seed)
torch.set_num_threads(1)

lstm_out = 256
lstm_in = lstm_out
enc_in = 3 # for pendulum
enc_hidden = 200
enc_out = lstm_in

class ActorCritic(nn.Module):
    def __init__(self , enc_in ):
        super(ActorCritic, self).__init__( )
        self.fc_enc_in  = nn.Linear(enc_in,enc_hidden) # enc_input_layer
        self.fc_enc_out  = nn.Linear(enc_hidden,enc_out) # enc_output_layer
        self.lstm = nn.LSTMCell(lstm_in, lstm_out)
        self.actor_mu = nn.Linear(lstm_out, 1)
        self.actor_sigma = nn.Linear(lstm_out, 1)
        self.critic_linear = nn.Linear(lstm_out, 1)   
        self.train()  
    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = F.relu(self.fc_enc_in(x))
        x = self.fc_enc_out(x)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_mu(x), self.actor_sigma(x), (hx, cx)

env = gym.envs.make("Pendulum-v0")

#global_model = ActorCritic(env.action_space.n)
#global_model.share_memory()

local_model = ActorCritic( enc_in )

#optimizer = AsyncAdam(global_model.parameters(), local_model.parameters(), lr=args.lr)

optimizer = optim.Adam( local_model.parameters(), lr=args.lr )

state = env.reset()
done = True

state = torch.from_numpy(state).float().unsqueeze(0) 
reward_sum = 0
running_reward = 0
episodes = 0

while True:
    #
    if done:
        cx = Variable(torch.zeros(1, 256))
        hx = Variable(torch.zeros(1, 256))
    else:
        cx = Variable(cx.data)
        hx = Variable(hx.data)
    #
    values = []
    log_probs = []
    rewards = []
    entropies = []
    #
    for step in range(args.num_steps):
        #
        value, mu, sigma, (hx, cx) = local_model((Variable(state), (hx, cx)))
        #
        Softplus=nn.Softplus()     
        sigma = Softplus(sigma + 1e-5) # constain to sensible values
        normal_dist = torch.normal(mu, sigma) # http://pytorch.org/docs/torch.html?highlight=normal#torch.normal
        #
        prob = normal_dist
        #
        ##-------------------------------------------------------------
        # TODO tidy this up
        # Calculate the Gaussian neg log-likelihood, log(1/sqrt(2sigma^2pi)) - (x - mu)^2/(2*sigma^2)
        # See - https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood
        #
        log_prob = torch.log(torch.pow( torch.sqrt(2. * sigma * np.pi) , -1)) - (normal_dist - mu)*(normal_dist - mu)*torch.pow((2. * sigma), -1)
        #
        entropy = 0.5 * (torch.log(2. * np.pi * sigma ) + 1.)
        ##--------------------------------------------------------------
        #
        action = Variable( prob.data )
        #
        #action=[0,]
        state, reward, done, _ = env.step([action.data[0][0]])
        #
        state = torch.from_numpy(state).float().unsqueeze(0) 
        #
        values.append(value)
        log_probs.append(log_prob)
        rewards.append(max(min(reward, 1), -1))
        entropies.append(entropy)
        #
        reward_sum += reward
        #
        t_start=0
        rank=0
        if done:
            t_now = datetime.now()
            t_elapsed = t_now# - t_start
            episodes += 1
            running_reward = running_reward * 0.99 + reward_sum * 0.01
            unbiased_running_reward = running_reward / (1 - pow(0.99, episodes))
            if rank == 0:
                print("{}\t{}\t{}\t{:.2f}\t{}".format(
                    t_now, t_elapsed, episodes, unbiased_running_reward, reward_sum))
            reward_sum = 0
            state = env.reset()
            state = torch.from_numpy(state).float().unsqueeze(0) 
            break
    #
    R = torch.zeros(1, 1)
    if not done:
        value, _ , _ , _ = local_model( (Variable(state), (hx, cx)) )
        R = value.data
        #
    R = Variable(R)
    values.append(R)
    gae = torch.zeros(1, 1)
    policy_loss = 0
    value_loss = 0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + args.gamma * R
        advantage = R - values[t]
        value_loss = value_loss + advantage.pow(2)
        #
        # Generalized Advantage Estimataion
        delta_t = rewards[t] + args.gamma * values[t + 1].data - values[t].data
        gae = gae * args.gamma * args.tau + delta_t
        #
        policy_loss = policy_loss - log_probs[t] * Variable(gae) - 0.01 * entropies[t]
    #   
    optimizer.zero_grad()
    (policy_loss + 0.5 * value_loss).backward()
    optimizer.step()
    for param in local_model.parameters:
        print(param)
    input()