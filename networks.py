import main
#import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self, action_size):
        super(Actor, self).__init__()
        self.action_size = action_size

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc =  nn.Sequential(
                    nn.Linear(144, 20),
                    nn.Tanh(),
                    nn.Linear(20, 40),
                    nn.ReLU(),
                    nn.Linear(40, 30),
                    nn.Sigmoid(),
                    nn.Linear(30, action_size))
        
     
    def forward(self, state):
        out = self.layer1(state)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out[0]




class Critic(nn.Module):
    def __init__(self, action_size):
        super(Critic, self).__init__()
        self.action_size = action_size
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc =  nn.Sequential(
                    nn.Linear(144, 20 ),
                    nn.Tanh(),
                    nn.Linear(20, 40),
                    nn.ReLU(),
                    nn.Linear(40, 30),
                    nn.Sigmoid(),
                    nn.Linear(30, action_size))

        
    def forward(self, state):

        value = self.layer1(state)
        value = self.layer2(value)
        value = value.view(value.size(0), -1)
        value = self.fc(value)
        return value



if __name__ == '__main__':
    main.main(numOfEpisodes = 100000)
