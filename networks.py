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

        self.layer3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc =  nn.Sequential(
                    nn.Linear(9, 20),
                    nn.Tanh(),
                    nn.Linear(20, 40),
                    nn.ReLU(),
                    nn.Linear(40, 20),
                    nn.Sigmoid(),
                    nn.Linear(20, action_size))
        
     
    def forward(self, state):
        out = self.layer1(state)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out.requires_grad_(True)
        #print(out)
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
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc =  nn.Sequential(
                    nn.Linear(9, 20),
                    nn.Tanh(),
                    nn.Linear(20, 40),
                    nn.ReLU(),
                    nn.Linear(40, 30),
                    nn.Sigmoid(),
                    nn.Linear(30, action_size))

        
    def forward(self, state):

        value = self.layer1(state)
        value = self.layer2(value)
        value = self.layer3(value)
        value = self.layer4(value)
        value = value.view(value.size(0), -1)
        value = self.fc(value)
        return value



if __name__ == '__main__':
    main.main(numOfEpisodes = 100000)
