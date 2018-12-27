import main
#import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn as nn
import torch
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.l1 = nn.Linear(4608, 20 )
        self.l2 = nn.Linear(20, 40)
        self.l3 = nn.Linear(40, 30)
        self.l4 = nn.Linear(30, 2)
        
     
    def forward(self, state):
        out = self.layer1(state)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        #out = F.relu(self.fc(out))
        out = torch.tanh(self.l1(out))
        out = torch.relu(self.l2(out))
        out = torch.sigmoid(self.l3(out))
        out = (self.l4(out))        
        return out[0]
    
        #distribution = Categorical(F.softmax(out, dim=-1))
        #return distribution
 



class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc = nn.Sequential(
            nn.Linear(4608, 20 ),
            nn.Linear(20, 20 * 2),
            nn.Linear(20 * 2, 300),
            nn.Linear(300, 1))

        self.l1 = nn.Linear(4608, 20 )
        self.l2 = nn.Linear(20, 40)
        self.l3 = nn.Linear(40, 30)
        self.l4 = nn.Linear(30, 1)

        
    def forward(self, state):
        
        value = self.layer1(state)
        value = self.layer2(value)
        value = value.view(value.size(0), -1)
        value = torch.tanh(self.l1(value))
        value = torch.relu(self.l2(value))
        value = torch.sigmoid(self.l3(value))
        value = (self.l4(value))
        #value = self.fc(value)


        
        return value



if __name__ == '__main__':
    main.main(numOfEpisodes = 100000)
