from torch.autograd import Variable
import random, os, networks, gameEnv, torch
from torchvision import transforms
import torch.optim as optim


def main(numOfEpisodes):
    print('Version 3')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_size = 50
    action_size = 2
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(state_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
            ])
    #if b['rect'].top > winH:
        #baddies.remove(b)
    if os.path.exists('model/actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = networks.Actor(action_size).to(device)
                
    if os.path.exists('model/critic.pkl') :
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = networks.Critic(action_size = 1).to(device)
     
    optimizerA = optim.Adam(actor.parameters(), lr=1e-4)
    optimizerC = optim.Adam(critic.parameters(), lr=1e-4)
    #a = 2

    
    for nop in range(numOfEpisodes):
        if (nop%100) == 0:
            print('episode: ',nop)

        game =  gameEnv.game(actor, critic, transform, level = 'EASY')
        actor_loss, critic_loss = game.play()
        actor_loss = Variable(actor_loss , requires_grad = True)
        critic_loss = Variable(critic_loss , requires_grad = True)

        
        #a = (list(actor.fc.parameters()))
        
        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()

        #b = (list(actor.fc.parameters()))
        #print('a==b? ', bool(a==b))
        #a+=1
        #if a == 3 and False:
            #a = 0
            #print(list(actor.fc[2].parameters()))
        torch.save(actor,'actor.pkl')
        torch.save(critic,'critic.pkl')
        #with open('models/actor.pkl', 'w') as f:
            #torch.save(actor_loss.state_dict(), f)
            #torch.save(actor_loss.state_dict() , f)
            #torch.save(critic_loss.state_dict(), f)

if __name__ == '__main__':
    main(numOfEpisodes = 10000)











'''
!pip install torch, pygame, torchvision
!pip install -q xlrd
!git clone https://github.com/MahmoodabadiHamid/frozenLake-Pytorch
import os
os.chdir('/content/frozenLake-Pytorch')

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"
%run main.py
'''
