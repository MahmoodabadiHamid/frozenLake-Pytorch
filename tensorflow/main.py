from torch.autograd import Variable
import random, os, gameEnv, torch
from torchvision import transforms
import torch.optim as optim
import pytorch_networks as networks


def main(episodes=100, gamma=0.95, display=False, lamb=1e-5, policy_lr=0.001, value_lr=0.1):
    print('Version 3')
    
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
        actor = networks.PolicyEstimator(env, lamb=lamb, learning_rate=policy_lr)
                
    if os.path.exists('model/critic.pkl') :
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = networks.Critic(action_size = 1)#.to(device)

    tf.reset_default_graph()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    stats = []
    optimizerA = optim.Adam(actor.parameters(), lr=1e-4)
    optimizerC = optim.Adam(critic.parameters(), lr=1e-4)
    

    
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

        torch.save(actor,'actor.pkl')
        torch.save(critic,'critic.pkl')


if __name__ == '__main__':
    policy_lr, value_lr, lamb, gamma = [0.0001, 0.0046415888336127773, 2.782559402207126e-05, 0.999]
    main(episodes=1000, gamma=gamma, display=False, lamb=lamb, policy_lr=policy_lr, value_lr=value_lr)











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
