import random, os, networks, gameEnv, pygame
from torchvision import transforms
import torch.optim as optim


def main(numOfEpisodes):
    state_size = 50
    action_size = 2
    if os.path.exists('model/actor.pkl') and False :
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = networks.Actor(state_size, action_size)
    if os.path.exists('model/critic.pkl') and False:
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = networks.Critic(state_size, action_size)
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(state_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
            ])
        #if b['rect'].top > winH:
             #baddies.remove(b)
    for _ in range(10):        
        game =  gameEnv.game(actor, critic, transform)
        actor_loss, critic_loss = game.play()
        
        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()

if __name__ == '__main__':
    main(numOfEpisodes = 10)
