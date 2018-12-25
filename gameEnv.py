import main
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
#from itertools import count
import torch
import pygame, random, sys
from pygame.locals import *
import time
import math

class game():
    
    def __init__(self, actor, critic, transform, level):
        self.level = level
        self.actor = actor
        self.critic = critic
        self.transform = transform
        self.playerImage = pygame.image.load('large.gif')
        self.baddieImage = pygame.image.load('baddie.jpg')
        self.playerRect = self.playerImage.get_rect()
        self.winW = 600
        self.winH = 600
        self.bgColor = (0, 0, 0)
        self.FPS = 200
        self.obstacleMinSiz = 20
        self.obstacleMaxSiz = 40

        self.obstacleMinSpd = 0
        self.obstacleMaxSpd = 0

        self.obstacleAddRate = 100
        self.playerMoveRate = 10
        #self.moveLeft = self.moveRight = self.moveUp = self.moveDown = False
        pygame.init()
        self.mainClock = pygame.time.Clock()
        self.windowSurface = pygame.display.set_mode((self.winW, self.winH))
        self.baddies = []
        self.obstacleAddCounter = 10
        self.log_probs = []
        if (self.level == 'EASY'):
            print('Easy level')
            self.rects = [
                [437, 396, 31, 31],
                [196, 449, 28, 28],
                [249, 363, 21, 21],
                [460, 592, 30, 30],
                [110, 583, 27, 27],
                [428, 111, 31, 31],
                [207, 281, 32, 32],
                [370, 162, 22, 22],
                [386, 333, 25, 25],
                [256, 171, 29, 29],
                ]
            for _ in range (self.obstacleAddCounter):
                self.baddiesize = random.randint(self.obstacleMinSiz, self.obstacleMaxSiz)
                
                newBaddie = {'rect'   : pygame.Rect(random.randint(0, self.winW-self.baddiesize), 0 - self.baddiesize, self.baddiesize, self.baddiesize),
                             'speed'  : 0,
                             'surface': pygame.transform.scale(self.baddieImage, (self.baddiesize, self.baddiesize)),
                                    }
                self.baddies.append(newBaddie)
            for b in self.baddies[:]:
                while True:
                    b['rect'].top = random.randint(0, self.winW)
                    b['rect'].left =random.randint(0, self.winH)
                    print(b['rect'])
                    if(not self.playerHasHitBaddie()):
                        break
            input()
            
        elif(self.level == 'MEDIUM'):
            print('Medium level')
            for _ in range (self.obstacleAddCounter):
                self.baddiesize = random.randint(self.obstacleMinSiz, self.obstacleMaxSiz)
                
                newBaddie = {'rect'   : pygame.Rect(random.randint(0, self.winW-self.baddiesize), 0 - self.baddiesize, self.baddiesize, self.baddiesize),
                             'speed'  : random.randint(self.obstacleMinSpd, self.obstacleMaxSpd),
                             'surface': pygame.transform.scale(self.baddieImage, (self.baddiesize, self.baddiesize)),
                                    }
                self.baddies.append(newBaddie)
            for b in self.baddies[:]:
                while True:
                    b['rect'].top = random.randint(0, self.winW)
                    b['rect'].left =random.randint(0, self.winH)
                    input(b['rect'])
                    if(not self.playerHasHitBaddie()):
                        break
    
        elif(self.level == 'HARD'):
            print('Hard level')
            self.obstacleMinSpd = 2
            self.obstacleMaxSpd = 5
            for _ in range (self.obstacleAddCounter):
                self.baddiesize = random.randint(self.obstacleMinSiz, self.obstacleMaxSiz)
                
                newBaddie = {'rect'   : pygame.Rect(random.randint(0, self.winW-self.baddiesize), 0 - self.baddiesize, self.baddiesize, self.baddiesize),
                             'speed'  : random.randint(self.obstacleMinSpd, self.obstacleMaxSpd),
                             'surface': pygame.transform.scale(self.baddieImage, (self.baddiesize, self.baddiesize)),
                                    }
                self.baddies.append(newBaddie)
            for b in self.baddies[:]:
                while True:
                    b['rect'].top = random.randint(0, self.winW)
                    b['rect'].left =random.randint(0, self.winH)
                    if(not self.playerHasHitBaddie()):
                        break
        

    def getState(self):
        state = pygame.surfarray.array3d(pygame.display.get_surface())
        #if (random.randint(0,100) == 2):
            #plt.imshow(state)
            #plt.show()
        state = state.transpose((2, 0, 1))
        state = np.ascontiguousarray(state, dtype=np.float32) / 255
        state = torch.from_numpy(state)
        state = self.transform(state).unsqueeze(0)
        return state
    
    def terminate(self):
        pygame.quit()
        #sys.exit()

    def waitForPlayerToPressKey():
        while True:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE: 
                        self.terminate()
                    return

    def playerHasHitBaddie(self):
        for b in self.baddies:
            if self.playerRect.colliderect(b['rect']):
                return True
        return False


    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns


    def step(self):
        reward = 0
        done = 0
        epsilon = 0.0001
        self.playerRect.x += math.ceil(math.sin(self.angle) + epsilon * self.playerMoveRate + epsilon)
        self.playerRect.y += math.ceil(math.cos(self.angle) + epsilon * self.playerMoveRate + epsilon)
        
        #self.playerRect.move_ip(math.sin(self.angle) * self.playerMoveRate,0)
        #self.playerRect.move_ip(0, math.cos(self.angle)* self.playerMoveRate)
        
        if (self.playerRect.top > self.winH or self.playerRect.top < 0 or self.playerRect.left > self.winW or self.playerRect.left < 0):
            reward = -1
            done = 1
        if self.playerHasHitBaddie():
            reward = -1
            done = 1
        return done, reward

    def updateDisplay(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.terminate()
        for b in self.baddies:
           b['rect'].move_ip(0, b['speed'])
        self.windowSurface.fill(self.bgColor)        
        self.windowSurface.blit(self.playerImage, self.playerRect)
        for b in self.baddies:
           self.windowSurface.blit(b['surface'], b['rect'])
        pygame.display.update()

        
    def play(self):

        self.values = []
        rewards = []
        masks = []
        reward = 0
        done = 0
        self.playerRect.topleft = (self.winW / 2, self.winH - 50)
        self.updateDisplay()
        state = self.getState()
        stepCounter = 0
        while True:
            stepCounter += 1    
            action = self.actor(state)

            self.angle, self.playerMoveRate = math.ceil(float(action[0])), math.ceil(float(action[1])) # action[0] -> distance; action[1] -> angle
            reward = self.step()
            value = self.critic(state)
            #log_prob = dist.log_prob(self.Action).unsqueeze(0)
            #self.log_probs.append(log_prob)

            state = self.getState()
            
            self.next_state = torch.FloatTensor(state)
            self.next_value = self.critic(self.next_state)
                
            self.values.append(self.next_value)
            rewards.append(torch.tensor([reward], dtype=torch.float))
            masks.append(torch.tensor([1-done], dtype=torch.float))

            if (self.playerHasHitBaddie()       or
              self.playerRect.top > self.winH   or
              self.playerRect.top < 0           or
              self.playerRect.left > self.winW  or
              self.playerRect.left < 0):
                break
            self.updateDisplay()
                              
            self.mainClock.tick(self.FPS)
        self.terminate()
        
        print('stepCounter', stepCounter)
        returns = self.compute_returns(self.next_value, rewards, masks)

        returns = torch.cat(returns).detach()
        self.values = torch.cat(self.values)

        advantage = returns - self.values
        
        self.actor_loss = -(advantage.detach()).mean()
        self.critic_loss = advantage.pow(2).mean()
        return  self.actor_loss, self.critic_loss
        
            
if __name__ == '__main__':
    main.main(numOfEpisodes = 10000)


