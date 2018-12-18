import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import gym, os
from itertools import count
import torch
from multiprocessing import Process
from threading import Thread
import pygame, random, sys
from pygame.locals import *
import time


class game():
    
    def __init__(self, actor, critic, transform):
        self.actor = actor
        self.critic = critic
        self.transform = transform
        self.playerImage = pygame.image.load('large.gif')
        self.baddieImage = pygame.image.load('baddie.jpg')
        self.playerRect = self.playerImage.get_rect()
        self.winW = 600
        self.winH = 600
        self.bgColor = (0, 0, 0)
        self.FPS = 40
        self.obstacleMinSiz = 50
        self.obstacleMaxSiz = 80

        self.obstacleMinSpd = 0
        self.obstacleMaxSpd = 2

        self.obstacleAddRate = 10
        self.playerMoveRate = 10
        self.moveLeft = self.moveRight = self.moveUp = self.moveDown = False
        pygame.init()
        self.mainClock = pygame.time.Clock()
        self.windowSurface = pygame.display.set_mode((self.winW, self.winH))
        self.baddies = []
        self.obstacleAddCounter = 0
        self.log_probs = []

    def terminate():
        pygame.quit()
        sys.exit()

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


    def compute_returns(next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns


    def step(self):
        reward = 0
        done = 0
        
        #if (Action == 5):
            #Fire key
        import math
        
        #we should change -> self.playerMoveRate
        self.playerRect.move_ip(math.sin(self.angle) * self.playerMoveRate,0)
        self.playerRect.move_ip(0, math.cos(self.angle)* self.playerMoveRate)

        '''
        if self.Action == 4 and self.playerRect.left > 0:
            self.playerRect.move_ip(-1 * self.playerMoveRate, 0)
            
        if self.Action == 7 and self.playerRect.top > 0:
           self.playerRect.move_ip(0, -1 * self.playerMoveRate)
           self.playerRect.move_ip(-1 * self.playerMoveRate, 0)
           

        if self.Action == 8 and self.playerRect.top > 0:
           self.playerRect.move_ip(0, -1 * self.playerMoveRate)
           
        if self.Action == 9 and self.playerRect.top > 0:
           self.playerRect.move_ip(0, -1 * self.playerMoveRate)
           self.playerRect.move_ip(self.playerMoveRate, 0)

        if self.Action == 6 and self.playerRect.right < self.winW:
           self.playerRect.move_ip(self.playerMoveRate, 0)
           
        if self.Action == 3 and self.playerRect.bottom < self.winH:
           self.playerRect.move_ip(0, self.playerMoveRate)
           self.playerRect.move_ip(self.playerMoveRate, 0)
           
        if self.Action == 2 and self.playerRect.bottom < self.winH:
           self.playerRect.move_ip(0, self.playerMoveRate)

        if self.Action == 1 and self.playerRect.bottom < self.winH:
           self.playerRect.move_ip(0, self.playerMoveRate)
           self.playerRect.move_ip(-1 * self.playerMoveRate, 0)   
        done = 0
        next_state = self.playerRect
        return next_state, done
        '''
    def play(self):
        for _ in range (10):
            obstacleAddCounter = 0
            self.baddiesize = random.randint(self.obstacleMinSiz, self.obstacleMaxSiz)
            newBaddie = {'rect': pygame.Rect(random.randint(0, self.winW-self.baddiesize), 0 - self.baddiesize, self.baddiesize, self.baddiesize),
                                'speed': random.randint(self.obstacleMinSpd, self.obstacleMaxSpd),
                                'surface':pygame.transform.scale(self.baddieImage, (self.baddiesize, self.baddiesize)),
                                }
            self.baddies.append(newBaddie)
        for b in self.baddies[:]:
            b['rect'].top = random.randint(0, self.winW)
            b['rect'].left =random.randint(0, self.winH)


        while True:
            self.playerRect.topleft = (self.winW / 2, self.winH - 50)
            moveLeft = moveRight = moveUp = moveDown = False
            i = 0
            while True:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        self.terminate()
                
                state = pygame.surfarray.array3d(pygame.display.get_surface())
                i +=1
                if i%50 == 0:
                    i = 0
                    print(self.angle)
                    print(self.playerMoveRate)
                    #state = state.transpose((2, 0, 1))
                    #print(state.shape)
                    #plt.imshow(state)
                    #plt.show()

                state = state.transpose((2, 0, 1))
                state = np.ascontiguousarray(state, dtype=np.float32) / 255
                state = torch.from_numpy(state)
                state = self.transform(state).unsqueeze(0)
                
                 
                dist, value = self.actor(state), self.critic(state)
                #print(dist)
                #self.Action = int(dist.sample())
                self.Action = dist

                self.angle = self.Action[0,0]  + random.randint(0,360)
                self.playerMoveRate = self.Action[0,1] 
                
                self.step()
                if self.playerHasHitBaddie():
                    reward = -1
                    break
            
                    
                #log_prob = dist.log_prob(self.Action).unsqueeze(0)
                #self.log_probs.append(log_prob)
                #rewards.append(torch.tensor([reward], dtype=torch.float))
                #masks.append(torch.tensor([1-done], dtype=torch.float))
                next_state = state
                next_state = torch.FloatTensor(next_state)
                next_value = self.critic(next_state)
                #returns = compute_returns(next_value, rewards, masks)
                
                
                
                for b in self.baddies:
                     b['rect'].move_ip(0, b['speed'])
           
                self.windowSurface.fill(self.bgColor)        
                self.windowSurface.blit(self.playerImage, self.playerRect)
                for b in self.baddies:
                    self.windowSurface.blit(b['surface'], b['rect'])
                pygame.display.update()
                
                
            self.mainClock.tick(self.FPS)
        self.terminate()
        self.actor_loss = -(advantage.detach()).mean()
        self.critic_loss = advantage.pow(2).mean()
        return  self.actor_loss, self.critic_loss
        
            
