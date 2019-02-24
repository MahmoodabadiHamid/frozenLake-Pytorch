from torchvision import transforms
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
    
    def __init__(self, level):
        pygame.init()
        #self.BARRIERRADIUS = 0.06
        #self.ROBOTRADIUS = 0.15
        #self.W = 2 * self.ROBOTRADIUS
        self.PLAYFIELDCORNERS = (-3.0, -3.0, 3.0, 3.0)
        #self.target = (PLAYFIELDCORNERS[2] + 1.0, 0)
        #self.k = 160 # pixels per metre for graphics
        #self.x = PLAYFIELDCORNERS[0] - 0.5
        #self.y = 0.0
        
        
        
        state_size = 25
        self.level = level
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(state_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
                ])
        self.playerImage = pygame.transform.scale(pygame.image.load('files/large.gif'),(20,20))                 
        self.baddieImage = pygame.image.load('files/baddie.jpg')
        self.destinyImage = pygame.image.load('files/destiny.png')
        self.playerRect = self.playerImage.get_rect()
        self.playerRect.x = random.randint(0,550)
        self.playerRect.y = random.randint(0,550)
        #print(self.playerRect)
        #input()
        #print('player ', self.playerRect)
        self.winW = 600
        self.winH = 600
        self.bgColor = (0, 0, 0)
        self.FPS = 80
        self.obstacleMinSiz = 20
        self.obstacleMaxSiz = 40
        self.destinyMinSiz = 60
        self.destinyMaxSiz = 60

        self.obstacleMinSpd = 0
        self.obstacleMaxSpd = 2

        self.obstacleAddRate = 100
        self.playerMoveRate = 10
        
        self.mainClock = pygame.time.Clock()
        self.mainClock.tick(self.FPS)
        self.windowSurface = pygame.display.set_mode((self.winW, self.winH))
        self.baddies = []
        self.destiny = []
        self.obstacleAddCounter = 10
        self.log_probs = []
        if (self.level == 'EASY'):
            
            self.rects = [
                [437, 396, 31, 31],
                [196, 449, 28, 28],
                [249, 363, 21, 21],
                [400, 482, 50, 50],
                [500, 402, 50, 50],
                
                [110, 583, 27, 27],
                [428, 111, 31, 31],
                [207, 281, 32, 32],
                [370, 162, 22, 22],
                [386, 333, 25, 25],
                [256, 171, 29, 29],
                [400, 400, 100, 100],
                ]
            
            
            for i in range (self.obstacleAddCounter):
                #self.baddiesize = random.randint(self.obstacleMinSiz, self.obstacleMaxSiz)
                
                newBaddie = {'rect'   : pygame.Rect(self.rects[i]),
                             'speed'  : 0,
                             'surface': pygame.transform.scale(self.baddieImage, (self.rects[i][2], self.rects[i][3])),
                                    }
                self.baddies.append(newBaddie)
                
            for i in range (1):
                #self.destinySize = random.randint(self.destinyMinSiz, self.destinyMaxSiz)
                
                destiny = {  'rect'   : pygame.Rect(10,10,40,40),
                             'speed'  : 0,
                             'surface': pygame.transform.scale(self.destinyImage, (self.rects[i][2], self.rects[i][3])),
                                    }
                self.destiny.append(destiny)

        elif(self.level == 'MEDIUM'):
            
            for _ in range (self.obstacleAddCounter):
                self.baddiesize = random.randint(self.obstacleMinSiz, self.obstacleMaxSiz)
                
                newBaddie = {'rect'   : pygame.Rect(random.randint(0, self.winW-self.baddiesize), 0 - self.baddiesize, self.baddiesize, self.baddiesize),
                             'speed'  : random.randint(self.obstacleMinSpd, self.obstacleMaxSpd),
                             'surface': pygame.transform.scale(self.baddieImage, (self.baddiesize, self.baddiesize)),
                                    }
                self.baddies.append(newBaddie)
                
            for i in range (1):
                self.destinySize = random.randint(self.destinyMinSiz, self.destinyMaxSiz)
                
                destiny = {'rect'   : pygame.Rect(0,0,40,40),
                             'speed'  : 0,
                             'surface': pygame.transform.scale(self.destinyImage, (self.destinySize, self.destinySize)),
                                    }
                self.destiny.append(destiny)
                
            for b in self.baddies[:]:
                while True:
                    b['rect'].top = random.randint(0, self.winW)
                    b['rect'].left =random.randint(0, self.winH)
                    
                    if(not self.playerHasHitBaddie()):
                        break
    
        elif(self.level == 'HARD'):
            
            self.obstacleMinSpd = 2
            self.obstacleMaxSpd = 5
            for _ in range (self.obstacleAddCounter):
                self.baddiesize = random.randint(self.obstacleMinSiz, self.obstacleMaxSiz)
                
                newBaddie = {'rect'   : pygame.Rect(random.randint(0, self.winW-self.baddiesize), 0 - self.baddiesize, self.baddiesize, self.baddiesize),
                             'speed'  : random.randint(self.obstacleMinSpd, self.obstacleMaxSpd),
                             'surface': pygame.transform.scale(self.baddieImage, (self.baddiesize, self.baddiesize)),
                                    }
                self.baddies.append(newBaddie)
            for i in range (1):
                self.destinySize = random.randint(self.destinyMinSiz, self.destinyMaxSiz)
                
                destiny = {'rect'   : pygame.Rect(0,0,40,40),
                             'speed'  : 0,
                             'surface': pygame.transform.scale(self.destinyImage, (self.destinySize, self.destinySize)),
                                    }
                self.destiny.append(destiny)
            for b in self.baddies[:]:
                while False:
                    b['rect'].top = random.randint(0, self.winW)
                    b['rect'].left =random.randint(0, self.winH)
                    if(not self.playerHasHitBaddie()):
                        break
        

    def getState(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = pygame.surfarray.array3d(pygame.display.get_surface())
       
        #if (random.randint(0,100) == 2):
            #plt.imshow(state)
            #plt.show()
        state = state.transpose((2, 0, 1))
        
        state = np.ascontiguousarray(state, dtype=np.float32) 

        state = torch.from_numpy(state)
        state = self.transform(state).unsqueeze(0).to(device)
        state=state*10000
        #plt.imshow(state)
        #plt.show()
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

    def nodeHasHitBaddie(self, rect):
        for b in self.baddies:
            if rect.colliderect(b['rect']):
                return True
        return False

    def playerHasRichDestiny(self):
         for d in self.destiny:
             if self.playerRect.colliderect(d['rect']):
                 return True
         return False
    def nodeHasRichDestiny(self, rect):
         for d in self.destiny:
             if rect.colliderect(d['rect']):
                 return True
         return False

    def compute_returns(self, next_value, rewards, masks, gamma=0.9):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns


    def step(self, distance, angle):
        reward =(-distance/10)
        done = 0
        epsilon = 0.1
        self.angle = (((angle)))
        self.playerMoveRate =  ((distance))
        
        self.playerRect.x += (math.sin(self.angle) ) * (self.playerMoveRate )
        self.playerRect.y += (math.cos(self.angle) ) * (self.playerMoveRate )
        
        if (self.playerRect.top > self.winH or self.playerRect.top < 0 or self.playerRect.left > self.winW or self.playerRect.left < 0):
            reward = float(-distance-1000)
            print(distance)
            done = 1

        if self.playerHasHitBaddie():
            reward = (-distance-1000)
            done = 1
        if self.playerHasRichDestiny():
             reward = (distance+1000)
             done = 1
        self.updateDisplay() 
        n_s = self.getState()
        #distance_cum += distance
        return n_s, reward, done, 'info'



    def updateDisplay(self):
        self.bgColor = (0, 0, 0)
        #for event in pygame.event.get():
        #    if event.type == QUIT:
        #        self.terminate()
        for b in self.baddies:
           b['rect'].move_ip(0, b['speed'])
        
        self.windowSurface.fill(self.bgColor)
        self.windowSurface.blit(self.playerImage, self.playerRect)
        
        for b in self.baddies:
            self.windowSurface.blit(b['surface'], b['rect'])
            posX = int(b['rect'].x + int(b['rect'].w/2))
            posY = int(b['rect'].y + int(b['rect'].w/2))
            r = int(math.sqrt((b['rect'].x - posX)**2 + (b['rect'].y - posY)**2))
            pygame.draw.circle(self.windowSurface, (0,0,0,0), (posX,posY), r,1)
            
        for d in self.destiny:
            self.windowSurface.blit(d['surface'], d['rect'])
            posX = int(d['rect'].x + int(d['rect'].w/2))
            posY = int(d['rect'].y + int(d['rect'].h/2))
            r = int(math.sqrt((d['rect'].x - posX)**2 + (d['rect'].y - posY)**2))
            pygame.draw.circle(self.windowSurface, (0,0,0,0), (posX,posY), r,1)

        pygame.draw.line(self.windowSurface, (255,0,0,0), (0, 0), (self.winW, 0))
        pygame.draw.line(self.windowSurface, (255,0,0,0), (0, 0), (0, self.winH))
        pygame.draw.line(self.windowSurface, (255,0,0,0), (0, self.winH-1), (self.winW-1, self.winH-1))
        pygame.draw.line(self.windowSurface, (255,0,0,0), (self.winW-1, 0), (self.winW-1, self.winH-1))
        
        pygame.display.update()


        
            
if __name__ == '__main__':
    n_iters = 100000# input('number of iteration? ')
    
    import cart_pole_PT_AC
    cart_pole_PT_AC.main(n_iters = 10000)