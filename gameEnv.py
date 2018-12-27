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
        #self.BARRIERRADIUS = 0.06
        #self.ROBOTRADIUS = 0.15
        #self.W = 2 * self.ROBOTRADIUS
        self.PLAYFIELDCORNERS = (-3.0, -3.0, 3.0, 3.0)
        #self.target = (PLAYFIELDCORNERS[2] + 1.0, 0)
        #self.k = 160 # pixels per metre for graphics
        #self.x = PLAYFIELDCORNERS[0] - 0.5
        #self.y = 0.0
        
        
        self.level = level
        self.actor = actor
        self.critic = critic
        self.transform = transform
        self.playerImage = pygame.image.load('files/large.gif')
        self.baddieImage = pygame.image.load('files/baddie.jpg')
        self.destinyImage = pygame.image.load('files/destiny.png')
        self.playerRect = self.playerImage.get_rect()
        self.winW = 600
        self.winH = 600
        self.bgColor = (0, 0, 0)
        self.FPS = 5000
        self.obstacleMinSiz = 20
        self.obstacleMaxSiz = 40
        self.destinyMinSiz = 60
        self.destinyMaxSiz = 60

        self.obstacleMinSpd = 0
        self.obstacleMaxSpd = 0

        self.obstacleAddRate = 100
        self.playerMoveRate = 10
        pygame.init()
        self.mainClock = pygame.time.Clock()
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
                [460, 592, 30, 30],
                [110, 583, 27, 27],
                [428, 111, 31, 31],
                [207, 281, 32, 32],
                [370, 162, 22, 22],
                [386, 333, 25, 25],
                [256, 171, 29, 29],
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
        state = pygame.surfarray.array3d(pygame.display.get_surface())
        #if (random.randint(0,100) == 2):
            #plt.imshow(state)
            #plt.show()
        state = state.transpose((2, 0, 1))
        state = np.ascontiguousarray(state, dtype=np.float32) / 255
        state = torch.from_numpy(state)
        state = self.transform(state).unsqueeze(0)
        return state
    def calculateClosestObstacleDistance(self,x, y, RRTFLAG):
        closestdist = 100000.0
        # Calculate distance to closest obstacle
        for barrier in self.baddies[:]:
        # Is this a barrier we know about? 
        # For local planning, only if we've seen it. For RRT, always
            #if(barrier[2] == 1 or RRTFLAG == 1):
                dx = barrier[0] - x
                dy = barrier[1] - y
                d = math.sqrt(dx**2 + dy**2)
                dist = d - self.BARRIERRADIUS - self.ROBOTRADIUS
                if (dist < closestdist):
                    closestdist = dist

        return closestdist  
    def findClosestNode(self,x, y, root):
    # find closest node in tree
        closestdist = 1000000
        for node in anytree.PreOrderIter(root):
            #print node.name
            vectortonode = (x - node.name[0], y - node.name[1])
            vectortonodelength = math.sqrt(vectortonode[0] **2 + vectortonode[1] **2)
            if(vectortonodelength < closestdist):
                closestdist = vectortonodelength
                closestnode = node
        return closestnode
    # Implement an RRT starting from robot position
    def RRT(self,x, y):
    #screen.fill(black)
        u0 = self.winW / 2
        v0 = self.winH / 2
        STEP = 2*self.ROBOTRADIUS
        #print "RRT!"
        root = anytree.Node((x, y))
        for i in range(10000):
            randompoint = ((random.uniform(self.PLAYFIELDCORNERS[0], self.PLAYFIELDCORNERS[2] + 1.0), random.uniform(self.PLAYFIELDCORNERS[1], self.PLAYFIELDCORNERS[3])))
            closestnode = findClosestNode(randompoint[0], randompoint[1], root)

            # Now we have closest node, try to create new node
            vectortonode = (randompoint[0] - closestnode.name[0], randompoint[1] - closestnode.name[1])
            vectortonodelength = math.sqrt(vectortonode[0] **2 + vectortonode[1] **2)
            if (vectortonodelength <= STEP):
                newpossiblepoint = randompoint
            else:
                stepvector = (vectortonode[0] * STEP / vectortonodelength, vectortonode[1] * STEP / vectortonodelength)
                newpossiblepoint = (closestnode.name[0] + stepvector[0], closestnode.name[1] + stepvector[1])

            # Is this node valid?
            obdist = calculateClosestObstacleDistance(newpossiblepoint[0], newpossiblepoint[1], 1)
            if (obdist > 0):
                nextnode = anytree.Node((newpossiblepoint[0], newpossiblepoint[1]), parent=closestnode)


            if (i % 1 == 0):
                pygame.draw.circle(screen, (255,100,0), (int(u0 + k * target[0]), int(v0 - k * target[1])), int(k * self.ROBOTRADIUS), 0)


# Draw robot
                u = u0 + k * x
                v = v0 - k * y
                pygame.draw.circle(screen, (255,255,255), (int(u), int(v)), int(k * self.ROBOTRADIUS), 3)
                #gameDisplay = pygame.display.set_mode((1500,1000))
                #gameDisplay.blit(playerImage, ((int(u),int(v))))

                for node in anytree.PreOrderIter(root):
                    if (node.parent != None):
                        pygame.draw.line(screen, (100,100,100), (int(u0 + k * node.name[0]), int(v0 - k * node.name[1])), (int(u0 + k * node.parent.name[0]), int(v0 - k * node.parent.name[1])))
                #drawBarriers(barriers)
                pygame.display.flip()
            #time.sleep(0.1)

            distfromtarget = math.sqrt((newpossiblepoint[0] - target[0])**2 + (newpossiblepoint[1] - target[1])**2)
            if (distfromtarget < 2* self.ROBOTRADIUS):
                break


        # Try making a path
        startnode = findClosestNode(x, y, root)
        targetnode = findClosestNode(target[0], target[1], root)
        #pygame.draw.line(screen, (255,100,100), (int(u0 + k * startnode.name[0]), int(v0 - k * startnode.name[1])), (int(u0 + k * targetnode.parent.name[0]), int(v0 - k * targetnode.parent.name[1])))
        pygame.display.flip()
        w = anytree.Walker()
        (upwardsnode, commonnode, downwardsnodes) = w.walk(startnode, targetnode)
        for i, node in enumerate (downwardsnodes):
            #print "Node", i, "\n"
            #print node, "\n"
            if node != []:
                pygame.draw.circle(screen, (255,0,0), (int(u0 + k * node.name[0]), int(v0 - k * node.name[1])), 5, 0)
                pygame.display.flip()
                time.sleep(0.1)
        return
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
    def playerHasRichDestiny(self):
         for d in self.destiny:
             if self.playerRect.colliderect(d['rect']):
                 return True
         return False

    def compute_returns(self, next_value, rewards, masks, gamma=0.9):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns


    def step(self):
        reward = -100
        done = 0
        epsilon = 0.0001
        self.playerRect.x += math.ceil(math.sin(self.angle) + epsilon * self.playerMoveRate + epsilon)
        self.playerRect.y += math.ceil(math.cos(self.angle) + epsilon * self.playerMoveRate + epsilon)
        
        #self.playerRect.move_ip(math.sin(self.angle) * self.playerMoveRate,0)
        #self.playerRect.move_ip(0, math.cos(self.angle)* self.playerMoveRate)
        
        if (self.playerRect.top > self.winH or self.playerRect.top < 0 or self.playerRect.left > self.winW or self.playerRect.left < 0):
            reward = -1000
            done = 1
        if self.playerHasHitBaddie():
            reward = -1000
            done = 1
        if self.playerHasRichDestiny():
             reward = +1000
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
    main.main(numOfEpisodes = 100000)


