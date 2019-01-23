import torch.nn as nn
import torch.optim as optim
import pygame, os, math, time, random, numpy, torch
from pygame.locals import *
import anytree
import gameEnv




class RRT():
    def __init__(self, game, actor_distance, actor_angle, convolution):
        #pygame.init()
        #game = gameEnv.game('actor_distance', 'actor_angle', 'critic', level='EASY')
        #self.WIDTH = 
        #self.HEIGHT = 
        self.game = game
        self.actor_distance = actor_distance
        self.actor_angle = actor_angle
        self.convolution = convolution
        # The region we will fill with obstacles
        self.PLAYFIELDCORNERS = (0, 0, self.game.winW, self.game.winH)

        # Barrier locations
        self.tmpBarriers = self.game.baddies
        self.barriers = []
        # barrier contents are (bx, by, visibilitymask)
        for b in self.tmpBarriers:
            #print(b['rect'])
            #print(b['rect'].x)
            (bx, by) = (b['rect'].x,b['rect'].y)#(random.uniform(PLAYFIELDCORNERS[0], PLAYFIELDCORNERS[2]), random.uniform(PLAYFIELDCORNERS[1], PLAYFIELDCORNERS[3]))
            self.barrier = [bx, by, 0]
            self.barriers.append(self.barrier)
            
        #barrier = [0, 2.5, 0]
        #barriers.append(barrier)

        self.BARRIERRADIUS = 10 

        self.ROBOTRADIUS = 0.15
        self.W = 2 * self.ROBOTRADIUS
        
        #print(self.game.destiny[0]['rect'])
        self.target = (self.game.destiny[0]['rect'].x, self.game.destiny[0]['rect'].y)

        # Array for path choices to draw
        self.pathstodraw = []

        self.x = self.game.playerRect.x
        self.y = self.game.playerRect.y
        self.firstX = self.x
        self.firstY = self.y

        self.size = [self.game.winW, self.game.winH]
        self.screen = pygame.display.set_mode(self.size)
        self.black = (20,20,40)

        # This makes the normal mouse pointer invisible in graphics window
        #pygame.mouse.set_visible(0)




    def calculateClosestObstacleDistance(self, x, y, RRTFLAG):
        closestdist = 100000
        # Calculate distance to closest obstacle
        for barrier in self.barriers:
            # Is this a barrier we know about? 
            # For local planning, only if we've seen it. For RRT, always
            if True:#(barrier[2] == 1 or RRTFLAG == 1):
                
                dx = barrier[0] - x
                dy = barrier[1] - y
                d = math.sqrt(dx**2 + dy**2)
                dist = d - self.BARRIERRADIUS - self.ROBOTRADIUS
                if (dist < closestdist):
                    closestdist = dist

        return closestdist



    def findClosestNode(self, x, y, root):
        # find closest node in tree
        closestdist = 100000
        for node in anytree.PreOrderIter(root):
            #print node.name
            vectortonode = (x - node.name[0], y - node.name[1])
            vectortonodelength = math.sqrt(vectortonode[0] **2 + vectortonode[1] **2)
            #print(vectortonodelength)
            if(vectortonodelength < closestdist):
                closestdist = vectortonodelength
                closestnode = node
        return closestnode

    # Implement an RRT starting from robot position

    def train(self, net,data, label_name, NUM_OF_RRT_EPOCH):
        

        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        
        for epoch in range(NUM_OF_RRT_EPOCH):  # loop over the dataset multiple times
            for i in range(len(data)):
                # get the inputs
                inputs, labels = torch.tensor(data[i]['features']) ,torch.tensor(data[i][label_name])
                #print(inputs)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output_mu, output_sigma = net(inputs)
                
                #print(torch.tensor(labels))
                loss = criterion(output_mu, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                
            print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, loss.item()))

        print('Finished Training')
        return net

    def runRRT(self, NUM_OF_RRT_EPOCH):
        print(NUM_OF_RRT_EPOCH)
        self.game.updateDisplay()
        #self.screen.fill(self.black)
        STEP = 100 * self.ROBOTRADIUS
        #print "RRT!"
        root = anytree.Node((self.x, self.y))
        for i in range(10000):
            
            while True:
              randompoint = ((random.randint(self.PLAYFIELDCORNERS[0], self.PLAYFIELDCORNERS[2] ), random.randint(self.PLAYFIELDCORNERS[1], self.PLAYFIELDCORNERS[3])))
              if not(self.game.nodeHasHitBaddie(pygame.Rect(randompoint[0],randompoint[1],self.game.playerRect.w+5,self.game.playerRect.h+5))):
                break 
            
                
            #print(randompoint)
            closestnode = self.findClosestNode(randompoint[0], randompoint[1], root)

            # Now we have closest node, try to create new node
            vectortonode = (randompoint[0] - closestnode.name[0], randompoint[1] - closestnode.name[1])
            vectortonodelength = math.sqrt(vectortonode[0] **2 + vectortonode[1] **2)
            if (vectortonodelength <= STEP):
                newpossiblepoint = randompoint
                
            else:
                stepvector = (vectortonode[0] * STEP / vectortonodelength, vectortonode[1] * STEP / vectortonodelength)
                newpossiblepoint = (closestnode.name[0] + stepvector[0], closestnode.name[1] + stepvector[1])

            # Is this node valid?
            obdist = self.calculateClosestObstacleDistance(newpossiblepoint[0], newpossiblepoint[1], 1)
            if (obdist > 30):
                nextnode = anytree.Node((newpossiblepoint[0], newpossiblepoint[1]), parent=closestnode)


            if (i % 1 == 0) :
                pygame.draw.circle(self.screen, (255,255,255), (int(self.target[0]), int(self.target[1])), int(self.ROBOTRADIUS), 0)


    # Draw robot
                #u = self.game.playerRect.x#u0 + k * x
                #v = self.game.playerRect.y#v0 - k * y
                #pygame.draw.circle(screen, (255,255,255), (int(u), int(v)), int(k * ROBOTRADIUS), 3)
                #gameDisplay = pygame.display.set_mode((1500,1000))
                #gameDisplay.blit(playerImage, ((int(u),int(v))))

                for node in anytree.PreOrderIter(root):
                    if (node.parent != None):
                        #print(node.name)
                        #input()
                        pygame.draw.line(self.screen, (100,100,100), (int( node.name[0]), int(node.name[1])), (int(node.parent.name[0]), int(node.parent.name[1])))
                
                pygame.display.flip()
                pygame.display.update()
                #time.sleep(0.1)

            #distfromtarget = math.sqrt((newpossiblepoint[0] - self.target[0])**2 + (newpossiblepoint[1] - self.target[1])**2)
            
            
                
            
            if(self.game.nodeHasRichDestiny(pygame.Rect(newpossiblepoint[0],newpossiblepoint[1],self.game.playerRect.w,self.game.playerRect.h))):
                break 
            #if (distfromtarget < 2* ROBOTRADIUS):#hasHitDestiny
                #break


        # Try making a path
        startnode = self.findClosestNode(self.x, self.y, root)
        targetnode = self.findClosestNode(self.target[0], self.target[1], root)
        #pygame.draw.line(screen, (255,255,255), (int(startnode.name[0]), int(startnode.name[1])), (int(targetnode.parent.name[0]), int(targetnode.parent.name[1])))
        pygame.display.flip()
        w = anytree.Walker()
        (upwardsnode, commonnode, downwardsnodes) = w.walk(startnode, targetnode)
        dataSet = []
        for i, node in enumerate (downwardsnodes):
            dst = {}
            #print "Node", i, "\n"
            #print node, "\n"
            #print(i,node)
            #input()
            if node != []:
                #pygame.draw.circle(screen, (255,0,0), (int(node.name[0]), int(node.name[1])), 5, 0)
                
                point_a = numpy.array((float(self.game.playerRect.x),float(self.game.playerRect.y)))
                point_b = numpy.array((float(node.name[0]),float(node.name[1])))
                state = (self.game.getState())
                state = self.convolution(state)
                state = (torch.FloatTensor(state))#.to(device))
                state_distance = numpy.linalg.norm(point_a - point_b )
                state_angle = math.degrees(math.atan2(float(node.name[1])-float(self.game.playerRect.y), float(node.name[0])-float(self.game.playerRect.x)))
                dst['features'] = state
                dst['distance'] = state_distance
                dst['angle'   ] =  state_angle
                dataSet.append(dst)
                
                #print(state_angle)
                #print(node.name)
                self.game.playerRect.x = int(node.name[0])
                self.game.playerRect.y = int(node.name[1])
                pygame.display.flip()
                # Convolve
                # Angle
                # DATASET creation
                self.game.updateDisplay()
                #time.sleep(0.1)

        self.game.playerRect.x = self.firstX
        self.game.playerRect.y = self.firstY
        self.game.updateDisplay()
        
        self.actor_distance = self.train(self.actor_distance, dataSet, 'distance', NUM_OF_RRT_EPOCH)
        self.actor_angle = self.train(self.actor_angle, dataSet, 'angle', NUM_OF_RRT_EPOCH)
        #for _ in (NUM_OF_RRT_EPOCH):
            #train_networks
        return self.actor_distance, self.actor_angle, self.convolution
    

    
#if __name__ == '__main__':
