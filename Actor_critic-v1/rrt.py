import pygame, os, math, time, random
from pygame.locals import *
import anytree
import gameEnv

class RRT():
    def __init__(self, game, actor_distance, actor_angle):
        pygame.init()
        game = gameEnv.game('actor_distance', 'actor_angle', 'critic', level='EASY')
        WIDTH = game.winW
        HEIGHT = game.winH

        # The region we will fill with obstacles
        PLAYFIELDCORNERS = (0, 0, game.winW, game.winH)

        # Barrier locations
        tmpBarriers = game.baddies
        barriers = []
        # barrier contents are (bx, by, visibilitymask)
        for b in tmpBarriers:
            #print(b['rect'])
            #print(b['rect'].x)
            (bx, by) = (b['rect'].x,b['rect'].y)#(random.uniform(PLAYFIELDCORNERS[0], PLAYFIELDCORNERS[2]), random.uniform(PLAYFIELDCORNERS[1], PLAYFIELDCORNERS[3]))
            barrier = [bx, by, 0]
            barriers.append(barrier)
            
        #barrier = [0, 2.5, 0]
        #barriers.append(barrier)

        BARRIERRADIUS = 10 

        ROBOTRADIUS = 0.15
        W = 2 * ROBOTRADIUS
        

        target = (game.destiny['rect'].x, game.destiny['rect'].y)

        # Array for path choices to draw
        pathstodraw = []


        x = game.playerRect.x
        y = game.playerRect.y
        

        size = [WIDTH, HEIGHT]
        screen = pygame.display.set_mode(size)
        black = (20,20,40)

        # This makes the normal mouse pointer invisible in graphics window
        pygame.mouse.set_visible(0)

        # time delta
        dt = 0.1


    def calculateClosestObstacleDistance(x, y, RRTFLAG):
        closestdist = 100000.0
        # Calculate distance to closest obstacle
        for barrier in barriers:
            # Is this a barrier we know about? 
            # For local planning, only if we've seen it. For RRT, always
            if True:#(barrier[2] == 1 or RRTFLAG == 1):
                #print(barrier[0])
                #input()
                dx = barrier[0] - x
                dy = barrier[1] - y
                d = math.sqrt(dx**2 + dy**2)
                dist = d - BARRIERRADIUS - ROBOTRADIUS
                if (dist < closestdist):
                    closestdist = dist

        return closestdist



    def findClosestNode(x, y, root):
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
    def runRRT(x, y):
        screen.fill(black)
        STEP = 100*ROBOTRADIUS
        #print "RRT!"
        root = anytree.Node((x, y))
        for i in range(10000):
            
            while True:
              randompoint = ((random.randint(PLAYFIELDCORNERS[0], PLAYFIELDCORNERS[2] ), random.randint(PLAYFIELDCORNERS[1], PLAYFIELDCORNERS[3])))
              if not(game.playerHasHitBaddie(pygame.Rect(randompoint[0],randompoint[1],game.playerRect.w+5,game.playerRect.h+5))):
                break 
            
                
            #print(randompoint)
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
            if (obdist > 30):
                nextnode = anytree.Node((newpossiblepoint[0], newpossiblepoint[1]), parent=closestnode)


            if (i % 1 == 0) :
                pygame.draw.circle(screen, (255,255,255), (int(target[0]), int(target[1])), int(ROBOTRADIUS), 0)


    # Draw robot
                u = game.playerRect.x#u0 + k * x
                v = game.playerRect.y#v0 - k * y
                #pygame.draw.circle(screen, (255,255,255), (int(u), int(v)), int(k * ROBOTRADIUS), 3)
                #gameDisplay = pygame.display.set_mode((1500,1000))
                #gameDisplay.blit(playerImage, ((int(u),int(v))))

                for node in anytree.PreOrderIter(root):
                    if (node.parent != None):
                        #print(node.name)
                        #input()
                        pygame.draw.line(screen, (100,100,100), (int( node.name[0]), int(node.name[1])), (int(node.parent.name[0]), int(node.parent.name[1])))
                #drawBarriers(barriers)
                pygame.display.flip()
                game.updateDisplay()
                #time.sleep(0.1)

            distfromtarget = math.sqrt((newpossiblepoint[0] - target[0])**2 + (newpossiblepoint[1] - target[1])**2)
            
            
                
            
            if(game.playerHasRichDestiny(pygame.Rect(newpossiblepoint[0],newpossiblepoint[1],game.playerRect.w,game.playerRect.h))):
                break 
            #if (distfromtarget < 2* ROBOTRADIUS):#hasHitDestiny
                #break


        # Try making a path
        startnode = findClosestNode(x, y, root)
        targetnode = findClosestNode(target[0], target[1], root)
        #pygame.draw.line(screen, (255,255,255), (int(startnode.name[0]), int(startnode.name[1])), (int(targetnode.parent.name[0]), int(targetnode.parent.name[1])))
        pygame.display.flip()
        w = anytree.Walker()
        (upwardsnode, commonnode, downwardsnodes) = w.walk(startnode, targetnode)
        for i, node in enumerate (downwardsnodes):
            #print "Node", i, "\n"
            #print node, "\n"
            if node != []:
                #pygame.draw.circle(screen, (255,0,0), (int(node.name[0]), int(node.name[1])), 5, 0)
                game.playerRect.x = int(node.name[0])
                game.playerRect.y = int(node.name[1])
                pygame.display.flip()
                # Convolve
                # Angle
                # Train: actor_distance, actorangle
                game.updateDisplay()
                time.sleep(0.1)
        return

    #RRT()
    #time.sleep(1)


if __name__ == '__main__':
    import rrt
    import gameEnv
    rrt = rrt.RRT('','','')
    print('hi')
    rrt.RRT(10,10)



