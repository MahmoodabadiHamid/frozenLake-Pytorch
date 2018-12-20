import pygame, os, math, time, random
from pygame.locals import *
import anytree

pygame.init()
playerImage = pygame.image.load('large.gif')
baddieImage = pygame.image.load('baddie.jpg')
# set the width and height of the screen
WIDTH = 1500
HEIGHT = 1000

# The region we will fill with obstacles
PLAYFIELDCORNERS = (-3.0, -3.0, 3.0, 3.0)

# Barrier locations
barriers = []
# barrier contents are (bx, by, visibilitymask)
for i in range(270):
    (bx, by) = (random.uniform(PLAYFIELDCORNERS[0], PLAYFIELDCORNERS[2]), random.uniform(PLAYFIELDCORNERS[1], PLAYFIELDCORNERS[3]))
    barrier = [bx, by, 0]
    barriers.append(barrier)
#barrier = [0, 2.5, 0]
#barriers.append(barrier)

BARRIERRADIUS = 0.06

ROBOTRADIUS = 0.15
W = 2 * ROBOTRADIUS
SAFEDIST = 0.2

MAXVELOCITY = 0.5     #ms^(-1) max speed of each wheel
MAXACCELERATION = 0.5 #ms^(-2) max rate we can change speed of each wheel


target = (PLAYFIELDCORNERS[2] + 1.0, 0)

# Array for path choices to draw
pathstodraw = []


k = 160 # pixels per metre for graphics
u0 = WIDTH / 2
v0 = HEIGHT / 2

x = PLAYFIELDCORNERS[0] - 0.5
y = 0.0
theta = 0.0

vL = 0.00
vR = 0.00




# Transformation from metric frame to graphics frame
# k pixels per metre
# u = u0 + k * x
# v = v0 - k * y


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
        if(barrier[2] == 1 or RRTFLAG == 1):
            dx = barrier[0] - x
            dy = barrier[1] - y
            d = math.sqrt(dx**2 + dy**2)
            dist = d - BARRIERRADIUS - ROBOTRADIUS
            if (dist < closestdist):
                closestdist = dist

    return closestdist

def drawBarriers(barriers):
    for barrier in barriers:
        if(barrier[2] == 0):
            pygame.draw.circle(screen, (0,40,150), (int(u0 + k * barrier[0]), int(v0 - k * barrier[1])), int(k * BARRIERRADIUS), 0)
        else:
            pygame.draw.circle(screen, (0,120,255), (int(u0 + k * barrier[0]), int(v0 - k * barrier[1])), int(k * BARRIERRADIUS), 0)
    return

def findClosestNode(x, y, root):
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
def RRT(x, y, theta):
    screen.fill(black)
    STEP = 2*ROBOTRADIUS
    #print "RRT!"
    root = anytree.Node((x, y))
    for i in range(10000):
        randompoint = ((random.uniform(PLAYFIELDCORNERS[0], PLAYFIELDCORNERS[2] + 1.0), random.uniform(PLAYFIELDCORNERS[1], PLAYFIELDCORNERS[3])))
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
            pygame.draw.circle(screen, (255,100,0), (int(u0 + k * target[0]), int(v0 - k * target[1])), int(k * ROBOTRADIUS), 0)


# Draw robot
            u = u0 + k * x
            v = v0 - k * y
            pygame.draw.circle(screen, (255,255,255), (int(u), int(v)), int(k * ROBOTRADIUS), 3)
            #gameDisplay = pygame.display.set_mode((1500,1000))
            #gameDisplay.blit(playerImage, ((int(u),int(v))))

            for node in anytree.PreOrderIter(root):
                if (node.parent != None):
                    pygame.draw.line(screen, (100,100,100), (int(u0 + k * node.name[0]), int(v0 - k * node.name[1])), (int(u0 + k * node.parent.name[0]), int(v0 - k * node.parent.name[1])))
            drawBarriers(barriers)
            pygame.display.flip()
            #time.sleep(0.1)

        distfromtarget = math.sqrt((newpossiblepoint[0] - target[0])**2 + (newpossiblepoint[1] - target[1])**2)
        if (distfromtarget < 2* ROBOTRADIUS):
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

RRT(x, y, theta)
time.sleep(100)
