import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from pyswarm import pso
import scipy.ndimage

import ElevationFcn
import read_file
def PsoPathplanning():
    global TerrainClearance
    global GoalPosition
    global ThreatPosition
    global ThreatRadius
    global PositionV
    global pvtemp
    InitialPosition, GoalPosition, ThreatPosition = read_file.read_file('position.txt')

    
    TerrainClearance = 500
    InitialPosition[2] = ElevationFcn.ElevationFcn( InitialPosition[: 2] ) + 3500;
    GoalPosition[2] = ElevationFcn.ElevationFcn( GoalPosition[0:2] ) + TerrainClearance
    LB = [0, -90, 0]
    UB = [360, +90, 1000]
    ThreatRadius = 3000
    PositionV=[]   
    PositionV.append(list(InitialPosition))
    PositionVTemp = []
    pvtemp = np.zeros(3, dtype = 'double');
    i = 0;
    FunVal = 11;
    xOpt = []
    while FunVal > 10:
        PositionVTemp = []
        pvtemp = PositionV[i][:]

        a, FunVal = pso(OptFcn, LB, UB, swarmsize=100, omega=0.2, phip=0.3, phig=0.9, maxiter=100, 
        minstep=1e-8, minfunc=1e-8, debug=False)
        xOpt.append(a[0])
        xOpt.append(a[1])
        xOpt.append(a[2])
    
        PositionVTemp.append(PositionV[i][0] + xOpt[2] * (math.cos(math.radians(xOpt[1]))) * (math.cos(math.radians(xOpt[0]))))
        PositionVTemp.append(PositionV[i][1] + xOpt[2] * (math.cos(math.radians(xOpt[1]))) * (math.sin(math.radians(xOpt[0]))))
        PositionVTemp.append(PositionV[i][2] + xOpt[2] * (math.sin(math.radians(xOpt[1]))))
        PositionV.append(list(PositionVTemp))
        xOpt = []
        i = i + 1

    TerrainMap()
    f = open("PositionV.txt", "w")
    f.writelines(["%s\n" % item  for item in PositionV])    
    f.close()
    return PositionV
#######################################################
def OptFcn(u):
    import math

    Phi = u[0] # [deg]
    Theta = u[1] # [deg]
    Distance = u[2] # [m]
    N = 1
    InitialPosition = pvtemp
    X = InitialPosition[0]
    Y = InitialPosition[1]
    Z = InitialPosition[2]
    Error = 0
    for i in range(N):
        X = X + Distance *  (math.cos(math.radians(Theta))) *  (math.cos(math.radians(Phi))) / N;
        Y = Y + Distance * (math.cos(math.radians(Theta))) * (math.sin(math.radians(Phi))) / N;
        Z  = Z + Distance * (math.sin(math.radians(Theta))) / N;
        Error = Error + TerrainCostFcn( [X, Y, Z] ) + ThreatCostFcn( [X, Y, Z] ) + GoalCostFcn( InitialPosition, [X, Y, Z] );

    return Error
########################################################

def TerrainCostFcn( Position ):
    import math
    import ElevationFcn
    global TerrainClearance
    elev = ElevationFcn.ElevationFcn( Position )
    clrnc =  TerrainClearance
    
    Sigma1 = TerrainClearance / 8 
    G1 = 1e-50
    m1 = clrnc
    if Position[2] > (elev + TerrainClearance):
        TerrainCost = 0
    elif Position[2] <= (elev + TerrainClearance):
        TerrainCost = ( math.sqrt( 2 * math.pi * Sigma1 ) / G1 ) * math.exp( ( Position[2] - elev -  m1 )**2 / (2 * Sigma1**2) );
    
    return TerrainCost
######################################################
def ThreatCostFcn( Position ):
    import math


    Re = ThreatRadius


    ThreatCostV = np.zeros(len( ThreatPosition ), dtype = 'double' );
    for j in range(len( ThreatPosition)):
        PX = math.sqrt( ( ThreatPosition[j][ 0] - Position[0] )**2 + ( ThreatPosition[j][1] - Position[1] )**2 );
        if PX < Re:
            ThreatCostV[j] = 1e10
        elif PX >= Re:
            ThreatCostV[j] = 0
    ThreatCost = sum( ThreatCostV )
    
    def LOSFcn( Position1, Position2 ):
        import ElevationFcn
        inLOS = 1;
        N = 100;
        for i in range( N ):
            x = Position1[0] + (i / N) * ( Position2[0] - Position1[0] )
            y = Position1[1] + (i / N) * ( Position2[1] - Position1[1] )
            z = Position1[2] + (i / N) * ( Position2[2] - Position1[2] )
            Elevation = ElevationFcn.ElevationFcn( [x, y] );
            if Elevation > z:
                inLOS = 0
                break
        return inLOS
    return ThreatCost

###############################################
def GoalCostFcn( InitialPosition, FinalPosition ):
    import math

    GoalCost1 = 1 * math.sqrt( ( FinalPosition[0] - GoalPosition[0] )**2 + ( FinalPosition[1] - GoalPosition[1] )**2 + ( FinalPosition[2] - GoalPosition[2] )**2 );
    GoalCost = GoalCost1;
    return GoalCost
########################################################
def TerrainMap():
    xDomain, yDomain = np.mgrid[0:40001:40j, 0:40001:40j]
    x, y = xDomain.shape
    ElevationV = np.zeros( [x, y], dtype = 'double' );
    for i in range(x):
        for j in range(y):
            x0 = xDomain[i, j]
            y0 = yDomain[i, j]
            ElevationV[i, j] = ElevationFcn.ElevationFcn([x0, y0])
    fig = plt.figure()  
    ax = fig.gca(projection = '3d')
    ax.plot_surface(xDomain, yDomain, ElevationV, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    
    u, v = np.mgrid[0 : 2 * np.pi : 200j, 0 : 2 * np.pi : 100j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    l = len(ThreatPosition)
    for i in range(l):
        ThreatPosition[i][2] = ElevationFcn.ElevationFcn([ThreatPosition[i][0], ThreatPosition[i][1]])
        x0 = x * ThreatRadius + ThreatPosition[i][0]
        y0 = y * ThreatRadius + ThreatPosition[i][1]
        z0 = z * ThreatRadius + ThreatPosition[i][2]
        ax.plot_surface(x0, y0, z0)
    
    ax.scatter([i[0] for i in PositionV], [i[1] for i in PositionV], [i[2] for i in PositionV])
    plt.show()
    
    plt.figure()
    plt.scatter([i[0] for i in PositionV], [i[1] for i in PositionV])
    a = [i[0] for i in ThreatPosition]
    b = [i[1] for i in ThreatPosition]
    plt.scatter(a , b)
    plt.show()
    return
######################################################
if __name__=="__main__":
    PsoPathplanning()

