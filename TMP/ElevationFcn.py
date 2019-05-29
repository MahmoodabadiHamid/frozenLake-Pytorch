def ElevationFcn(Position):
    import math

    a0, a1, a2, a3, a4, a5, a6, a7, a8 = 300, 300, 0, 50, 100, 0, 1, 1, 1
    
    Elevation = a0 + a1 * math.sin( a2 + Position[1] ) + a3 * math.sin( Position[0] )  + a4 * math.cos( a5 * math.sqrt( Position[0]**2 + Position[1]**2 ) ) + a6 * math.cos( Position[1] ) + a7 * math.sin( a7 * math.sqrt( Position[0]**2 + Position[1]**2 ) ) + a8 * math.cos( Position[1] )
    return Elevation
