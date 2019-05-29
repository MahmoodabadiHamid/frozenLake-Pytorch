def read_file(fname):
    with open(fname) as f:
#        w, h, z = [int(x) for x in next(f).split()] # read first line
        array = []
        for line in f: # read rest of lines
            array.append([int(x) for x in line.split()])
    InitialPosition = array[0]
    GoalPosition = array[1]
    ThreatPosition = array[2:]
    print(ThreatPosition)
    return InitialPosition, GoalPosition, ThreatPosition

#################################
if __name__=="__main__":
    InitialPosition, GoalPosition, ThreatPosition = read_file('Position1.txt')