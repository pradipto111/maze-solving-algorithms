import math

def euc(i,j, end):
    return math.sqrt(((i-end[0])**2) +((j-end[1])**2))

def manh(i,j, end):
    return abs(i-end[0])+ abs(j - end[1])

def diag(i,j, end):
    return max(abs(i-end[0]), abs(j-end[1]))

def dijkstra(i,j, end):
    return 0

def Astar3(i,j,start,end):
    l = euc(i,j,end)
    L = euc(start[0], start[1], end)
    alpha = math.atan((end[0]-start[0])/(end[1]-start[1]))
    if end[1] == j:
        if end[0]> start[0]:
            theta = math.pi/2
        else:
            theta = -1*math.pi/2
    else:
        theta = math.atan((end[0]-i)/(end[1]-j))
    return 0.55*(l/L) + 0.45*(theta/alpha)


