import sys
import numpy as np

from anytree import Node, RenderTree, AsciiStyle, Walker, PreOrderIter

infilename = sys.argv[1]
infile = open(infilename,'r')


################################################################################
directions = {'R':np.array([1,0]), \
              'L':np.array([-1,0]), \
              'U':np.array([0,1]), \
              'D':np.array([0,-1])}

################################################################################
def separation(hcoord, tcoord):
    diff = hcoord - tcoord

    magsq = diff[0]*diff[0] + diff[1]*diff[1]
    #print(magsq)
    dx = hcoord[0] - tcoord[0]
    dy = hcoord[1] - tcoord[1]

    return magsq,dx,dy

################################################################################
def tail_move(hcoord,tcoord):

    magsq,dx,dy = separation(hcoord,tcoord)
    #print(f"here: {dx} {dy}")
    dxsign = 0
    dysign = 0
    if dx!=0:
        dxsign = int(dx/abs(dx))
    if dy!=0:
        dysign = int(dy/abs(dy))
    #print(f"sign: {dxsign} {dysign}")
    #print("here: ",tcoord)
    if magsq > 2:
        # Move tcoord
        if magsq>4: # Move diagonal
            tcoord[0] += dxsign
            tcoord[1] += dysign
        else:
            tcoord[0] += dxsign
            tcoord[1] += dysign
    #print("HERE: ",tcoord)

    return tcoord


################################################################################

# Part A
'''
hcoord = np.array([0,0])
tcoord = np.array([0,0])

alltcoords = []

for line in infile:
    #print(line)
    hdir,hnsteps = line.split()
    hnsteps = int(hnsteps)
    for n in range(hnsteps):
        step = directions[hdir]
        hcoord += step
        tcoord = tail_move(hcoord,tcoord)
        print(hcoord,tcoord)

        # Store them
        is_visited = False
        for t in alltcoords:
            #print(t, tcoord, t[0],tcoord[0] , t[1],tcoord[1])
            if t[0]==tcoord[0] and t[1]==tcoord[1]:
                is_visited = True
                #print("ISVISITED!!!!! ",is_visited)
                continue
        #print("HERE   ",tcoord,is_visited,alltcoords)
        if is_visited is False:
            #print("ADDING!!!!!!!!!!!!!!!!!!!!!!!!!")
            alltcoords.append([tcoord[0], tcoord[1]])

print(alltcoords)
print(f"# of places visited: {len(alltcoords)}")
'''


#hcoord = np.array([0,0])
rcoords = np.zeros((10,2))

print(rcoords)
print(rcoords[0])

alltcoords = []

for line in infile:
    #print(line)
    hdir,hnsteps = line.split()
    hnsteps = int(hnsteps)
    for n in range(hnsteps):
        step = directions[hdir]
        # Move the head
        rcoords[0] += step
        for j in range(1,len(rcoords)):
            rcoords[j] = tail_move(rcoords[j-1],rcoords[j])
            print(rcoords[j-1], rcoords[j])

            if j==len(rcoords)-1:
                # Store them
                is_visited = False
                for t in alltcoords:
                    #print(t, tcoord, t[0],tcoord[0] , t[1],tcoord[1])
                    if t[0]==rcoords[j][0] and t[1]==rcoords[j][1]:
                        is_visited = True
                        #print("ISVISITED!!!!! ",is_visited)
                        continue
                #print("HERE   ",tcoord,is_visited,alltcoords)
                if is_visited is False:
                    #print("ADDING!!!!!!!!!!!!!!!!!!!!!!!!!")
                    alltcoords.append([rcoords[j][0], rcoords[j][1]])

print(alltcoords)
print(f"# of places visited: {len(alltcoords)}")
