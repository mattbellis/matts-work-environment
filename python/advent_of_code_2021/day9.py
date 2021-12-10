import numpy as np
import sys

################################################################################

data = np.loadtxt(sys.argv[1],dtype=str)

print(data)

temp = []
for d in data:
    row = []
    print(d)
    for i in d:
        row.append(int(i))
    temp.append(row)
print(temp)

data = np.array(temp)

width = len(data[0])
height = len(data)

print(width,height)

lowest = np.zeros(shape=data.shape,dtype=bool)
print(lowest)

print(data.size)

indices_of_low_points = []

for i in range(0,height):
    for j in range(0,width):
        #print(i,j)
        val = data[i][j]
        # Top left
        if i==0 and j==0:
            if val<data[i+1][j] and val<data[i][j+1]:
                lowest[i][j] = True
        # top right
        elif i==0 and j==width-1:
            if val<data[i+1][j] and val<data[i][j-1]:
                lowest[i][j] = True
        # bottom left
        elif i==height-1 and j==0:
            if val<data[i-1][j] and val<data[i][j+1]:
                lowest[i][j] = True
        # bottom right
        elif i==height-1 and j==width-1:
            if val<data[i-1][j-1] and val<data[i-1][j-1]:
                lowest[i][j] = True
        # left column
        elif j==0:
            if val<data[i-1][j] and val<data[i+1][j] and val<data[i][j+1]:
                lowest[i][j] = True
        # right column
        elif j==width-1:
            if val<data[i-1][j] and val<data[i+1][j] and val<data[i][j-1]:
                lowest[i][j] = True
        # top row
        elif i==0:
            if val<data[i][j-1] and val<data[i][j+1] and val<data[i+1][j]:
                lowest[i][j] = True
        # bottom row
        elif i==height-1:
            if val<data[i][j-1] and val<data[i][j+1] and val<data[i-1][j]:
                lowest[i][j] = True
        # everywhere else
        else:
            if val<data[i][j-1] and val<data[i][j+1] and val<data[i-1][j] and val<data[i+1][j]:
                lowest[i][j] = True

for i in range(0,height):
    for j in range(0,width):
        if lowest[i][j]:
            indices_of_low_points.append([i,j])

print("Indices: ")
print(indices_of_low_points)
print()
print(lowest)
answer = data[lowest]
print(answer)

tot = np.sum(answer + 1)
print(f"Total: {tot}")
            
################################################################################
print()
print("Part 2 -=-=-=-=-=-=-=-=-=-=-=-")
print()

nines = data==9

print("Basins")
print(lowest)
print()
print("Where are the 9's?")
print(nines)

################################################################################
#exit()







def walk(i,j,grid,visited,traversed=[],nsteps=0):
    # Adapted from https://www.techiedelight.com/find-shortest-path-in-maze/
    height,width = grid.shape
    print("In walk!")
    print(i,j,height,width)
    print()
    if nsteps>100:
        return traversed,nsteps,visited

    visited[i][j] = True

    if i<0 or i==height or j<0 or j==width:
        print("A")
        return traversed,nsteps,visited

    #print("here!!!!!!!!!")
    print(grid[i][j])
    # Stay
    if grid[i][j] == False:
        print("A")
        traversed.append([i,j])
        #traversed = walk(i,j,grid,traversed=traversed)
        nsteps += 1

    # Go down 1
    if i+1<height and grid[i+1][j] == False:
        print("B")
        traversed.append([i+1,j])
        nsteps += 1
        traversed,nsteps,visited = walk(i+1,j,grid,visited,traversed=traversed,nsteps=nsteps)

    # Go right 1
    if j+1<width and grid[i][j+1] == False:
        print("D")
        traversed.append([i,j+1])
        nsteps += 1
        traversed,nsteps,visited = walk(i,j+1,grid,visited,traversed=traversed,nsteps=nsteps)

    # Go up 1
    if i-1>=0 and grid[i-1][j] == False :
        print("C")
        traversed.append([i-1,j])
        nsteps += 1
        traversed,nsteps,visited = walk(i-1,j,grid,visited,traversed=traversed,nsteps=nsteps)


    # Go left 1
    if j-1>=0 and grid[i][j-1] == False :
        print("E")
        traversed.append([i,j-1])
        nsteps += 1
        traversed,nsteps,visited = walk(i,j-1,grid,visited,traversed=traversed,nsteps=nsteps)

    visited[i][j] = False

    return traversed,nsteps,visited

'''
nsteps = 0
traversed = []
#x,nsteps = walk(0,1,nines,traversed=traversed)
x,nsteps = walk(2,2,nines,traversed=traversed)
print(nsteps)
print(x)
'''


#indices_of_low_points = x
#print(indices_of_low_points)
    
#'''
all_nsteps = []
for (i,j) in indices_of_low_points[0:2]:
    nsteps = 0
    traversed = []
    visited = np.zeros(shape=data.shape,dtype=bool)
    ggi,returned_nsteps,visited = walk(i,j,nines,visited,nsteps=nsteps,traversed=traversed)
    #print(f"nsteps: {nsteps}")
    print("ggi: ")
    print(ggi)
    unique_ggi = []
    tempmatch = np.zeros(shape=data.shape,dtype=bool)
    for g in ggi:
        #print(g)
        if g not in unique_ggi:
            unique_ggi.append(g)
            tempmatch[g[0]][g[1]] = True
            #print("\t",unique_ggi)
    print(unique_ggi)
    ugnsteps = len(unique_ggi)
    #print(tempmatch)
    print("COUNTING THE NSTEPS: ",ugnsteps)
    all_nsteps.append(ugnsteps)
print(all_nsteps)

#exit()

sorted = np.sort(all_nsteps)

print(sorted)
tot = 1
for n in sorted[-3:]:
    tot *= n
    print(n,tot)

print(f"tot: {tot}")
#'''

