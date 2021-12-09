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
exit()

def walk_around(i,j,grid,nsteps=0):
    # Figure out the range in the row
    # Move left
    print(f"Walking around --- {i} {j}")
    for lox in range(j,-1,-1):
        if lox==0:
            break
        if grid[i][lox] == True:
            break
    # Move right
    for hix in range(j,width+1,1):
        if hix==width-1:
            break
        if grid[i][hix] == True:
            break

    # Need an offset to start correctly in the search
    lox+=1
    print(f"lox:hix {lox} {hix}")
    for jstep in range(lox,hix+1):
        # Move down
        for istep in range(i,height):
            #print(f"STEPPING DOWN -  istep: {istep}\tjstep: {jstep}\tgrid: {grid[istep][jstep]}\tdata: {data[istep][jstep]}\tnsteps: {nsteps}")
            if grid[istep][jstep] == False:
                nsteps += 1
            else:
                break

        # Move up
        for istep in range(i-1,0,-1):
            if i<0:
                break
            #print(f"STEPPING UP   -  istep: {istep}\tjstep: {jstep}\tgrid: {grid[istep][jstep]}\tdata: {data[istep][jstep]}\tnsteps: {nsteps}")
            if grid[istep][jstep] == False:
                nsteps += 1
            else:
                break

    return nsteps


all_nsteps = []
for (i,j) in indices_of_low_points:
    nsteps = 0
    nsteps = walk_around(i,j,nines,nsteps=nsteps)
    print(f"nsteps: {nsteps}")
    all_nsteps.append(nsteps)
print(all_nsteps)

sorted = np.sort(all_nsteps)

print(sorted)
tot = 1
for n in sorted[-3:]:
    tot *= n
    print(n,tot)

print(f"tot: {tot}")
'''
def walk(i,j,grid,nsteps=0,openpaths=[True,True,True,True],traversed=[]):
    # openpaths = [right,left,up,down]
    height,width = grid.shape
    print("In walk!")
    print(i,j,nsteps,height,width)
    print()

    istep = i
    jstep = j

    for idx,path in enumerate(openpaths):
        if not path:
            continue 

        if idx==0:
            istep += 1
        elif idx==1:
            istep -= 1
        elif idx==2:
            jstep -= 1
        elif idx==3:
            jstep += 1

        print(f"\nidx: {idx}\tistep: {istep}\tjstep: {jstep}")
        if istep<0 or istep==width-1 or jstep<0 or jstep==height-1:
            print("A")
            openpaths[idx] = False
            nsteps = walk(i,j,grid,nsteps=nsteps,openpaths=openpaths,traversed=traversed)

        print(f"idx: {idx}\tistep: {istep}\tjstep: {jstep}\tgrid: {grid[istep][jstep]}")
        if grid[istep][jstep] is False:
            print("B")

            nsteps += 1

            traversed.append([istep,jstep])

            nsteps = walk(istep,jstep,grid,nsteps=nsteps,openpaths=openpaths,traversed=traversed)
        else:
            print("C")
            openpaths[idx] = False

    return nsteps

nsteps = 0
openpaths=[True,True,True,True]
traversed = []
x = walk(0,1,nines,nsteps,openpaths=openpaths,traversed=traversed)
print(x)
'''


    

