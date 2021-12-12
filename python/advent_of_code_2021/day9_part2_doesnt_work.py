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

def walk_around(i,j,grid,nsteps=0):
    # Figure out the range in the row
    # Move left
    print(f"Walking around --- {i} {j}")
    '''
    for lox in range(j,-1,-1):
        if lox==0:
            break
        if grid[i][lox] == True:
            break
    # Move right
    for hix in range(j,width+1,1):
        if hix==width-1:
            hix = width
            break
        if grid[i][hix] == True:
            break

    # Figure out the range in the column
    # Move up
    for loy in range(i,-1,-1):
        if loy==0:
            break
        if grid[loy][j] == True:
            break
    # Move down
    for hiy in range(i,height,1):
        if hiy==height-1:
            hiy = height
            break
        if grid[hiy][j] == True:
            break

    '''
    good_grid_idx = []
    # Search left
    jstep = j
    while jstep>=0:
        # If we get to a 9 on the horizontal, then break
        if grid[i][jstep] == True:
            break
        # Search down
        istep = i
        while istep<height:
            if grid[istep][jstep] == False:
                good_grid_idx.append([istep,jstep])
            else:
                break
            istep += 1
        # Search up
        istep = i
        while istep>=0:
            if grid[istep][jstep] == False:
                good_grid_idx.append([istep,jstep])
            else:
                break
            istep -= 1
        jstep -= 1

    # Search right
    jstep = j
    while jstep<width:
        # If we get to a 9 on the horizontal, then break
        if grid[i][jstep] == True:
            break
        # Search down
        istep = i
        while istep<height:
            if grid[istep][jstep] == False:
                good_grid_idx.append([istep,jstep])
            else:
                break
            istep += 1
        # Search up
        istep = i
        while istep>=0:
            if grid[istep][jstep] == False:
                good_grid_idx.append([istep,jstep])
            else:
                break
            istep -= 1
        jstep += 1

    # Search up
    istep = i
    while istep>=0:
        # If we get to a 9 on the horizontal, then break
        if grid[istep][j] == True:
            break
        # Search right
        jstep = j
        while jstep<width:
            if grid[istep][jstep] == False:
                good_grid_idx.append([istep,jstep])
            else:
                break
            jstep += 1
        # Search left
        jstep = j
        while jstep>=0:
            if grid[istep][jstep] == False:
                good_grid_idx.append([istep,jstep])
            else:
                break
            jstep -= 1
        istep -= 1

    # Search down
    istep = i
    while istep<height:
        # If we get to a 9 on the horizontal, then break
        if grid[istep][j] == True:
            break
        # Search down
        jstep = j
        while jstep<height:
            if grid[istep][jstep] == False:
                good_grid_idx.append([istep,jstep])
            else:
                break
            jstep += 1
        # Search up
        jstep = j
        while jstep>=0:
            if grid[istep][jstep] == False:
                good_grid_idx.append([istep,jstep])
            else:
                break
            jstep -= 1
        istep += 1

    return good_grid_idx


    #print(f"VER STEPPING L    -  istep: {istep}\tjstep: {jstep}\tgrid: {grid[istep][jstep]}\tdata: {data[istep][jstep]}\tnsteps: {nsteps}")


all_nsteps = []
for (i,j) in indices_of_low_points:
    nsteps = 0
    ggi = walk_around(i,j,nines,nsteps=nsteps)
    #print(f"nsteps: {nsteps}")
    #print("ggi: ")
    #print(ggi)
    unique_ggi = []
    tempmatch = np.zeros(shape=data.shape,dtype=bool)
    for g in ggi:
        #print(g)
        if g not in unique_ggi:
            unique_ggi.append(g)
            tempmatch[g[0]][g[1]] = True
            #print("\t",unique_ggi)
    #print(unique_ggi)
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


    

