import sys
import numpy as np

from anytree import Node, RenderTree, AsciiStyle, Walker, PreOrderIter

infilename = sys.argv[1]
infile = open(infilename,'r')

mymap = []

for line in infile:
    print(line)
    vals = [*line[:-1]]
    print(vals)
    mymap.append(vals)

mymap = np.array(mymap).T.astype(int)

xdim,ydim = mymap.shape

total = 2*xdim + 2*(ydim-2)
print(f"total: {total}")

# Part A
'''
coords = []

# traverse the top row down t --> b
print("Traverse top row, down")
for i in range(0,xdim):
    tallest = mymap[i][0]
    for j in range(1,ydim):
        tree1 = mymap[i][j-1]
        tree2 = mymap[i][j]
        print(i,j,tree1,tree2,tallest)
        if tree2>tree1 and tree2>tallest:
            coord = (i,j)
            print("found!: ",coord)
            if coord not in coords:
                coords.append(coord)
            tallest = tree2
        elif tallest ==9:
            break

print()
print(mymap.T)
print()
print(coords)
print()

#exit()

# traverse the left column down t --> b
print("Traverse left column, down")
for i in range(0,ydim):
    tallest = mymap[0][i]
    for j in range(1,xdim):
        tree1 = mymap[j-1][i]
        tree2 = mymap[j][i]
        print(i,j,tree1,tree2,tallest)
        if tree2>tree1 and tree2>tallest:
            coord = (j,i)
            print("found!: ",coord)
            if coord not in coords:
                coords.append(coord)
            tallest = tree2
        elif tallest ==9:
            break

print()
print(mymap.T)
print()
print(coords)

# traverse the bottom row up b --> t
print("Traverse the bottom row up")
for i in range(0,xdim):
    tallest = mymap[i][ydim-1]
    for j in range(ydim-1,0,-1):
        tree1 = mymap[i][j]
        tree2 = mymap[i][j-1]
        print(i,j,tree1,tree2,tallest)
        if tree2>tree1 and tree2>tallest:
            coord = (i,j-1)
            print("found!: ",coord)
            if coord not in coords:
                coords.append(coord)
            tallest = tree2
        elif tallest ==9:
            break

print()
print(mymap.T)
print()
print(coords)
print()

#exit()

# traverse the right column left r --> l
print("Traverse the right column left")
for i in range(0,ydim):
    tallest = mymap[xdim-1][i]
    for j in range(xdim-1,0,-1):
        tree1 = mymap[j][i]
        tree2 = mymap[j-1][i]
        print(i,j,tree1,tree2,tallest)
        if tree2>tree1 and tree2>tallest:
            coord = (j-1,i)
            print("found!: ",coord)
            if coord not in coords:
                coords.append(coord)
            tallest = tree2
        elif tallest ==9:
            break

print()
print(mymap.T)
print()
print(coords)
print(len(coords))
print(f"total: {total}")
for c in coords:
    if c[0]>0 and c[0]<xdim-1 and \
            c[1]>0 and c[1]<ydim-1:
                print(c)
                total += 1
print(f"total: {total}")

print(mymap.shape)
print()

test = np.zeros(mymap.shape, dtype=int)
for c in coords:
    if c[0]>0 and c[0]<xdim-1 and \
            c[1]>0 and c[1]<ydim-1:
                #print(c)
                test[c[0],c[1]] = 1
#print(test)
for line in test.T:
    output = ""
    for c in line:
        output += str(c)
    print(output)

coords = []

# traverse the top row down t --> b
print("Traverse top row, down")
for i in range(0,xdim):
    tallest = mymap[i][0]
    for j in range(1,ydim):
        tree1 = mymap[i][j-1]
        tree2 = mymap[i][j]
        print(i,j,tree1,tree2,tallest)
        if tree2>tree1 and tree2>tallest:
            coord = (i,j)
            print("found!: ",coord)
            if coord not in coords:
                coords.append(coord)
            tallest = tree2
        elif tallest ==9:
            break

print()
print(mymap.T)
print()
print(coords)
print()

#exit()

# traverse the left column down t --> b
print("Traverse left column, down")
for i in range(0,ydim):
    tallest = mymap[0][i]
    for j in range(1,xdim):
        tree1 = mymap[j-1][i]
        tree2 = mymap[j][i]
        print(i,j,tree1,tree2,tallest)
        if tree2>tree1 and tree2>tallest:
            coord = (j,i)
            print("found!: ",coord)
            if coord not in coords:
                coords.append(coord)
            tallest = tree2
        elif tallest ==9:
            break

print()
print(mymap.T)
print()
print(coords)

# traverse the bottom row up b --> t
print("Traverse the bottom row up")
for i in range(0,xdim):
    tallest = mymap[i][ydim-1]
    for j in range(ydim-1,0,-1):
        tree1 = mymap[i][j]
        tree2 = mymap[i][j-1]
        print(i,j,tree1,tree2,tallest)
        if tree2>tree1 and tree2>tallest:
            coord = (i,j-1)
            print("found!: ",coord)
            if coord not in coords:
                coords.append(coord)
            tallest = tree2
        elif tallest ==9:
            break

print()
print(mymap.T)
print()
print(coords)
print()

#exit()

# traverse the right column left r --> l
print("Traverse the right column left")
for i in range(0,ydim):
    tallest = mymap[xdim-1][i]
    for j in range(xdim-1,0,-1):
        tree1 = mymap[j][i]
        tree2 = mymap[j-1][i]
        print(i,j,tree1,tree2,tallest)
        if tree2>tree1 and tree2>tallest:
            coord = (j-1,i)
            print("found!: ",coord)
            if coord not in coords:
                coords.append(coord)
            tallest = tree2
        elif tallest ==9:
            break

print()
print(mymap.T)
print()
print(coords)
print(len(coords))
print(f"total: {total}")
for c in coords:
    if c[0]>0 and c[0]<xdim-1 and \
            c[1]>0 and c[1]<ydim-1:
                print(c)
                total += 1
print(f"total: {total}")

print(mymap.shape)
print()

test = np.zeros(mymap.shape, dtype=int)
for c in coords:
    if c[0]>0 and c[0]<xdim-1 and \
            c[1]>0 and c[1]<ydim-1:
                #print(c)
                test[c[0],c[1]] = 1
#print(test)
for line in test.T:
    output = ""
    for c in line:
        output += str(c)
    print(output)
'''


# Part B

def distance(coord):
    i0 = coord[0]
    j0 = coord[1]
    # Can always see the tree right next to you
    #icount = [1, 1, 1, 1]
    icount = [0, 0, 0, 0]

    tree0 = mymap[i0][j0]

    # Go right
    tallest = 0
    for i in range(i0+1,xdim):
        tree2 = mymap[i][j0]
        if tree2<tree0:# and tree2>=tree1:
            icount[0] += 1
        elif tree2>=tree0:# and tree2>=tree1:
            icount[0] += 1
            break

    # Go left
    tallest = 0
    for i in range(i0-1,-1,-1):
        tree2 = mymap[i][j0]
        if tree2<tree0:# and tree2>=tree1:
            icount[1] += 1
        elif tree2>=tree0:# and tree2>=tree1:
            icount[1] += 1
            break

    # Go down
    tallest = 0
    for j in range(j0+1,ydim):
        tree2 = mymap[i0][j]
        if tree2<tree0:# and tree2>=tree1:
            icount[2] += 1
        elif tree2>=tree0:# and tree2>=tree1:
            icount[2] += 1
            break

    # Go up
    tallest = 0
    for j in range(j0-1,-1,-1):
        tree2 = mymap[i0][j]
        if tree2<tree0:# and tree2>=tree1:
            icount[3] += 1
        elif tree2>=tree0:# and tree2>=tree1:
            icount[3] += 1
            break

    return icount


for i in range(1,xdim-1):
    for j in range(1,ydim-1):
        x = distance((i,j))
        print(x)
        total = 1
        for n in x:
            total *= n
        print(f'TOTAL: {i}  {j}  {total}  {x}')


