import numpy as np
import sys

################################################################################

infilename = sys.argv[1]
infile = open(infilename)

segments = []

xmin,xmax = 1e16,-1e16
ymin,ymax = 1e16,-1e16
for i,line in enumerate(infile):
    vals = line.split(' -> ')
    x0 = int(vals[0].split(',')[0])
    y0 = int(vals[0].split(',')[1])
    x1 = int(vals[1].split(',')[0])
    y1 = int(vals[1].split(',')[1])
    if x0>xmax:
        xmax = x0
    if x1>xmax:
        xmax = x1
    if x0<xmin:
        xmin = x0
    if x1<xmin:
        xmin = x1
    if y0>ymax:
        ymax = y0
    if y1>ymax:
        ymax = y1
    if y0<ymin:
        ymin = y0
    if y1<ymin:
        ymin = y1

    segments.append([[x0,y0],[x1,y1]])

segments = np.array(segments)
xmin = 0
ymin = 0
xwidth = xmax-xmin + 1
ywidth = ymax-ymin + 1
print(xmin,xmax)
print(ymin,ymax)
print(xwidth,ywidth)
print(segments)


################################################################################
print("Processing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
grid = np.zeros(shape=(xwidth,ywidth))

for segment in segments:
    x0 = segment[0][0]
    y0 = segment[0][1]
    x1 = segment[1][0]
    y1 = segment[1][1]
    #print(x0,y0,x1,y1)

    # Vertical or horizontal
    if x0==x1 or y0==y1:

        if x0==x1:
            if y0>y1:
                temp = y0
                y0 = y1
                y1 = temp
            xidx = np.arange(y0,y1+1,dtype=int)
            yidx = x0*np.ones(len(xidx),dtype=int)
        if y0==y1:
            if x0>x1:
                temp = x0
                x0 = x1
                x1 = temp
            yidx = np.arange(x0,x1+1,dtype=int)
            xidx = y0*np.ones(len(yidx),dtype=int)

        coords = np.array([xidx,yidx]).T

        #print(coords)
        for coord in coords:
            #print("coord: ",coord)
            grid[coord[0],coord[1]] += 1

print(grid)

print()
print("Counting the grid!!!!!")
print()
tot = 0
for g in grid:
    print(g)
    x = len(g[g>=2])
    print(x)
    tot += x

print(tot)

################################################################################
print("Part 2 -------             !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
grid = np.zeros(shape=(xwidth,ywidth))

for segment in segments:
    x0 = segment[0][0]
    y0 = segment[0][1]
    x1 = segment[1][0]
    y1 = segment[1][1]
    #print(x0,y0,x1,y1)

    # Vertical or horizontal
    if x0==x1 or y0==y1:

        #print("HORIZONTAL/VERTICAL: =================")
        #print(x0,y0,x1,y1)

        if x0==x1:
            if y0>y1:
                temp = y0
                y0 = y1
                y1 = temp
            xidx = np.arange(y0,y1+1,dtype=int)
            yidx = x0*np.ones(len(xidx),dtype=int)
        if y0==y1:
            if x0>x1:
                temp = x0
                x0 = x1
                x1 = temp
            yidx = np.arange(x0,x1+1,dtype=int)
            xidx = y0*np.ones(len(yidx),dtype=int)

        coords = np.array([xidx,yidx]).T

        #print(coords)
        for coord in coords:
            #print("coord: ",coord)
            grid[coord[0],coord[1]] += 1

    # Diagonal
    elif np.abs(x0-x1) == np.abs(y0-y1):
        #print("DIAGONAL!!!!!")
        #print(x0,y0,x1,y1)

        npts = np.abs(x0-x1) + 1

        #print(x0,x1,y0,y1)
        xidx = []
        yidx = []
        if x1>x0 and y1>y0:
            #print("A")
            for i in range(npts):
                xidx.append(x0 + i)
                yidx.append(y0 + i)
        elif x0>x1 and y0>y1:
            #print("B")
            for i in range(npts):
                xidx.append(x1 + i)
                yidx.append(y1 + i)
        elif x0>x1 and y0<y1:
            #print("C")
            for i in range(npts):
                xidx.append(x0 - i)
                yidx.append(y0 + i)
        elif x1>x0 and y0>y1:
            #print("D")
            for i in range(npts):
                xidx.append(x0 + i)
                yidx.append(y0 - i)

        xidx = np.array(xidx)
        yidx = np.array(yidx)
        #print("DIAG: ",xidx,yidx)
        coords = np.array([yidx,xidx]).T

        ##print(coords)
        for coord in coords:
            #print("coord: ",coord)
            grid[coord[0],coord[1]] += 1

#print(grid)

print()
print("Counting the grid!!!!!")
print()
tot = 0
for g in grid:
    #print(g)
    x = len(g[g>=2])
    #print(x)
    tot += x

print(tot)
