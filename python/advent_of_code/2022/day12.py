import sys
import numpy as np

from anytree import Node, RenderTree, AsciiStyle, Walker, PreOrderIter

import copy

infilename = sys.argv[1]
infile = open(infilename,'r')

monkeys = []
monkey_num = -1

# Part A
map = []
for line in infile:
    nchars = len(line)
    row = []
    for i in range(nchars):
        c = line[i]
        oc = ord(c)
        print(oc, c)
        if oc >= 97:
            row.append(oc-97)
        elif c == 'S':
            row.append(0)
        elif c == 'E':
            row.append(25)

    map.append(row)

def walk(map, coord=(0,0), nstep=0):
    nrow = len(map)
    ncol = len(map[0])
    x,y = coord
    val = map[y][x]
    
    for i in range(0,4):
        if i==0: # Step right
            if x<ncol-1:
                x1 = x+1
                y1 = y
                nval = map[y1][x1]
                print(f"RIGHT  current: {y} {x} {val}     step: {y1} {x1} {nval}")
                if nval <= val+1:
                    nstep += 1
                    nstep += walk(map, coord=(x1,y1), nstep=nstep)
        if i==1: # Step left
            if x>0:
                x1 = x-1
                y1 = y
                print(f"LEFT   current: {y} {x} {val}     step: {y1} {x1} {nval}")
                if nval <= val+1:
                    nstep += 1
                    nstep += walk(map, coord=(x1,y1), nstep=nstep)
        if i==2: # Step down
            if y<nrow-1:
                x1 = x
                y1 = y+1
                print(f"DOWN   current: {y} {x} {val}     step: {y1} {x1} {nval}")
                if nval <= val+1:
                    nstep += 1
                    nstep += walk(map, coord=(x1,y1), nstep=nstep)
        if i==3: # Step up
            if y>0:
                x1 = x
                y1 = y-1
                print(f"UP     current: {y} {x} {val}     step: {y1} {x1} {nval}")
                if nval <= val+1:
                    nstep += 1
                    nstep += walk(map, coord=(x1,y1), nstep=nstep)

    return nstep


map = np.array(map)
for row in map:
    print(row)
print()

nstep = walk(map, coord=(0,0), nstep=0)
print(nstep)
