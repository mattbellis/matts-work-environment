import sys
import numpy as np

from anytree import Node, RenderTree, AsciiStyle, Walker, PreOrderIter

import copy

from collections import deque

infilename = sys.argv[1]
infile = open(infilename,'r')

monkeys = []
monkey_num = -1

# Part A
map = []
start_coord = None
end_coord = None
icount = 0
for line in infile:
    nchars = len(line)
    row = []
    for i in range(nchars):
        c = line[i]
        oc = ord(c)
        #print(oc, c)
        if oc >= 97:
            row.append(oc-97)
        elif c == 'S':
            row.append(0)
            start_coord = (icount, i)
        elif c == 'E':
            row.append(25)
            end_coord = (icount, i)
    icount += 1

    map.append(row)

################################################################################
# Maybe hints?
# https://stackoverflow.com/questions/47896461/get-shortest-path-to-a-cell-in-a-2d-array-in-python
################################################################################
def walk(grid, start):
    queue = deque([[start]])
    seen = set([start])
    nrow = len(grid)
    ncol = len(grid[0])
    print(f"nrow: {nrow}    ncol: {ncol}")

    while queue:
        #print("queue, path, queue")
        #print(queue)
        #print(len(queue))
        print("\nIn while...............")
        path = queue.popleft()
        print(path)
        print(f"LEN PATH: {len(path)}")
        #print(queue)
        x,y = path[-1]
        print(f"CHECKING: {(x,y)}  {end_coord}")
        if x == end_coord[0] and y == end_coord[1]:
            #if path[-1] == end_coord:
            print(f"MADE IT!!!!!!!!!!!!!!!!    {len(path)}")
            return path
        
        val = grid[y][x]
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            print(f"LOOPING HERE: {x} {y} {x2} {y2}")
            if 0 <= x2 < ncol and 0 <= y2 < nrow and (x2,y2) not in seen:
                val2 = grid[y2][x2]
                print(f"VALUES: {val} {val2}    {x} {y}    {x2} {y2}")
                if val2 <= val + 1:
                    print("VALUES STEP ----")
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))

    print(f"AT THE END OUTSIDE THE WHILE")

##########################################################################
map = np.array(map)
output = ""
for i,row in enumerate(map):
    #print(row)
    output += f"{i:2d}   "
    for r in row:
        output += f"{r:2d} "
    output += "\n"
print(output)
print()
print(start_coord)
print(end_coord)
print()

nstep = walk(map, start=start_coord)
print(nstep)
