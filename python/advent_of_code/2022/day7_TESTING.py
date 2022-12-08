import sys
import numpy as np

from anytree import Node, RenderTree, AsciiStyle, Walker, PreOrderIter

infilename = sys.argv[1]
infile = open(infilename,'r')

level = 0
icount = 0
for line in infile:
    icount += 1
    if line.find('$')>=0:
        is_command = True
        is_file = False
        if line[2:4] == 'ls':
            1
        elif line[2:4] == 'cd':
            if line.find('/')>=0:
                level = 0
            elif line.find('..')>=0:
                level -= 1
            else:
                level += 1
    print(f"LINE: {icount}   LEVEL {level}  {line[:-1]}")
