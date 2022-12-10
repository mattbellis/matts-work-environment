import sys
import numpy as np

from anytree import Node, RenderTree, AsciiStyle, Walker, PreOrderIter

infilename = sys.argv[1]
infile = open(infilename,'r')

'''
# Part A
X = 1
icount = 1
total = 0
for line in infile:
    #print(line)
    if line[0:4] == 'noop':
        icount += 1
        print(f"noop {icount} {line}")
        if icount==20 or (icount-20)%40==0:
            print(f"register: {icount} {X} {icount*X} ")
            total += icount*X
    elif line[0:4] == 'addx':
        increment = int(line.split()[1])
        for i in range(2):
            icount += 1
            print(f"addx {icount} {line}")
            if i==1: # Increment at the end of the cycle
                X += increment
            if icount==20 or (icount-20)%40==0:
                print(f"register: {icount} {X} {icount*X}")
                total += icount*X
        print(f"Added {increment} to make X {X}")
print(f"ENDING register: {icount} {X} {icount*X}")
print(f"total: {total}")
'''


# Part B
X = 1
icount = 1
total = 0
output = ""
for line in infile:
    #print(line)
    if icount>40:
        icount = 1
    if line[0:4] == 'noop':
        print(f"addx icount: {icount}   X: {X}        {line}")
        if icount == X or icount == X+1 or icount == X+2:
            output += "#"
        else:
            output += '.'
        icount += 1
        if (icount-1)%40==0:
            output += "\n"
        print("\n",output,"\n")
    elif line[0:4] == 'addx':
        increment = int(line.split()[1])
        for i in range(2):
            print(f"addx icount: {icount}   X: {X}        {line}")
            if icount == X or icount == X+1 or icount == X+2:
                output += "#"
            else:
                output += '.'
            if i==1: # Increment at the end of the cycle
                X += increment
            icount += 1
            if icount>40:
                icount = 1
            if (icount-1)%40==0:
                output += "\n"
            print("\n",output,"\n")
        #print(f"Added {increment} to make X {X}")
#print(f"ENDING register: {icount} {X} {icount*X}")
#print(f"total: {total}")
print(output)
print(len(output))

