import numpy as np
import sys

testdata = [['forward','5'],
        ['down','5'],
        ['forward','8'],
        ['up','3'],
        ['down','8'],
        ['forward','2'],
        ]

testdata = np.array(testdata)

infilename = sys.argv[1]
data = np.loadtxt(infilename,unpack=False,dtype=str)

print(data)

def move_sub(data):
    position = np.array([0,0])

    for move in data:
        #a,b = move.split()[0],int(move.split()[1])
        a,b = move[0],int(move[1])
        if a=='forward':
            position[0] += b
        elif a=='down':
            position[1] += b
        elif a=='up':
            position[1] -= b

    print(position)
    print(position[0]*position[1])

move_sub(testdata)

move_sub(data)

################################################################################
# Part 2

def move_sub2(data):
    # x, depth, aim
    position = np.array([0,0,0])

    for move in data:
        #a,b = move.split()[0],int(move.split()[1])
        a,b = move[0],int(move[1])
        if a=='forward':
            position[0] += b
            position[1] += (position[2]*b)
        elif a=='down':
            position[2] += b
        elif a=='up':
            position[2] -= b

    print(position)
    print(position[0]*position[1])

print("Day 2")

move_sub2(testdata)

move_sub2(data)

################################################################################
# Part 2

