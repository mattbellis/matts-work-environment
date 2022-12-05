import sys
import numpy as np


infilename = sys.argv[1]
infile = open(infilename,'r')


# Part A
# Read in the input file
# Testing
#ncols = 3
# Full dataset
ncols = 9

columns = []
for i in range(ncols):
    columns.append([])

instructions = {"num":[], "from":[], "to":[]}

for line in infile:
    nchars = len(line)
    print(nchars,line)
    if line.find('[')>=0:
        for i in range(ncols):
            idx = 4*i + 1
            c = line[idx]
            print(c)
            if c != ' ':
                columns[i].append(c)
    elif line.find('move')>=0:
        vals = line.split()
        instructions['num'].append(int(vals[1]))
        instructions['from'].append(int(vals[3]))
        instructions['to'].append(int(vals[5]))
        
print(columns)
print(instructions)

# Part A
# Do the instructions
'''
print("\nStarting the move\n")
print(columns)
ninstructions = len(instructions['num'])

for n in range(ninstructions):
    num = instructions['num'][n]
    f = instructions['from'][n] - 1 # Substract 1 because the columns start at 1
    t = instructions['to'][n] - 1 # Substract 1 because the columns start at 1

    for i in range(num):
        val = columns[f].pop(0)
        columns[t].insert(0,val)
        print(columns)

print(columns)
output = ""
for col in columns:
    output += col[0]

print(f"Top layer: {output}")
'''


# Part B
# Do the instructions
print("\nStarting the move\n")
print(columns)
ninstructions = len(instructions['num'])

for n in range(ninstructions):
    num = instructions['num'][n]
    f = instructions['from'][n] - 1 # Substract 1 because the columns start at 1
    t = instructions['to'][n] - 1 # Substract 1 because the columns start at 1

    vals_to_move = columns[f][0:num]
    print(vals_to_move)
    for i in range(num):
        val = columns[f].pop(0)
    for i in range(len(vals_to_move)-1,-1,-1):
        columns[t].insert(0,vals_to_move[i])
    print(columns)

print(columns)
output = ""
for col in columns:
    output += col[0]

print(f"Top layer: {output}")




