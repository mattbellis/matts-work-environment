import sys
import numpy as np

infilename = sys.argv[1]

# Part A
#data = np.loadtxt(infilename,dtype=int,unpack=True)
#print(data)
'''
infile = open(infilename,'r')
max_cal = 0
elf_cal = 0
for row in infile:
    #print(row)
    if row != "" and row != '\n':
        print(int(row))
        elf_cal += int(row)
    else:
        print("New elf!")
        print(max_cal, elf_cal)
        if max_cal < elf_cal:
            max_cal = elf_cal
        elf_cal = 0
# Get the last elf
if max_cal < elf_cal:
    max_cal = elf_cal

print(f"max cal: {max_cal}")
'''

#exit()


# Part B
#data = np.loadtxt(infilename,dtype=int,unpack=True)
#print(data)
infile = open(infilename,'r')
calories = []
elf_cal = 0
for row in infile:
    #print(row)
    if row != "" and row != '\n':
        print(int(row))
        elf_cal += int(row)
    else:
        print("New elf!")
        print(elf_cal)
        calories.append(elf_cal)
        elf_cal = 0
# Get the last elf
calories.append(elf_cal)
print(calories)

calories.sort()

print(calories)
print(calories[-3:])
print(sum(calories[-3:]))
