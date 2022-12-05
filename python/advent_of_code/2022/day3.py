import sys
import numpy as np


def priority(x):
    if x == '\n':
        return -1 

    idx = ord(x)
    if idx<97:
        idx -= 38
    else:
        idx -= 96
    return idx


infilename = sys.argv[1]
infile = open(infilename,'r')

# Part A
'''
total = 0
for line in infile:
    nchars = len(line)
    middle = int(nchars/2)
    half1 = line[0:middle]
    half2 = line[middle:]
    print(half1)
    print(half2)
    print()

    arr1 = np.zeros(52, dtype=bool)
    arr2 = np.zeros(52, dtype=bool)
    for x in half1:
        idx = priority(x) - 1
        print(x,idx)
        arr1[idx] = True
    for x in half2:
        idx = priority(x) - 1
        print(x,idx)
        arr2[idx] = True
    print(arr1)
    print(arr2)

    arr3 = arr1 & arr2
    print(arr3)

    priority_idx = arr3.tolist().index(True)
    
    total += priority_idx+1
    print(total,priority_idx)

print(f"total: {total}")
'''

# Part B
#'''

def parse_elf_bag(line):

    arr1 = np.zeros(52, dtype=int)
    for x in line:
        if priority(x)<0:
            continue
        idx = priority(x) - 1
        print(x,idx)
        arr1[idx] = 1
    print(arr1)
    return arr1


total = 0
group_counter = 0
elves = []
for line in infile:
    print(f"group_counter: {group_counter}")

    if group_counter<2:
        elves.append(line)
        group_counter += 1
    else:
        print("TESTING!!!!!!!!!!!!!!!")
        elves.append(line)
        print(elves)

        total_arr = np.zeros(52,dtype=int)
        for elf in elves:
            total_arr += parse_elf_bag(elf)

        print("Total arr: ")
        print(total_arr)
        priority_idx = total_arr.tolist().index(3)

        elves = []
        group_counter = 0

        total += priority_idx+1
        print(f"total: {total}    priority_idx: {priority_idx}")

print(f"total: {total}")
#'''





