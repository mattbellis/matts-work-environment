import sys
import numpy as np


infilename = sys.argv[1]
infile = open(infilename,'r')

def get_ranges(sections):
    idx1 = int(sections.split('-')[0])
    idx2 = int(sections.split('-')[1])

    return idx1,idx2


# Part A
'''
total = 0
for line in infile:
    sec1 = np.zeros(200,dtype=bool)
    sec2 = np.zeros(200,dtype=bool)

    elf1,elf2 = line.split(',')

    idx1,idx2 = get_ranges(elf1)
    sec1[idx1:idx2+1] = True
    idx1,idx2 = get_ranges(elf2)
    sec2[idx1:idx2+1] = True

    sec3 = sec1 | sec2

    if (sec3 == sec1).all() or (sec3 == sec2).all():
        total += 1
        print(elf1,elf2)

print(f"total: {total}")
'''

# Part B
total = 0
for line in infile:
    sec1 = np.zeros(200,dtype=bool)
    sec2 = np.zeros(200,dtype=bool)

    elf1,elf2 = line.split(',')

    idx1,idx2 = get_ranges(elf1)
    sec1[idx1:idx2+1] = True
    idx1,idx2 = get_ranges(elf2)
    sec2[idx1:idx2+1] = True

    sec3 = sec1 | sec2

    if len(sec3[sec3]) < len(sec1[sec1]) + len(sec2[sec2]):
        total += 1
        print(elf1,elf2)

print(f"total: {total}")
