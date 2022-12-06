import sys
import numpy as np


infilename = sys.argv[1]
infile = open(infilename,'r')

'''
# Part A
def isdiff(chunk):
    b1 = chunk[0] != chunk[1]
    b2 = chunk[0] != chunk[2]
    b3 = chunk[0] != chunk[3]
    b4 = chunk[1] != chunk[2]
    b5 = chunk[1] != chunk[3]
    b6 = chunk[2] != chunk[3]

    if b1 and b2 and b3 and b4 and b5 and b6:
        return True
    else:
        return False


for line in infile:
    nchars = len(line)
    print(line)
    for lo in range(0,nchars-4):
        hi = lo+4
        chunk = line[lo:hi]
        print(chunk)
        if isdiff(chunk):
            print(hi)
            break
'''

# Part B
def isdiff(chunk):
    nchars = len(chunk)
    arr = np.zeros(26,dtype=int)
    for i in range(nchars):
        arr[ord(chunk[i])-97] += 1
    #print(arr)
    if len(arr[arr>0])==nchars:
        return True
    else:
        return False



for line in infile:
    nchars = len(line)
    print(line)
    for lo in range(0,nchars-14):
        hi = lo+14
        chunk = line[lo:hi]
        print(chunk)
        if isdiff(chunk):
            print(hi)
            break
