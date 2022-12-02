import numpy as np
import sys

################################################################################

infilename = sys.argv[1]

positions = np.loadtxt(infilename,delimiter=',',unpack=False,dtype=int)

print("positions -------------")
print(positions)
print()

diffs = []
sum_diffs = []
for p in positions:
    diff = np.abs(positions-p)
    diffs.append(diff)
    #print(diff)
    print(np.sum(diff))
    sum_diffs.append(np.sum(diff))

best = np.min(sum_diffs)
print("best: ",best)
idx = sum_diffs.index(best)
print("idx: ",idx)
print("idx type: ",type(idx))
print(positions[idx])

print("Part 2 ==========================================")
print()
print("positions -------------")
print(positions)
print()

def series(n):
    #print(f"In here: {n}")
    tot = 0
    for i in range(1,n+1):
        tot += i
    return tot


diffs = []
sum_diffs = []
lo = min(positions)
hi = max(positions)
for p in range(lo,hi+1):
    diff = np.abs(positions-p)
    #diffs.append(diff)
    #print(diff)
    mysum = np.zeros(len(positions))
    for i in range(len(mysum)):
        n = series(diff[i])
        mysum[i] = n
    #print(mysum)

    print(lo,hi,p,np.sum(mysum))
    sum_diffs.append(np.sum(mysum))

best = np.min(sum_diffs)
print("best: ",best)
idx = sum_diffs.index(best)
print("idx: ",idx)
print("idx type: ",type(idx))
print(positions[idx])
