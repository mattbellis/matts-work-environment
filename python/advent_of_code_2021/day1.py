import numpy as np
import sys

testdata = [199,
        200,
        208,
        210,
        200,
        207,
        240,
        269,
        260,
        263]

testdata = np.array(testdata)

# Part 1
diff = testdata[1:] - testdata[:-1]
print(diff)
print(len(diff[diff>0]))


infilename = sys.argv[1]
data = np.loadtxt(infilename,unpack=True,dtype=int)

print(data)

diff = data[1:] - data[:-1]
print(diff)
print(len(diff[diff>0]))

# Part 2
print("Part 2-----------")
sums = testdata[0:-2] + testdata[1:-1] + testdata[2:]
print(testdata)
print(sums)
diff = sums[1:] - sums[:-1]
print(diff)
print(len(diff[diff>0]))


sums = data[0:-2] + data[1:-1] + data[2:]
print(data)
print(sums)
diff = sums[1:] - sums[:-1]
print(diff)
print(len(diff[diff>0]))
