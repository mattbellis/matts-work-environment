import numpy as np
import matplotlib.pylab as plt

import sys

infile = open(sys.argv[1])

content = np.array(infile.read().split()).astype('float')
ncolumns = 2
nvals = len(content)

index = np.arange(0,nvals,ncolumns)

x = content[index]
y = content[index+1]

plt.plot(x,y)
plt.ylim(-1.0,5.0)
plt.show()
