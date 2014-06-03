import sys
import matplotlib.pylab as plt
import numpy as np

infilename = sys.argv[1]
nsig0 = float(sys.argv[2])

vals = np.loadtxt(infilename,dtype='str')
newvals = vals.swapaxes(0,1)
print newvals
x = newvals[1].astype(float)
xerr = newvals[2].astype(float)

pull = (x-nsig0)/xerr

plt.figure()
plt.hist(pull,bins=25)

plt.figure()
plt.hist(x,bins=25)

plt.show()
