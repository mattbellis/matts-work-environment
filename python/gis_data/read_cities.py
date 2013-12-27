import matplotlib.pylab as plt
import numpy as np

import sys

import csv

reader = csv.reader(open(sys.argv[1]))

x = []
y = []
for i,row in enumerate(reader):

    if i!=0:
        x.append(float(row[12]))
        y.append(float(row[11]))

        print i

x = np.array(x)
y = np.array(y)

plt.plot(x,y,'o',markersize=20)
plt.show()

