import numpy as np
import matplotlib.pylab as plt

#xpts = np.linspace(0,1)
#ypts = np.linspace(0,1)

xpts = np.array([])
ypts = np.array([])

for i in range(0,40):
    for j in range(0,40):
        xpts = np.append(xpts,i)
        ypts = np.append(ypts,j)

p = plt.plot(xpts,ypts,'o')

frame1 = plt.gca()

frame1.axes.get_xaxis().set_visible(False)

frame1.axes.get_yaxis().set_visible(False)

plt.show()

