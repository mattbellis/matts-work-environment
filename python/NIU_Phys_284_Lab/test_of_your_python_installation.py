import numpy as np
import pylab as plt

x = [0,2,3,5,9]
y = [0,4,9,25,81]

print "mean of x: %f" % (np.mean(x))
print "std. dev. of y: %f" % (np.std(y))

plt.plot(x,y,'o')
plt.show()

