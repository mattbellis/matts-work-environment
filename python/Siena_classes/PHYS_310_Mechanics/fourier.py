import matplotlib.pylab as plt
import numpy as np

x = np.linspace(-10,10,1000)

L = np.pi

y = np.zeros(len(x))

for i in range(1,7,2):

    coeff = 1.0/(L*i)
    
    y += coeff*np.sin((2*i*np.pi/L) * x)

    plt.plot(x,y)

plt.plot(x,y)
plt.show()
