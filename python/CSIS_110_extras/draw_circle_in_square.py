import matplotlib.pylab as plt
import numpy as np

x = np.linspace(-1,1,1000)
y0 = np.sqrt(1.0 - x*x)  
y1 = -np.sqrt(1.0 - x*x)  

fig = plt.figure(figsize=(4,4),dpi=50)
ax = fig.add_subplot(1,1,1)
ax.plot(x,y0,'b-',linewidth=4)
ax.plot(x,y1,'b-',linewidth=4)
ax.grid(which='major')

plt.subplots_adjust(left=0.07,right=0.93,bottom=0.07,top=0.93)

plt.savefig('circle_in_square.png')

#plt.show()
