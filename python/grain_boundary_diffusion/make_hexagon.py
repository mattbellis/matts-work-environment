import matplotlib.pylab as plt
import numpy as np

x = np.array([])
y = np.array([])

x = np.array([4,3,1,0,1,3])
y = np.array([2,0,0,2,4,4])

'''
for i in range(100):
    for j in range(100):
        if j<80 and j>20 and i==20:
            x = np.append(x,i)
            y = np.append(y,j)
        if i>20 and i<60 and j>80 and j<110:
            x = np.append(x,i)
            y = np.append(y,i+60)
'''


plt.plot(x,y,'-')
#plt.ylim(0,150)
#plt.xlim(0,150)
plt.ylim(-1,5)
plt.xlim(-1,5)
plt.show()
