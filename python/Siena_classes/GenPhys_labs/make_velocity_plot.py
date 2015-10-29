import matplotlib.pylab as plt
import numpy as np
import seaborn as sn

x = [0,1,2,3,4,5, 6, 7, 8, 9, 10]
y = [0,1,2,3,4,9,14, 14.5,15,15.5,16.0]

plt.plot(x,y)
plt.plot(x,y,'bo')
plt.xlabel('Time (seconds)',fontsize=24)
plt.ylabel('Position (meters)',fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.ylim(0,20)
plt.plot(x,y,'bo',markersize=50)
plt.gca().text(0.2,0.7,'A',fontsize=24)
plt.gca().text(4.5,4,'B',fontsize=24)
plt.gca().text(6,14.5,'C',fontsize=24)
plt.gca().text(9.5,17,'D',fontsize=24)
plt.tight_layout()

plt.savefig('velocity.png')

plt.show()
