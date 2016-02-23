import numpy as np
import matplotlib.pylab as plt

import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)

p = patches.Ellipse((0,0.5), 1.4, 0.4,facecolor='gray',alpha=0.8)
ax.add_patch(p)

p = patches.Rectangle((-0.7,-0.5), 1.4, 1.0,facecolor='black',alpha=1.0)#, ec="none")
ax.add_patch(p)

p = patches.Rectangle((-0.7+0.05,-0.5+0.05), 1.4-0.10, 1.0-0.10,facecolor='yellow',alpha=1.0)#, ec="none")
ax.add_patch(p)

p = patches.Rectangle((-0.7+0.10,-0.5+0.10), 1.4-0.20, 1.0-0.20,facecolor='white',alpha=1.0)#, ec="none")
ax.add_patch(p)

p = patches.Arc((0,-0.5), 1.4, 0.4,facecolor='gray',alpha=0.4,theta1=30,theta2=150,linewidth=4)
ax.add_patch(p)


plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(-0.75,0.75)
ax.set_ylim(-0.55,0.75)
#plt.axis('equal')
plt.axis('off')

plt.savefig('cogent_cutaway_0.png')

plt.show()
