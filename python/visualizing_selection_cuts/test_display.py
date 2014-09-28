from jets import *
from mpl_toolkits.mplot3d import Axes3D
from draw_objects3D import *
import numpy as np
import mpl_toolkits.mplot3d.art3d as a3

from cms_tools import get_collisions

#fig = plt.figure(figsize=(7,5),dpi=100)
#ax = fig.add_subplot(1,1,1)
#ax = fig.gca(projection='3d')
#plt.subplots_adjust(top=0.98,bottom=0.02,right=0.98,left=0.02)



f = open('small_test_file.dat')
collisions = get_collisions(f)

#lines = []
#lines += display_collision3D(collisions[0])
display_collision3D(collisions[0])

'''
for l in lines:
    ax.add_line(l)

ax.set_xlim(-300,300)
ax.set_ylim(-300,300)
ax.set_zlim(-300,300)
'''

plt.show()




