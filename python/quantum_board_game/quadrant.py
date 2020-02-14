import numpy as np
import matplotlib.pylab as plt

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

################################################################################
# For one quadrant
################################################################################
#nplaces = 8
#degrees = np.pi/2/nplaces
#
#vals = [np.pi/2.]
#
#for i in range(nplaces):
    #vals.append(degrees)
#vals.append(np.pi)
#
#vals = np.array(vals)
################################################################################
################################################################################
# For full circle
################################################################################
nplaces = 48
degrees = 2*np.pi/nplaces

vals = []

for i in range(nplaces):
    vals.append(degrees)
    #vals.append(0.05) # To make transparent ones
#vals.append(np.pi)

vals = np.array(vals)
print(vals)
################################################################################


size = 0.3

fig, ax = plt.subplots(dpi=600)

# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
################################################################################
# For quadrant
################################################################################
#cmap = plt.get_cmap("tab20c")
##outer_colors = cmap([16,17,18,19])
#outer_colors = cmap([16,17,18])
#
#cmap = plt.get_cmap("Dark2")
#outer_colors = cmap([1,2])
################################################################################
################################################################################
# For full circle
################################################################################
#cmap = plt.get_cmap("Spectral")
cmap = plt.get_cmap("hsv")
outer_colors = cmap(np.arange(0,256,int(256/nplaces)))
print(outer_colors)
print(np.arange(0,256,int(256/nplaces)))
#outer_colors = cmap([16,17,18])

################################################################################

# https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_hatch
hatches = ['///','o','\\','O']
#hatches = 'o'

patches = ax.pie(vals, radius=1, colors=outer_colors, wedgeprops=dict(width=size, edgecolor='k'))
#patches = ax.pie(vals, radius=1, colors=outer_colors, wedgeprops=dict(width=size, edgecolor='k',linewidth=3))
#patches = ax.pie(vals, radius=1, colors=outer_colors, wedgeprops=dict(width=size, edgecolor=(1,1,1,0.5),linewidth=2))
# For quadrant
#patches = ax.pie(vals, radius=1, colors=outer_colors, wedgeprops=dict(width=size, edgecolor='w'))
# For hatches
#for i,p in enumerate(patches[0]):
    #p.set_hatch(hatches[i%len(hatches)])


################################################################################
# To get just a quadrant
################################################################################
ax.set(aspect="equal")
#ax.set_ylim(0.15,1.0)
#ax.set_xlim(-1.0,-0.45)


plt.savefig('wedge.png',transparent=True)

plt.show()

