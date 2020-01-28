import numpy as np
import matplotlib.pylab as plt

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

################################################################################
nplaces = 8
degrees = np.pi/2/nplaces

vals = [np.pi/2.]

for i in range(nplaces):
    vals.append(degrees)
vals.append(np.pi)

vals = np.array(vals)
################################################################################

size = 0.3

fig, ax = plt.subplots(dpi=200)

# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
cmap = plt.get_cmap("tab20c")
#outer_colors = cmap([16,17,18,19])
outer_colors = cmap([16,17,18])

cmap = plt.get_cmap("Dark2")
outer_colors = cmap([1,2])

# https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_hatch
hatches = ['///','o','\\','O']
#hatches = 'o'

#cmap = plt.get_cmap("Dark2")
#outer_colors = cmap([1,2,3])

patches = ax.pie(vals, radius=1, colors=outer_colors, wedgeprops=dict(width=size, edgecolor='w'))
for i,p in enumerate(patches[0]):
    p.set_hatch(hatches[i%len(hatches)])


ax.set(aspect="equal")
ax.set_ylim(0.15,1.0)
ax.set_xlim(-1.0,-0.45)

plt.show()

plt.savefig('wedge.png',transparent=True)
