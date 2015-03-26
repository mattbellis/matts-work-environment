import numpy as np
import matplotlib.pylab as plt

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

fig = plt.figure(figsize=(11.5,8),dpi=100)
ax = plt.subplot(1,1,1)

# This is a comment
energies = [0.2,0.7,0.8]

for e in energies:
    line = mlines.Line2D((e,e), (0.0,1.), lw=2., alpha=0.4,color='red',markeredgecolor='red')
    ax.add_line(line)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout(h_pad=0.0, w_pad=0.0, pad=0.0)
plt.savefig('image.png')
plt.show()
