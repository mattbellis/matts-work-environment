#!/usr/bin/env python

import numpy as n
from pylab import figure, show
import matplotlib.cm as cm
import matplotlib.colors as colors

fig = figure()
ax = fig.add_subplot(111)
Ntotal = 1000
N, bins, patches = ax.hist(n.random.rand(Ntotal), 20)

#I'll color code by height, but you could use any scalar


# we need to normalize the data to 0..1 for the full
# range of the colormap
fracs = N.astype(float)/N.max()
norm = colors.normalize(fracs.min(), fracs.max())

for thisfrac, thispatch in zip(fracs, patches):
    color = cm.jet(norm(thisfrac))
    thispatch.set_facecolor(color)



show()

batchMode = False
## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if (not batchMode):
  if __name__ == '__main__':
    rep = ''
    while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
        rep = rep[0]
