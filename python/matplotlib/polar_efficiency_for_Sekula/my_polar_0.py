#!/usr/bin/env python

import numpy as np
import matplotlib.cm as cm
from matplotlib.pyplot import figure, show, rc

import sys


# Pass in a text file of angle/eff
inputfile = open(sys.argv[1])

# force square figure and square axes looks better for polar, IMO
fig = figure(figsize=(8,4))
ax = fig.add_axes([0.1, -0.75, 0.8, 1.6], projection='polar')

# For debugging
#ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar')

theta = []
radii = []
width = []
width_val = np.pi / 100.0
for line in inputfile:
  # Don't read commented lines
  if line[0] != '#':
    rad = float(line.split()[0])
    eff = float(line.split()[1])
    print "%3.3f %3.3f" % ( rad, eff)

    theta.append( rad )
    radii.append(eff)
    width.append( width_val )

# Make the bars/wedges
bars = ax.bar(theta, radii, width=width, bottom=0.0)
for r,bar in zip(radii, bars):
  # Set a color based on angle (I think)
  bar.set_facecolor( cm.jet(r/10.))
  # Set the transparency
  bar.set_alpha(0.5)

show()
