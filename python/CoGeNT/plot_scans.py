import numpy as np
import matplotlib.pylab as plt

import sys

infilename = sys.argv[1]
infile = open(infilename)

tag = "default"
if len(sys.argv)>2:
    tag = sys.argv[2]

x = []
y = []
z = []

for line in infile:
    vals = line.split()
    x.append(float(vals[0]))
    y.append(float(vals[1]))
    z.append(float(vals[2]))

plt.figure()
plt.plot(x,y,'o')
plt.xscale('log')
plt.ylabel('WIMP mass (GeV)')
plt.xlabel('WIMP-nucleon cross-section')
name = "Plots/xsec_vs_mass_%s.png" % (tag)
plt.savefig(name)

plt.figure()
plt.plot(x,z,'o')
plt.xscale('log')
plt.xlabel('WIMP-nucleon cross-section')
plt.ylabel('-log(lh)')
name = "Plots/lh_vs_xsec_%s.png" % (tag)
plt.savefig(name)

plt.figure()
plt.plot(y,z,'o')
plt.xlabel('WIMP mass (GeV)')
plt.ylabel('-log(lh)')
name = "Plots/lh_vs_mass_%s.png" % (tag)
plt.savefig(name)

plt.show()
