import numpy as np
import matplotlib.pylab as plt
import XKCD_plots as xkcd

x0 = np.array([])
y0 = np.array([])
infile = open('constrainedList.tsv','r')
for line in infile:
    vals = line.split()
    x0 = np.append(x0,float(vals[0]))
    y0 = np.append(y0,float(vals[1]))

x1 = np.array([])
y1 = np.array([])
infile = open('unconstrainedList.tsv','r')
for line in infile:
    vals = line.split()
    x1 = np.append(x1,float(vals[0]))
    y1 = np.append(y1,float(vals[1]))

np.random.seed(0)

fig = plt.figure(figsize=(10,6))
ax = plt.axes()

# The lines
xpt = np.linspace(0, 180, 100)
ypt = 0.05*np.ones(100)
ax.plot(xpt, ypt, 'b--', lw=0.5)
ypt = 0.32*np.ones(100)
ax.plot(xpt, ypt, 'r--', lw=0.5)
ypt = 1.00*np.ones(100)
ax.plot(xpt, ypt, 'k', lw=0.5)

#x = np.linspace(0, 180, 100)
#ax.plot(x, np.exp(-((x-90)**2)/120), 'b', lw=1, label='BaBar (2012)')
ax.plot(x1, y1, 'k--', lw=2, label='BaBar (2012) - unconstrained')
ax.plot(x0, y0, 'r', lw=1, label='BaBar (2012) - isospin constrained')

ax.set_title('Likelihood scan of $\\alpha$')
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$\\Sigma$')

ax.legend(loc='lower right')

ax.text(10.05, 0.7, "If I never see another\nrobustness study,\nit will be too soon!")

#ax.set_xlim(0, 10)
ax.set_ylim(-0.3, 1.3)

#XKCDify the axes -- this operates in-place
xkcd.XKCDify(ax, xaxis_loc=0.0, yaxis_loc=1.0, xaxis_arrow='+-', yaxis_arrow='+-', expand_axes=True)

plt.show()
