import numpy as np
import matplotlib.pylab as plt
import XKCD_plots as xkcd

np.random.seed(0)

ax = plt.axes()

x = np.linspace(0, 180, 100)
ax.plot(x, np.exp(-((x-90)**2)/120), 'b', lw=1, label='BaBar (2012)')

ax.set_title('Likelihood scan of $\\alpha$')
ax.set_xlabel('$\\alpha$')
ax.set_ylabel('$\\Sigma$',fontsize=24)

ax.legend(loc='lower right')

ax.text(10.05, 0.1, "This was\na lot\nof work!")

#ax.set_xlim(0, 10)
#ax.set_ylim(-1.0, 1.0)

#XKCDify the axes -- this operates in-place
xkcd.XKCDify(ax, xaxis_loc=0.0, yaxis_loc=1.0, xaxis_arrow='+-', yaxis_arrow='+-', expand_axes=True)

plt.show()
