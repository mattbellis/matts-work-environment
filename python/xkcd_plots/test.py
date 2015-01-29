import numpy as np
import matplotlib.pylab as plt
import XKCD_plots as xkcd

np.random.seed(0)

ax = plt.axes()

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(2.0*x), 'b', lw=1, label='undamped sine')
ax.plot(x, np.sin(2.0*x) * np.exp(-0.2 * x ), 'r', lw=1, label='damped sine')

ax.set_title('Simple harmonic oscillator')
ax.set_xlabel('time')
ax.set_ylabel('displacement')

ax.legend(loc='lower right')

ax.set_xlim(0, 10)
ax.set_ylim(-1.0, 1.0)

#XKCDify the axes -- this operates in-place
xkcd.XKCDify(ax, xaxis_loc=0.0, yaxis_loc=1.0, xaxis_arrow='+-', yaxis_arrow='+-', expand_axes=True)

plt.show()
