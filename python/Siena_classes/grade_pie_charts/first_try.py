import numpy as np
import matplotlib.pylab as plt

fig, ax = plt.subplots(subplot_kw=dict(polar=True))

size = 0.3
vals = np.array([[60., 60.], [40., 40.], [29., 29.]])
#normalize vals to 2 pi
valsnorm = vals/np.sum(vals)*2*np.pi
#obtain the ordinates of the bar edges
valsleft = np.cumsum(np.append(0, valsnorm.flatten()[:-1])).reshape(vals.shape)

print(valsleft)
print(valsleft[:,0])
print(valsnorm.flatten())

cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

ax.bar(x=valsleft[:,0],
               width=valsnorm.sum(axis=1), bottom=0, height=1,
                      color='lightgrey', edgecolor='w', linewidth=1, align="edge")

ax.bar(x=valsleft[:, 0],
               width=valsnorm.sum(axis=1), bottom=0, height=[0.2, 0.5, 0.1],
                      color=outer_colors, edgecolor='w', linewidth=1, align="edge")
'''
ax.bar(x=valsleft.flatten(),
               width=valsnorm.flatten(), bottom=1-2*size, height=size,
                      color=inner_colors, edgecolor='w', linewidth=1, align="edge")
'''

ax.set(title="Pie plot with `ax.bar` and polar coordinates")
ax.set_axis_off()
plt.show()

