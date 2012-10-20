import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy.interpolate import spline


fig = plt.figure(figsize=(6,6))
fig.canvas.set_window_title('A Boxplot Example')
ax1 = fig.add_subplot(111)
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.10)

ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

dates = ['06-07','07-08','08-09','09-10']
x = [0,1,2,3]
y = [1.0,3.6,6.0,7.5]

ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.0f m'))

xnew = np.linspace(min(x),max(x),300)
power_smooth = spline(x,y,xnew)

#ax1.plot(y,'y-',linewidth=15,solid_capstyle='round')
ax1.plot(xnew,power_smooth,'y-',linewidth=15,solid_capstyle='round')


plt.xticks(x,dates)
plt.xlim(-0.5,3.5)
plt.ylim(0.0,8.0)
plt.xticks(fontsize=24)
plt.yticks(fontsize=20)

fig.savefig('curca_plot.png')

plt.show()
