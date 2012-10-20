import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

fig = plt.figure(figsize=(6,10))
fig.canvas.set_window_title('A Boxplot Example')
ax1 = fig.add_subplot(111)
plt.subplots_adjust(left=0.15, right=0.95, top=0.98, bottom=0.10)

ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

dates = ['06-07','07-08','08-09','09-10']
x = [0,1,2,3]
y = [1.0,3.6,6.0,7.5]

ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.0f m'))
ax1.plot(y,'y-',linewidth=5)
plt.xticks(x,dates)
plt.xlim(-0.2,3.2)
plt.ylim(0.0,8.0)
plt.xticks(fontsize=24)
plt.yticks(fontsize=20)

plt.show()
