import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

import seaborn as sn

import sys

sn.set_context(rc={'lines.markeredgewidth': 0.1})

infilename = sys.argv[1]

vals = np.loadtxt(infilename,unpack=True,skiprows=2,delimiter=',',dtype='str')

year = vals[0]
tot = vals[1].astype(int)
women = vals[2].astype(int)
men = tot - women

print year
print tot
print women

x = np.arange(0,len(year),1)


fig = plt.figure(figsize=(10,6))
#fig.canvas.set_window_title('A Boxplot Example')
ax1 = fig.add_subplot(111)
#plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
#ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
#ax1.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)


#ax1.yaxis.set_major_formatter(FormatStrFormatter('\$%0.0f M'))

ax1.set_ylabel('# of students',fontsize=18)

plt.xticks(x,year,rotation=300)
plt.xlim(-0.5,18.5)
#plt.ylim(0.0,9.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.title('Active award total',fontsize=24,weight='bold')

plt.plot(x,men,'bs',markersize=15,label='# of Male Physics majors')
plt.plot(x,women,'ro',markersize=15,label='# of Female Physics majors')

plt.tight_layout()

plt.legend(loc='upper left',fontsize=24)

fig.savefig('dept_plot.png')

sn.plt.show()

