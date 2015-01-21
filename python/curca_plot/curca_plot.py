import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy.interpolate import spline


fig = plt.figure(figsize=(10,6))
fig.canvas.set_window_title('A Boxplot Example')
ax1 = fig.add_subplot(111)
plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)

ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#dates = ['06-07','07-08','08-09','09-10']
#x = [0,1,2,3]
#y = [0.9,4.9,6.8,7.5]

dates = ['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']
x = np.arange(0,len(dates),1)
y = [0.133189,0.488851,0.546971,0.864112,1.656295,1.750319,2.347459,4.799748,4.816397,7.471802,8.465154,8.880227,7.407593,7.053410]


print len(dates)
print len(x)
print len(y)

ax1.yaxis.set_major_formatter(FormatStrFormatter('\$%0.0f M'))

xnew = np.linspace(min(x),max(x),300)
power_smooth = spline(x,y,xnew)

#ax1.plot(y,'y-',linewidth=15,solid_capstyle='round')
ax1.plot(xnew,power_smooth,'-',color='orange',linewidth=15,solid_capstyle='round')


plt.xticks(x,dates,rotation=300)
plt.xlim(-0.5,14.5)
plt.ylim(0.0,9.5)
plt.xticks(fontsize=24)
plt.yticks(fontsize=20)
plt.title('Active award total',fontsize=24,weight='bold')

fig.savefig('curca_plot.png')

plt.show()
