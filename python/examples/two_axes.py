import numpy as np
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from datetime import datetime,timedelta

start_date = datetime(2009, 12, 3)

x2 = []
ndivs = 5
for i in range(ndivs):
    days = 200/ndivs
    date = start_date + timedelta(days=i*days)
    x2.append(date)
date = start_date + timedelta(days=(i+1)*days)
x2.append(date)
y = range(len(x2)) # many thanks to Kyss Tao for setting me straight here

print x2


# plot f(x)=x for two different x ranges
x1 = np.linspace(1, 200, 50)
y1 = np.linspace(1, 200, 50)
#x2 = np.linspace(1, 200, 50)
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(x1, y1,'b--')

ax2 = ax1.twiny()
ax2.plot(x2, np.ones(len(x2)), 'go',alpha=0)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator())

fig.autofmt_xdate()


plt.show()
