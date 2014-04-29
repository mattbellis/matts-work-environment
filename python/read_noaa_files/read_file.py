import numpy as np
import matplotlib.pylab as plt

import datetime as dt

import sys

plt.figure()
for filename in sys.argv[1:]:

    print filename

    #times,hours,temp = np.loadtxt(filename, dtype=str, delimiter=',', usecols=(1,2,10), skiprows=6, unpack=True)
    times,hours,temp = np.loadtxt(filename, dtype=str, delimiter=',', usecols=(1,2,30), skiprows=6, unpack=True)

    print times
    print temp

    times = times[temp!='M']
    hours = hours[temp!='M']
    temp = temp[temp!='M']

    times = times[temp!=' ']
    hours = hours[temp!=' ']
    temp = temp[temp!=' ']

    date = []
    for time,hour in zip(times,hours):
        date.append(dt.datetime(year=int(time[0:4]),month=int(time[4:6]),day=int(time[6:8]),hour=int(hour[0:2]),minute=int(hour[2:])))

    plt.plot(date,temp,'ko',markersize=2)

plt.show()

