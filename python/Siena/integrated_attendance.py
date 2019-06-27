import numpy as np
import matplotlib.pylab as plt
import matplotlib.dates as mdates

import datetime as dt

import sys

infilenames = sys.argv[1:]
labels = ['SSU','MAC']

#plt.figure()

for i,infilename in enumerate(infilenames):
    vals = np.loadtxt(infilename,unpack=True,delimiter=",",dtype=str,skiprows=6)

    check_in_time = vals[6]

    print(check_in_time)

    t = []
    for c in check_in_time:

        clock,ampm = c.split()[1:3]
        hour,minute = clock.split(':')

        hour = int(hour)
        minute = int(minute)

        ampm = ampm.replace('"','')
        if ampm == 'PM' and hour!=12:
            hour += 12

        print(clock,ampm,hour)

        #t.append(dt.time(hour,minute).strftime('%X'))
        t.append(dt.datetime(2019,4,26,hour,minute))

    nstudents =len(check_in_time)

    students = np.arange(1,nstudents+1,1)

    print(nstudents,len(students))
    print(students)

    print(t)
    plt.figure()
    plt.plot(t,students,label=labels[i])
    plt.gcf().autofmt_xdate()
    plt.locator_params(axis='x', nbins=4)
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=30))   #to get a tick every 15 minutes
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))     #optional formatting 

    plt.xlabel('Time (4/26/2019)',fontsize=14)
    plt.ylabel('Integrated number of students',fontsize=14)

    plt.tight_layout()

    plt.legend()

    plt.savefig(labels[i]+'.png')

plt.show()
