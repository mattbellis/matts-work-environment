import numpy as np
import matplotlib.pylab as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

import datetime as dt

import sys

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1,1,1)

#ax.xaxis.set_major_formatter(DateFormatter('%m/%d/%Y %H:%M'))
ax.xaxis_date()

infile = open(sys.argv[1],'r')

left_breast_feeding = []
right_breast_feeding = []

first_l = True
first_r = True
first_pee = True
first_poop = True

lbf_y = 1.6
lbf_col = 'b'
lbf_width = 8

rbf_y = 1.4
rbf_col = 'r'
rbf_width = 8

dayofyear = '10/13/2013'

orgdate = None
enddate = None

for i,line in enumerate(infile):

    if i>0:
        vals = line.split(',')

        print vals

        if vals[0] != '':
            dayofyear = vals[0]
            print dayofyear

            year = int(dayofyear.split('/')[2])
            month = int(dayofyear.split('/')[0])
            day = int(dayofyear.split('/')[1])

            date = dt.datetime(year=year,month=month,day=day,hour=0,minute=0)
            ax.plot([date,date],[0,3.0],linewidth=1,color='k')

            date = dt.datetime(year=year,month=month,day=day,hour=12,minute=0)
            ax.plot([date,date],[0,3.0],linewidth=1,color='k',linestyle='--')

        if vals[1] != '':
            hour = int(vals[1].split(':')[0])
            minute = int(vals[1].split(':')[1])

            date = dt.datetime(year=year,month=month,day=day,hour=hour,minute=minute)

            if i==1:
                orgdate=date

            enddate = date


        if vals[2] != '':

            duration = int(vals[2])

            start = date
            stop = date+dt.timedelta(minutes=duration)
            print start,stop
            left_breast_feeding.append([start,stop])

            if first_l:
                ax.plot([start,stop],[lbf_y,lbf_y],linewidth=lbf_width,color=lbf_col,label='Left breast')
            else:
                ax.plot([start,stop],[lbf_y,lbf_y],linewidth=lbf_width,color=lbf_col)

            first_l = False

        if vals[3] != '':

            duration = int(vals[3])

            start = date
            stop = date+dt.timedelta(minutes=duration)
            print start,stop
            left_breast_feeding.append([start,stop])

            #ax.axhline(y=1,xmin=start,xmax=stop,linewidth=4,color='b')
            if first_r:
                ax.plot([start,stop],[rbf_y,rbf_y],linewidth=rbf_width,color=rbf_col,label='Right breast')
            else:
                ax.plot([start,stop],[rbf_y,rbf_y],linewidth=rbf_width,color=rbf_col)

            first_r = False

        if vals[4]=='x' or vals[4]=='X':
            start = date
            print start
            if first_pee:
                ax.plot(start,0.6,'yo',markersize=10,alpha=0.4,label='pee')
            else:
                ax.plot(start,0.6,'yo',markersize=10,alpha=0.4)

            first_pee = False

        if vals[5].strip()=='x' or vals[5].strip()=='X':
            start = date
            print 'poop',start

            if first_poop:
                ax.plot(start,0.4,'ko',markersize=10,alpha=0.4,label='poop')
            else:
                ax.plot(start,0.4,'ko',markersize=10,alpha=0.4)

            first_poop = False



ax.set_ylim(0,3.0)
ax.legend()
ax.set_xlim(orgdate-dt.timedelta(minutes=20),enddate+dt.timedelta(minutes=20))

ax2 = ax.twiny()
ax2.plot([orgdate,enddate],[0.0,0.0],alpha=0)
ax2.xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
#ax2.xaxis.set_major_locator(DayLocator())





plt.show()
