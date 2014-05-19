#!/usr/bin/env python

import datetime
from matplotlib.pyplot import figure, show
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from numpy import arange

################################################################################
# Make a figure on which to plot stuff.
fig1 = figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
#
# Usage is XYZ: X=how many rows to divide.
#               Y=how many columns to divide.
#               Z=which plot to plot based on the first being '1'.
# So '111' is just one plot on the main figure.
################################################################################
subplots = []
for i in range(0,1):
    division = 111 + i
    subplots.append(fig1.add_subplot(division))



date1 = datetime.datetime( 2000, 3, 2)
date2 = datetime.datetime( 2000, 3, 6)
delta = datetime.timedelta(hours=6)
dates = drange(date1, date2, delta)

end_date = date1
for i in range(0,10):

    end_date += delta
    plot_dates = drange(date1,end_date,delta)

    #y = arange( len(dates)*1.0)
    y = arange( len(plot_dates)*1.0)

    #fig = figure()
    #ax = fig.add_subplot(111)
    #ax.plot_date(dates, y*y)
    subplots[0].plot_date(plot_dates, y*y, '-')

    # this is superfluous, since the autoscaler should get it right, but
    # use date2num and num2date to to convert between dates and floats if
    # you want; both date2num and num2date convert an instance or sequence
    subplots[0].set_xlim( dates[0], dates[-1] )
    subplots[0].set_ylim(0, 100)

    # The hour locator takes the hour or sequence of hours you want to
    # tick, not the base multiple

    subplots[0].xaxis.set_major_locator( DayLocator() )
    subplots[0].xaxis.set_minor_locator( HourLocator(arange(0,25,6)) )
    subplots[0].xaxis.set_major_formatter( DateFormatter('%Y-%m-%d') )

    subplots[0].fmt_xdata = DateFormatter('%Y-%m-%d %H:%M:%S')
    fig1.autofmt_xdate()

    #show()

    filename = "figures/test_%d.pdf" % (i)
    print filename
    fig1.savefig(filename)

