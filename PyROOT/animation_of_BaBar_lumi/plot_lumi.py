#!/usr/bin/env python

import datetime
from matplotlib.pyplot import figure, show
from matplotlib.dates import DayLocator, HourLocator, MonthLocator, YearLocator, DateFormatter, drange
from matplotlib.ticker import NullLocator
from numpy import arange

################################################################################
def epem_xsec(lumi, process):
    xsec = 1.0
    if process=='bbbar':
        xsec = 1.08
    elif process=='ccbar':
        xsec = 1.30
    elif process=='uds':
        xsec = 2.09
    elif process=='mupmum':
        xsec = 1.16
    elif process=='tauptaum':
        xsec = 0.94

    ret = lumi * xsec
    return ret
################################################################################

################################################################################
def update_ax2_bbbar(ax1, ax2):
    y1, y2 = ax1.get_ylim()
    ax2.set_ylim(bbbar_xsec(y1), bbbar_xsec(y2))
    ax2.figure.canvas.draw()
################################################################################



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
subplots_twin = []
for j in range(0,6):
    subplots_twin.append([])

for i in range(0,1):
    division = 111 + i
    subplots.append(fig1.add_subplot(division))
    for j in range(0,6):
        subplots_twin[j].append(subplots[i].twinx())

# Read in the lumi info from file
years = []
months = []
lumi_on = []
lumi_off = []
lumi_on_tot = []
lumi_off_tot = []
infile = open('luminosity_vals.txt')
dates = []

# Fermion anti-fermion pairs
ff_pairs = []
ff_names = ['epem','mupmum','tauptaum','uds','ccbar','bbbar']
ff_colors = ['black','purple','cyan','red','green','blue']
for j in range(0,6):
    ff_pairs.append([])

# Read out the values and write to a file.
# 
outfilename = "xsec_values.dat"
outfile = open(outfilename,'w+')

for i,line in enumerate(infile):
    vals = line.split()
    if len(vals)>0 and line[0]!='#':
        years.append(int(vals[0]))
        months.append(int(vals[1]))
        # lumi is in ipb in file.
        lumi_on.append(float(vals[2])/1000.0)
        lumi_off.append(float(vals[3])/1000.0)
        if i>0:
            lumi_on_tot.append(lumi_on_tot[i-1]+lumi_on[i])
            lumi_off_tot.append(lumi_off_tot[i-1]+lumi_off[i])
        else:
            lumi_on_tot.append(float(vals[2]))
            lumi_off_tot.append(float(vals[3]))

        #print "%f %f" % (float(vals[2])/1000.0, lumi_on_tot[i])

        the_date = datetime.datetime(years[i],months[i],1)
        dates.append(the_date)

        output = "%s " % (the_date)
        for j in range(0,6):
            xsec = epem_xsec(lumi_on_tot[i],ff_names[j])
            ff_pairs[j].append(xsec)
            output += "%f " % (xsec)
        print output
        output += "\n"
        outfile.write(output)

outfile.close()

npts = len(years)
print years


#date1 = datetime.datetime( years[0], months[0], 0)
#date2 = datetime.datetime( years[-1, monthts[-1], 0)
#delta = datetime.timedelta(hours=6)
#dates = drange(date1, date2, delta)

end_date = dates[0]

plot_dates = []
y_lumi = []
y_ff = []
for j in range(0,6):
    y_ff.append([])

for i in range(0,npts):

    #end_date += delta
    plot_dates.append(dates[i])
    #print plot_dates

    #y = arange( len(dates)*1.0)
    #y = arange( len(plot_dates)*1.0)
    y_lumi.append(lumi_on_tot[i])
    for j in range(0,6):
        y_ff[j].append(ff_pairs[j][i])

    #subplots[0].callbacks.connect("ylim_changed", update_ax2_bbbar(subplots[0], subplots_twin[0]))
    #fig = figure()
    #ax = fig.add_subplot(111)
    #ax.plot_date(dates, y*y)
    subplots[0].plot_date(plot_dates, y_lumi, '-o', color='black')

    for j in range(0,6):
        subplots_twin[j][0].plot_date(plot_dates, y_ff[j], '-', color=ff_colors[j])
        subplots_twin[j][0].set_ylim(0, 1000)

    # this is superfluous, since the autoscaler should get it right, but
    # use date2num and num2date to to convert between dates and floats if
    # you want; both date2num and num2date convert an instance or sequence
    subplots[0].set_xlim( dates[0], dates[-1] )
    subplots[0].set_ylim(0, 500)


    # The hour locator takes the hour or sequence of hours you want to
    # tick, not the base multiple

    #subplots[0].xaxis.set_major_locator( DayLocator() )
    #subplots[0].xaxis.set_minor_locator( HourLocator(arange(0,25,6)) )
    #subplots[0].xaxis.set_major_formatter( DateFormatter('%Y-%m-%d') )
    subplots[0].xaxis.set_major_locator( YearLocator() )
    subplots[0].xaxis.set_minor_locator( MonthLocator() )
    subplots[0].xaxis.set_major_formatter( DateFormatter('%Y-%m') )

    subplots[0].fmt_xdata = DateFormatter('%Y-%m-%d %H:%M:%S')
    fig1.autofmt_xdate()

    #show()

    #filename = "figures/test_%d.pdf" % (i)
    filename = "figures/test_%d.png" % (i)
    print filename
    fig1.savefig(filename)

