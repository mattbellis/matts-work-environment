#!/usr/bin/env python
# make a horizontal bar chart

from matplotlib import patches
import matplotlib.pyplot as plt
from pylab import *
import sys

filename = sys.argv[1]

infile = open(filename)

production = []
sites = []
countries = []
barcolors = []

pie_country = []
pie_production = []
pie_colors = []

total = 0
bars = []

count = 0
for line in infile:
  vals = line.split(":")
  if len(vals) > 2 and vals[1].strip() != 'Totals':
    site = vals[1].strip()
    prod = int(vals[2].strip())
    country = vals[3].strip()

    total += prod

    production.append(prod)
    sites.append(site)
    countries.append(country)

    if country == "USA":
      barcolors.append('y')
    elif country == "UK":
      barcolors.append('c')
    elif country == "France":
      barcolors.append('b')
    elif country == "Germany":
      barcolors.append('m')
    elif country == "Canada":
      barcolors.append('r')
    elif country == "Italy":
      barcolors.append('g')


    if country in pie_country:
      index = pie_country.index(country)
      pie_production[index] += prod
    else:
      pie_country.append(country)
      pie_production.append(prod)
      pie_colors.append(barcolors[count])
      bars.append(bar(0, 0.1, bottom=1, color=barcolors[count]))

    count += 1

############################################
pie_pct = []
for p in pie_production:
  pie_pct.append(100*float(p)/total)


############################################

#val = 3+10*rand(5)    # the bar lengths
pos = arange( len(sites) )+.5    # the bar centers on the y axis

plt.figure(figsize=(6,6), num=1)

barchart = barh(pos, (production), align='center', ecolor='r', color=(barcolors))

yticks(pos, (sites))
leg = legend(bars, pie_country, loc='upper right')

xlabel('Events produced')
title('SP production by site')
grid(True)

# Pie chart
plt.figure(num=2, figsize=(6,6))
ax = axes([0.1, 0.1, 0.8, 0.8])


#labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
#fracs = [15,30,45, 10]

print pie_country
print bars
piechart = pie(pie_pct, labels=pie_country, autopct='%1.1f%%', shadow=True, colors=pie_colors)
title('SP % by country', bbox={'facecolor':'0.8', 'pad':5})

################################################################################
# Save the figures
################################################################################
filename_base = filename.split('.')[0].split('/')[-1]
save_format = 'png'

name = "%s_bar_chart.%s" % (filename_base,save_format)
plt.figure(num=1).savefig(name,format=save_format)

name = "%s_pie_chart.%s" % (filename_base,save_format)
plt.figure(num=2).savefig(name,format=save_format)

################################################################################



"""
plt.figure(2)
barh(pos, (production), xerr=rand(5), ecolor='r', align='center')
yticks(pos, (sites))
xlabel('Performance')
"""

show()

