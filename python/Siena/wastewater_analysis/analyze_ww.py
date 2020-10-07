import sys

import matplotlib.pylab as plt
import matplotlib.lines as mlines


import numpy as np

import pandas as pd

import seaborn as sns

import datetime as datetime

sns.set_style('darkgrid')

locations_from_file = None

infilename = sys.argv[1]

if len(sys.argv)>2:
    locfile = sys.argv[2]
    locations_from_file = np.loadtxt(locfile,delimiter='\t',unpack=True,skiprows=1,dtype=str)
    print(locations_from_file)

df = pd.read_csv(infilename,delimiter='\t')


dates = df['DateReceived']

#results = df['Result']
results = df['CountCOVID']

locations = df['Sampling Location']
#locations = df['Facility']

print(locations)
print(locations.unique())

#sampling_locations = np.sort(locations.unique())
sampling_locations = locations.unique()

plt.figure(figsize=(12,4))
#for i,(date,result,location) in enumerate(zip(dates,results,locations)):
for i,location in enumerate(sampling_locations):

    #print(date,result,location)
    dftemp = df.loc[df['Sampling Location']==location]
    #dftemp = df.loc[df['Facility']==location]
    print("-----------")
    print(location)

    #rs = dftemp['Result']
    rs = dftemp['CountCOVID']
    #dts = pd.to_datetime(dftemp['DateReceived'])
    dts = pd.to_datetime(dftemp['DateReceived']).dt.date

    for r,d in zip(rs,dts):
        fmt = 'k.'
        size = 2
        color='black'

        print(i,r,d)
        #if r.lower().find('no sar')>=0:
        if r==0:
            fmt = 'ks'
            size = 5

        #elif r.lower().find('not q')>=0:
        elif r==1 or r==2:
            fmt = 'yo'
            size = 10
            #color='yellow'
            color='orange'

        else:
            fmt = 'r^'
            size = 20
            color='red'

        print(d)
        print(type(d))
        plt.plot([d],[i],fmt,markersize=size,color=color)


print(sampling_locations)
#plt.gca().set_yticklabels(sampling_locations)
x = np.arange(0,len(sampling_locations),1)
plt.yticks(x,sampling_locations,fontsize=14)
plt.xticks(fontsize=14)

custom_lines = [mlines.Line2D([], [], color='k', marker='s', linestyle='None', markersize=5, label='No SARS2 detected'),
        #mlines.Line2D([], [], color='yellow', marker='o', linestyle='None', markersize=8, label='SARS2 detected, not quantifiable'),
                mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=8, label='SARS2 detected, not quantifiable'),
                mlines.Line2D([], [], color='r', marker='^', linestyle='None', markersize=10, label='SARS2 detected')
        ]

plt.legend(handles=custom_lines,fontsize=18,facecolor='w')

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig('ww_summary.png')
#plt.savefig('ww_summary2.png')

plt.figure()
for a,b in zip(locations_from_file[0],locations_from_file[1]):
    #plt.text(0,1.0,a)
    print(a,b)



plt.show()
