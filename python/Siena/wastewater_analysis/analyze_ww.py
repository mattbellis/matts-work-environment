import sys

import matplotlib.pylab as plt
import matplotlib.lines as mlines

import numpy as np

import pandas as pd

import seaborn as sns

sns.set_style('darkgrid')

infilename = sys.argv[1]

df = pd.read_csv(infilename,delimiter='\t')

dates = df['DateReceived']

results = df['Result']

locations = df['Sampling Location']

print(locations)
print(locations.unique())

sampling_locations = locations.unique()

plt.figure(figsize=(12,4))
#for i,(date,result,location) in enumerate(zip(dates,results,locations)):
for i,location in enumerate(sampling_locations):

    #print(date,result,location)
    dftemp = df.loc[df['Sampling Location']==location]
    print("-----------")
    print(location)

    rs = dftemp['Result']
    dts = dftemp['DateReceived']

    for r,d in zip(rs,dts):
        fmt = 'k.'
        size = 2

        print(i,r,d)
        if r.lower().find('no sar')>=0:
            fmt = 'ks'
            size = 5

        elif r.lower().find('not q')>=0:
            fmt = 'yo'
            size = 10

        else:
            fmt = 'rv'
            size = 20

        plt.plot([d],[i],fmt,markersize=size)


print(sampling_locations)
#plt.gca().set_yticklabels(sampling_locations)
x = np.arange(0,len(sampling_locations),1)
plt.yticks(x,sampling_locations,fontsize=14)
plt.xticks(fontsize=14)

custom_lines = [mlines.Line2D([], [], color='k', marker='s', linestyle='None', markersize=5, label='No SARS2 detected'),
                mlines.Line2D([], [], color='y', marker='o', linestyle='None', markersize=8, label='SARS2 detected, not quantifiable'),
                mlines.Line2D([], [], color='r', marker='v', linestyle='None', markersize=10, label='SARS2 detected')
        ]

plt.legend(handles=custom_lines,fontsize=18,facecolor='w')

plt.tight_layout()

plt.savefig('ww_summary.png')

plt.show()
