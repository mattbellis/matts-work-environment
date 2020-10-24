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

phages = [ ['QtAveCOVID','QtSDCOVID'],  
           ['QtAvePhageDNA','QtSDPhageDNA'],  
           ['QtAvePhagecDNA','QtSDPhagecDNA'],
           [None,None] 
           ]

for phage in phages:
    plt.figure(figsize=(14,7))
    #for i,(date,result,location) in enumerate(zip(dates,results,locations)):
    for i,location in enumerate(sampling_locations):

        #print(date,result,location)
        dftemp = df.loc[df['Sampling Location']==location].fillna(0)
        #dftemp = df.loc[df['Facility']==location]
        #print("-----------")
        #print(location)

        rs = dftemp['Result']
        #rs = dftemp['CountCOVID']
        #dts = pd.to_datetime(dftemp['DateReceived'])
        dts = pd.to_datetime(dftemp['DateReceived']).dt.date

        meas = None
        meas_uncert = None

        if phage[0] is not None:
            meas = dftemp[phage[0]]
            meas_uncert = dftemp[phage[1]]

            false_fields = ['NA', '< LOQ', '<LOQ', 'NaN', 'NA ']
            for f in false_fields:
                #print(f)
                meas[meas==f] = '0'
                meas_uncert[meas_uncert==f] = '0'

            meas = meas.astype(float)
            meas_uncert = meas_uncert.astype(float)

            meas.fillna(0)
            meas_uncert.fillna(0)

        else:
            meas0 = dftemp['QtAveCOVID']
            meas_uncert0 = dftemp['QtSDCOVID']
            meas1 = dftemp['QtAvePhagecDNA']
            meas_uncert1 = dftemp['QtSDPhagecDNA']

            false_fields = ['NA', '< LOQ', '<LOQ', 'NaN', 'NA ']
            for f in false_fields:
                meas0[meas0==f] = '0'
                meas_uncert0[meas_uncert0==f] = '0'
                meas1[meas1==f] = '0'
                meas_uncert1[meas_uncert1==f] = '0'

            meas0 = meas0.astype(float)
            meas_uncert0 = meas_uncert0.astype(float)
            meas0.fillna(0)
            meas_uncert0.fillna(0)

            meas1 = meas1.astype(float)
            meas_uncert1 = meas_uncert1.astype(float)
            meas1.fillna(0)
            meas_uncert1.fillna(0)

            meas = meas0/meas1
            meas_uncert = meas*np.sqrt((meas_uncert0/meas0)**2 + (meas_uncert1/meas1)**2)


        #plt.figure(figsize=(12,4))
        plt.subplot(3,4,i+1)
        #print(location)
        #print(meas.values)
        #print(meas_uncert.values)
        plt.errorbar(dts,meas,yerr=meas_uncert,fmt='o',markersize=5,color='k')
        plt.xlim(datetime.date(2020,8,15), datetime.date(2020,11,1))
        #plt.ylim(-0.1)
        if phage[0] is not None:
            plt.ylim(1,500000)
            plt.yscale('log')
        plt.title(location)
        plt.xticks(rotation=45)

        #plt.show()



    plt.tight_layout()

    if phage[0] is not None:
        plt.savefig('ww_summary_phages_{0}_LOG.png'.format(phage[0]))
    else:
        plt.savefig('ww_summary_phages_ratio.png'.format(phage[0]))


plt.show()
