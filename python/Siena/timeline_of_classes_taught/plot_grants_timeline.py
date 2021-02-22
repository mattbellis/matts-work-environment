import matplotlib.pylab as plt
import numpy as np

import sys

import datetime as dt
import pandas as pd

import seaborn as sns

################################################################################
################################################################################
def term2date(term):

    year = 2000 + int(term[1:])
    monthstart = 1
    monthend = 5
    daystart = 15
    dayend = 1
    if term[0] == 'F':
        monthstart = 9
        monthend = 12
        daystart = 1
        dayend = 1
    day = 15
    year = 2000 + int(term[1:])
    start = dt.datetime(year, monthstart, daystart, 0, 0)
    end = dt.datetime(year, monthend, dayend, 0, 0)

    return start,end

################################################################################

grants = []

grant = {'institution':'Siena College','funding_agency':'NSF', 'program1': 'EPP', 'progron2':'RUI', \
        'external':True, 'role':'PI, CO-PI, contributor', 'coPIs':None, \
        'name':'PHY-XXXXX', 'title':'RUI: Searches for New Physics with the CMS Detector at the LHC', \
        'short_title':'CMS stuff', \
        'amount':190000, 'start':dt.datetime(2013,6,1), 'duration':dt.timedelta(days=3*365), 'funded':True, \
        'long_description':'ddddddddddddddd', \
        'short_description':'ddddddddddddddd'}
grants.append(grant)

grant = {'institution':'Siena College','funding_agency':'NSF', 'program1': 'EPP', 'progron2':'RUI', \
        'external':True, 'role':'PI, CO-PI, contributor', 'coPIs':None, \
        'name':'PHY-YYYYY', 'title':'RUI: Searches for New Physics with the CMS Detector at the LHC', \
        'short_title':'CMS stuff', \
        'amount':190000, 'start':dt.datetime(2016,6,1), 'duration':dt.timedelta(days=3*365), 'funded':True, \
        'long_description':'ddddddddddddddd', \
        'short_description':'ddddddddddddddd'}
grants.append(grant)

grant = {'institution':'Siena College','funding_agency':'NSF', 'program1': 'EPP', 'progron2':'RUI', \
        'external':True, 'role':'PI, CO-PI, contributor', 'coPIs':None, \
        'name':'PHY-ZZZZ', 'title':'RUI: Searches for New Physics with the CMS Detector at the LHC', \
        'short_title':'CMS stuff', \
        'amount':190000, 'start':dt.datetime(2019,6,1), 'duration':dt.timedelta(days=3*365), 'funded':True, \
        'long_description':'ddddddddddddddd', \
        'short_description':'ddddddddddddddd'}
grants.append(grant)

grant = {'institution':'Siena College','funding_agency':'NSF', 'program1': 'EPP', 'progron2':'RUI', \
        'external':True, 'role':'PI, CO-PI, contributor', 'coPIs':None, \
        'name':'PHY-MMMM', 'title':' RUI: Searching for an annual modulation of naturally-occurring isotopes in atmospheric aerosols as an explanation for the DAMA-LIBRA dark matter signal', \
        'short_title':'Dark matter', \
        'amount':190000, 'start':dt.datetime(2019,6,1), 'duration':dt.timedelta(days=3*365), 'funded':False, \
        'long_description':'ddddddddddddddd', \
        'short_description':'ddddddddddddddd'}
grants.append(grant)

grant = {'institution':'Siena College','funding_agency':'CURCA', 'program1': 'Summer Scholars', 'progron2':None, \
        'external':False, 'role':'PI', 'coPIs':None, \
        'name':'CURCA-this', 'title':'A research', \
        'short_title':'Cloud chamber', \
        'amount':2000, 'start':dt.datetime(2013,7,1), 'duration':dt.timedelta(days=45), 'funded':True, \
        'long_description':'ddddddddddddddd', \
        'short_description':'ddddddddddddddd'}
grants.append(grant)

print(grants)
ngrants = len(grants)

#############################################
# Pandas stuff
#############################################
print("Building pandas stuf........")
df_dict = {}
for key in grants[0].keys():
    print(key)
    df_dict[key] = []

for grant in grants:
    print(grant)
    for key in grant.keys():
        print(key)
        df_dict[key].append(grant[key])

df = pd.DataFrame.from_dict(df_dict)

#plt.figure()
#sns.catplot(data=df, y='name',x='start',hue='funding_agency',kind='bar')
#############################################

#plt.show()

print(grants)

colors = {'NSF':'blue', 'APS':'red', 'CMS internal':'orange', 'AAPT':'green', 'CURCA':'yellow'}

################################################################################
# External
################################################################################

fig, ax = plt.subplots(figsize=(12,4))
grant_names = []
for i,grant in enumerate(grants):

    if grant['external'] is False:
        continue 

    #identifier = grant['identifier'][0]
    #identifier = '{0} - {1}'.format(grant['instances'][0]['name'],grant['identifier'][0])
    identifier = '{0} - {1}'.format(grant['name'],grant['start'].year)
    grant_names.append(identifier)
    print(identifier)

    start,end = grant['start'], grant['start'] + grant['duration']
    xranges = [(start,end-start)]
    yrange = (ngrants - i - 1,1.0)
    # Plot the broken horizontal bars
    print(xranges)
    fc = 'blue'
    if grant['funding_agency'] in colors.keys():
        fc = colors[grant['funding_agency']]

    alpha = 1.0
    if grant['funded']==False:
        alpha = 0.2
    plt.broken_barh(xranges, yrange, facecolors=fc, alpha=alpha)

#grant_names.reverse()
print(grant_names)
grant_names.reverse()
print(grant_names)
y_pos = np.arange(0,len(grants),1) + 0.5
ax.set_yticks(y_pos)
ax.set_yticklabels(grant_names)
plt.grid(axis='y')

ax.xaxis_date()
plt.tight_layout()


################################################################################
# Internal
#
# Stacked bar graph?
#
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html
################################################################################

fig, ax = plt.subplots(figsize=(12,4))
grant_names = []
for i,grant in enumerate(grants):

    if grant['external'] is True:
        continue 

    #identifier = grant['identifier'][0]
    #identifier = '{0} - {1}'.format(grant['instances'][0]['name'],grant['identifier'][0])
    identifier = '{0} - {1}'.format(grant['name'],grant['start'].year)
    grant_names.append(identifier)
    print(identifier)

    start,end = grant['start'], grant['start'] + grant['duration']
    xranges = [(start,end-start)]
    yrange = (ngrants - i - 1,1.0)
    # Plot the broken horizontal bars
    print(xranges)
    fc = 'blue'
    if grant['funding_agency'] in colors.keys():
        fc = colors[grant['funding_agency']]

    alpha = 1.0
    if grant['funded']==False:
        alpha = 0.2
    plt.broken_barh(xranges, yrange, facecolors=fc, alpha=alpha)

#grant_names.reverse()
print(grant_names)
grant_names.reverse()
print(grant_names)
y_pos = np.arange(0,len(grants),1) + 0.5
ax.set_yticks(y_pos)
ax.set_yticklabels(grant_names)
plt.grid(axis='y')

ax.xaxis_date()
plt.tight_layout()






plt.show()
      





plt.show()
      

