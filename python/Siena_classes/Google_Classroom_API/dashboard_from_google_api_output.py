import matplotlib.pylab as plt
import numpy as np
import random
import string

import pandas as pd

from datetime import date

import seaborn as sns

import sys

infilename = sys.argv[1]

df = pd.read_hdf(infilename,'df')
dfst = pd.read_hdf(infilename,'dfst')
dfwo = pd.read_hdf(infilename,'dfwo')

# Get ready to munge them
df['title'] = 'default'
df['dueDate'] = date(2020,1,1)
df['maxPoints'] = -1.
df['assignmentType'] = 'ungraded'
df['weight'] = 25

df['fullName'] = 'default'
df['emailAddress'] = 'default'

################################################################################
# Add in the other info
################################################################################
for index,row in dfwo.iterrows():
    id = row['id']
    title = row['title']
    dueDate = row['dueDate']
    maxPoints = row['maxPoints']

    d = date(dueDate['year'],dueDate['month'],dueDate['day'])

    ########### FOR SOME REASON IT'S NOT DOWNLOADING ALL THE INDIVIDUAL ASSIGNMENTS FOR STUDENTS????? ###########
    df.loc[df['courseWorkId']==id, ['title','maxPoints','dueDate']] = title,maxPoints,d


for index,row in dfst.iterrows():
    id = row['userId']
    fullName = row['fullName']
    emailAddress = row['emailAddress']

    df.loc[df['userId']==id, ['fullName','emailAddress']] = fullName,emailAddress

df['percentage'] = df['grade']/df['maxPoints']

################################################################################
# Add in the assignment type and weighting info
################################################################################
# PHYS 250, 
# Homework/quizzes 50
# Midterm          25
# Final            25
weighting = {'homework':50, 'quiz':50, 'midterm':25, 'final': 25}

df.loc[df['title'].str.lower().str.contains('homework'), 'assignmentType'] = "homework"
df.loc[df['title'].str.lower().str.contains('self'), 'assignmentType'] = "homework"

df.loc[df['title'].str.lower().str.contains('quiz'), 'assignmentType'] = "quiz"
df.loc[df['title'].str.lower().str.contains('getting'), 'assignmentType'] = "quiz"

df.loc[df['title'].str.lower().str.contains('midterm'), 'assignmentType'] = "midterm"

df.loc[df['title'].str.lower().str.contains('final'), 'assignmentType'] = "final"

# Get the dates into a pandas format and sort
df['dueDate'] = pd.to_datetime(df['dueDate'])
df.sort_values(by='dueDate')


# Calculated weighted averages
df['weight'] = 0.0
for key in weighting.keys():
    df.loc[df.assignmentType==key, 'weight'] = weighting[key]

df['weighted'] = 0.0
df['weighted'] = df['percentage']*df['weight']/100

####### SOMETHING LIKE THIS
g = df.groupby('fullName')
# https://stackoverflow.com/questions/31521027/groupby-weighted-average-and-sum-in-pandas-dataframe
# Not sure if this is right
g.weighted.sum()/1000

# For mean of subgroups
df.loc[df['assignmentType']=='homework'].groupby('fullName').mean()
df.loc[df['assignmentType']=='homework'].groupby('fullName')['percentage'].mean()

# Is this the way to go
dffinal = pd.DataFrame()
dffinal['homework'] = df.loc[df['assignmentType']=='homework'].groupby('fullName')['percentage'].mean()
dffinal['quiz'] = df.loc[df['assignmentType']=='quiz'].groupby('fullName')['percentage'].mean()
dffinal['midterm'] = df.loc[df['assignmentType']=='midterm'].groupby('fullName')['percentage'].mean()
dffinal['final'] = df.loc[df['assignmentType']=='final'].groupby('fullName')['percentage'].mean()

exit()



################################################################################

# Set colors
cmap = plt.get_cmap("binary")
ncategories = 4
colors = cmap(np.arange(0,256,int(256/ncategories)))
################################################################################



#####################################
# Put it into a data frame
#####################################
name = []
dates = []
ass_type = []
ass_name = []
grade = []
fraction = []


'''
# These were working pretty well!
assignment_types = np.array(['hw','quizzes','midterms','final exam'])
breakdown = np.array([0.3, 0.35, 0.25, 0.2])

plt.figure()
sns.boxplot(data=df[df['assignmentType']=='homework'],x='title',y='percentage',hue='assignmentType')

plt.figure()
sns.boxplot(data=df[df['assignmentType']=='homework'],x='dueDate',y='percentage',hue='assignmentType')

plt.figure()
sns.boxplot(data=df[df['assignmentType']=='quiz'],x='title',y='percentage',hue='assignmentType')
plt.figure()
sns.boxplot(data=df[df['assignmentType']=='quiz'],x='dueDate',y='percentage',hue='assignmentType')

plt.figure()
sns.boxplot(data=df,x='title',y='percentage',hue='assignmentType')
plt.figure()
sns.boxplot(data=df,x='dueDate',y='percentage',hue='assignmentType')
'''

# Histograms
#df.groupby('ass_name').boxplot(column='grade')
#df.boxplot(column='grade',by='ass_name')
#df.boxplot(column='grade',by='date')

'''
plt.figure()
ax = sns.boxplot(x="date", y="grade", hue='ass_type', data=df)
plt.ylim(0,110)
plt.legend()

plt.figure()
ax = sns.boxplot(x="name", y="grade", hue='ass_type', data=df)
plt.ylim(0,110)
plt.legend()


plt.figure()
ax = sns.boxplot(x="name", y="grade", data=df)
plt.ylim(0,110)
plt.legend()
'''

###########################
'''
n = df['name'].unique()

nnames = len(n)

for i in range(0,nnames):
    if i%6==0:
        idx = False

    idx |= df['name']==n[i]

    if i%6==5 or i==nnames-1:

        dftemp = df[idx]

        plt.figure()
        ax = sns.boxplot(x="name", y="grade", data=dftemp)
        plt.ylim(0,110)
        plt.legend()

'''
plt.show()
