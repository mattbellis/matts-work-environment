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

df.loc[df['title'].str.lower().str.contains('homework'), 'assignmentType'] = "homework"
df.loc[df['title'].str.lower().str.contains('quiz'), 'assignmentType'] = "quiz"
df.loc[df['title'].str.lower().str.contains('getting'), 'assignmentType'] = "quiz"
df.loc[df['title'].str.lower().str.contains('self'), 'assignmentType'] = "quiz"

df.loc[df['title'].str.lower().str.contains('midterm'), 'assignmentType'] = "midterm"

df.loc[df['title'].str.lower().str.contains('final'), 'assignmentType'] = "final"


################################################################################

# Set colors
cmap = plt.get_cmap("binary")
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


assignment_types = np.array(['hw','quizzes','midterms','final exam'])
breakdown = np.array([0.3, 0.35, 0.25, 0.2])

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
