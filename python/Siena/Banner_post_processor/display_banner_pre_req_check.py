import numpy as np
import matplotlib.pylab as plt
import pandas as pd

import sys

infilename = sys.argv[1]

df = pd.read_csv(infilename)

# Select the PHYS and APHY courses
df = df[df['Course'].str.contains('PHY')]

for a in df['Title']:
    print(a)

courses = df['Course'].unique()

mycourses = {}

for course in courses:

    mycourses[course] = {}
    mycourses[course]['title'] = df.loc[df['Course']==course]['Title'].values[0]
    mycourses[course]['reqs'] = []

    #print(course) 
    subdf = df.loc[df['Course']==course]
    #print(subdf)
    #'''
    OR_FLAG = False
    for i,row in subdf.iterrows():
        #mycourses[course]['reqs'].append(row['Course'])
        #print("Here ----------")
        #print(i)
        #print(row)
        if row['PreqCourse']!=row['PreqCourse']:
            continue

        if row['LParen']=='(':
            or_reqs = []
            or_reqs.append(row['PreqCourse'])
            OR_FLAG = True

        elif row['RParen']==')':
            or_reqs.append(row['PreqCourse'])
            OR_FLAG = False
            mycourses[course]['reqs'].append(or_reqs)

        elif OR_FLAG is True and row['RParen']!=')' and row['LParen']!='(':
            or_reqs.append(row['PreqCourse'])
        else:
            mycourses[course]['reqs'].append(row['PreqCourse'])
    #'''

print()
print("----------")

#print(mycourses)
for key in mycourses.keys():
    print("\n------------------------")
    print(key,mycourses[key]['title'])
    reqs = mycourses[key]['reqs']
    print("------reqs-----------")
    for req in reqs:
        if type(req)==str:
            print(req)
        elif type(req)==list:
            output = ""
            for i,r in enumerate(req):
                if i==0:
                    output += r
                else:
                    output += " or " + r
            print(output)
