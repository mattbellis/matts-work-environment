import matplotlib.pylab as plt
import numpy as np

import sys

import datetime as dt
import pandas as pd

import seaborn as sns

################################################################################
def sort_courses(courses):

    depts = []
    numbers = []
    unique_courses = []

    for course in courses:
        depts.append(course['id'].split()[0])
        numbers.append(int(course['id'].split()[1]))
        
        identifier = (course['id'],course['type'])
        if identifier not in unique_courses:
            unique_courses.append(identifier)

    idx = np.argsort(numbers)

    print(idx)

    sorted_courses = []
    for i in idx:
        print(courses[i])
        sorted_courses.append(courses[i])

    print(unique_courses)
    unique_courses.sort()
    print(unique_courses)

    new_courses = []
    for i,c in enumerate(unique_courses):

        new_courses.append({'identifier':c,'instances':[]})

        for course in sorted_courses:
            
            identifier = (course['id'],course['type'])
            if identifier == c:
                new_courses[i]['instances'].append(course)


    return new_courses

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

grant = {'institution':'Siena College','funding_agency':'NSF', 'program1': 'EPP - could be None', 'progron2':'RUI', \
        'external':True, role='PI, CO-PI, contributor', 'coPIs':None, \
        'name':'PHY-XXXXX', 'amount':190000, 'start':datetime(2012,1,1), 'duration':deltatime(years=3), \
        'long_description'='ddddddddddddddd', \
        'short_description'='ddddddddddddddd'}
grants.append(grant)



print(grants)

#############################################
# Pandas stuff
#############################################
print("Building pandas stuf........")
df_dict = {}
for key in courses[0].keys():
    print(key)
    df_dict[key] = []

for course in courses:
    print(course)
    for key in course.keys():
        print(key)
        df_dict[key].append(course[key])

df = pd.DataFrame.from_dict(df_dict)

plt.figure()
sns.catplot(data=df, y='nstudents',x='term',hue='id',kind='bar')
#############################################

#plt.show()

courses = sort_courses(courses)

print(courses)

fig, ax = plt.subplots(figsize=(12,4))
course_names = []
for i,course in enumerate(courses):
    #identifier = course['identifier'][0]
    identifier = '{0} - {1}'.format(course['instances'][0]['name'],course['identifier'][0])
    course_names.append(identifier)
    print(identifier)
    for instance in course['instances']:
        start,end = term2date(instance['term'])
        xranges = [(start,end-start)]
        yrange = (i,1.0)
        # Plot the broken horizontal bars
        print(xranges)
        fc = 'blue'
        if instance['type']==1:
            fc = 'orange'
        plt.broken_barh(xranges, yrange, facecolors=fc)

#course_names.reverse()
print(course_names)
y_pos = np.arange(0,len(courses),1) + 0.5
ax.set_yticks(y_pos)
ax.set_yticklabels(course_names)
plt.grid(axis='y')

ax.xaxis_date()
plt.tight_layout()






plt.show()
      

