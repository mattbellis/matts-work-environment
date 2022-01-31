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

    # By hand
    #year = 2000 + int(term[1:])
    # Read in
    year = int(term.split()[-1])
    monthstart = 1
    monthend = 5
    daystart = 15
    dayend = 1
    # By hand
    #if term[0] == 'F':
    # Read in
    if term.find('Fall')>=0:
        monthstart = 9
        monthend = 12
        daystart = 1
        dayend = 15
    #day = 15
    # By hand
    #year = 2000 + int(term[1:])
    # Read in
    year = int(term.split()[-1])
    start = dt.datetime(year, monthstart, daystart, 0, 0)
    end = dt.datetime(year, monthend, dayend, 0, 0)

    return start,end

################################################################################

courses = []

# type: 0 - lecture, 1 - lab, 2 - studio, 3 - tutorial
courses.append({'id':'PHYS 310', 'name':'Mechanics', 'type':0, 'nstudents':25, 'term':'F12'})
courses.append({'id':'PHYS 310', 'name':'Mechanics', 'type':0, 'nstudents':25, 'term':'F13'})
courses.append({'id':'PHYS 310', 'name':'Mechanics', 'type':0, 'nstudents':25, 'term':'F14'})

courses.append({'id':'PHYS 110', 'name':'General Physics Ia', 'type':0, 'nstudents':20, 'term':'F12'})
courses.append({'id':'PHYS 110', 'name':'General Physics Ia', 'type':0, 'nstudents':20, 'term':'F14'})
courses.append({'id':'PHYS 110', 'name':'General Physics Ia', 'type':0, 'nstudents':20, 'term':'S17'})
courses.append({'id':'PHYS 110', 'name':'General Physics Ia', 'type':0, 'nstudents':20, 'term':'S18'})

courses.append({'id':'PHYS 110', 'name':'General Physics Ia', 'type':1, 'nstudents':20, 'term':'S17'})

courses.append({'id':'PHYS 130', 'name':'General Physics I', 'type':0, 'nstudents':20, 'term':'F13'})
courses.append({'id':'PHYS 130', 'name':'General Physics I', 'type':0, 'nstudents':20, 'term':'F14'})

courses.append({'id':'PHYS 140', 'name':'General Physics I', 'type':1, 'nstudents':20, 'term':'S14'})
courses.append({'id':'PHYS 140', 'name':'General Physics I', 'type':1, 'nstudents':20, 'term':'S16'})

courses.append({'id':'PHYS 220', 'name':'Modern Physics', 'type':1, 'nstudents':14, 'term':'F12'})

courses.append({'id':'PHYS 120', 'name':'General Physics IIa', 'type':0, 'nstudents':15, 'term':'S13'})

courses.append({'id':'PHYS 260', 'name':'Thermal Physics', 'type':0, 'nstudents':25, 'term':'S13'})
courses.append({'id':'PHYS 260', 'name':'Thermal Physics', 'type':0, 'nstudents':25, 'term':'S14'})
courses.append({'id':'PHYS 260', 'name':'Thermal Physics', 'type':0, 'nstudents':25, 'term':'S15'})
courses.append({'id':'PHYS 260', 'name':'Thermal Physics', 'type':0, 'nstudents':25, 'term':'S16'})
courses.append({'id':'PHYS 260', 'name':'Thermal Physics', 'type':0, 'nstudents':25, 'term':'S18'})

courses.append({'id':'SCDV 001', 'name':'Intro to Engineering', 'type':0, 'nstudents':25, 'term':'F13'})

courses.append({'id':'PHYS 015', 'name':'Quarks, Quanta, and Quasars', 'type':0, 'nstudents':25, 'term':'S17'})
courses.append({'id':'PHYS 400', 'name':'Nuclear and Particle Physics', 'type':0, 'nstudents':25, 'term':'S16'})
courses.append({'id':'PHYS 400', 'name':'Nuclear and Particle Physics', 'type':2, 'nstudents':2, 'term':'S17'})

courses.append({'id':'PHYS 440', 'name':'Quantum Physics', 'type':0, 'nstudents':18, 'term':'F17'})

courses.append({'id':'CSIS 200', 'name':'Software Tools for Physicists', 'type':0, 'nstudents':18, 'term':'F15'})
courses.append({'id':'CSIS 200', 'name':'Software Tools for Physicists', 'type':0, 'nstudents':18, 'term':'F16'})
courses.append({'id':'CSIS 200', 'name':'Software Tools for Physicists', 'type':0, 'nstudents':18, 'term':'F17'})


print(courses)

################################################################################
# Reading in from what we download from Banner
################################################################################
df = pd.read_html(sys.argv[1])
df = df[-1]
#df['name'] = df['Course Title']
#df['id'] = df['Course']
#df['term'] = df['Associated Term']
courses = []
nc = len(df)
for i in range(nc):
    course = {}
    course['id'] = df['Course'].iloc[i]
    course['name'] = df['Course Title'].iloc[i]
    course['term'] = df['Associated Term'].iloc[i]
    course['nstudents'] = 18
    if df['Course Title'].iloc[i].find('Lab')>=0:
        course['type'] = 1
    elif df['Course'].iloc[i].find('499')>=0:
        course['type'] = 2
    else:
        course['type'] = 0

    courses.append(course)


#############################################
# Pandas stuff
#############################################
'''
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
'''
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
        elif instance['type']==2:
            fc = 'red'
        plt.broken_barh(xranges, yrange, facecolors=fc)

#course_names.reverse()
print(course_names)
y_pos = np.arange(0,len(courses),1) + 0.5
ax.set_yticks(y_pos)
ax.set_yticklabels(course_names)
plt.grid(axis='y')

plt.xlim(dt.datetime(2012,1,1,1,1))

ax.xaxis_date()
plt.tight_layout()

########################################################


plt.show()
