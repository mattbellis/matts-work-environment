import matplotlib.pylab as plt
import numpy as np

import sys

import altair as alt
alt.renderers.enable('vegascope')

import datetime as dt
import pandas as pd


courses = []

# type: 0 - lecture, 1 - lab, 2 - studio
courses.append({'id':'PHYS 110', 'name':'General Physics Ia', 'type':0, 'nstudents':20, 'term':'F12'})
courses.append({'id':'PHYS 220', 'name':'Modern Physics', 'type':1, 'nstudents':14, 'term':'F12'})

courses.append({'id':'PHYS 120', 'name':'General Physics IIa', 'type':0, 'nstudents':15, 'term':'S13'})
courses.append({'id':'PHYS 260', 'name':'Thermal Physics', 'type':0, 'nstudents':25, 'term':'S13'})

print(courses)


alt.renderers.enable('jupyterlab')

data = pd.DataFrame()
end = []
start = []
classid = []
for course in courses:
    year = 2000 + int(course['term'][1:])
    monthstart = 1
    monthend = 5
    if course['term'][0] == 'F':
        monthstart = 9
        monthend = 12
    day = 15
    year = 2000 + int(course['term'][1:])
    start.append(dt.datetime(year, monthstart, day, 0, 0))
    end.append(dt.datetime(year, monthend, day, 0, 0))
    classid.append(course['id'])

data['from'] = start
data['to'] = end
data['activity'] = classid

chart = alt.Chart(data).mark_bar().encode(
            x='from',
                x2='to',
                    y='activity',
                        color=alt.Color('activity', scale=alt.Scale(scheme='dark2'))
                        ).interactive()

chart.display()
#plt.show()
