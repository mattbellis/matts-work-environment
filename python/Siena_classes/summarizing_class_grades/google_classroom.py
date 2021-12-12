import numpy as np
import matplotlib.pylab as plt
import pandas as pd

import sys

infilename = sys.argv[1]

df = pd.read_csv(infilename)

colnames = df.columns

quizinfo = {}
hwinfo = {}
midterminfo = {}
finalinfo = {}

for name in colnames:
    if name.find('QUIZ')>=0:
        quizinfo[name] = []
    elif name.find('HW')>=0 or name.find('HOMEWORK')>=0:
        hwinfo[name] = []
    elif name.find('MID')>=0:
        midterminfo[name] = []
    elif name.find('FINAL')>=0:
        finalinfo[name] = []

print(quizinfo)
print()
print(hwinfo)
print()
print(midterminfo)
print()
print(finalinfo)
print()

for key in quizinfo.keys():
    d = df[key].iloc[0]
    p = float(df[key].iloc[1])
    quizinfo[key] = [d,p]
    print(key,d,p)
for key in hwinfo.keys():
    d = df[key].iloc[0]
    p = float(df[key].iloc[1])
    hwinfo[key] = [d,p]
    print(key,d,p)
for key in midterminfo.keys():
    d = df[key].iloc[0]
    p = float(df[key].iloc[1])
    midterminfo[key] = [d,p]
    print(key,d,p)
for key in finalinfo.keys():
    d = df[key].iloc[0]
    p = float(df[key].iloc[1])
    finalinfo[key] = [d,p]
    print(key,d,p)

for i in range(2,len(df)):
    dftemp = df.iloc[i]
    lname = dftemp['Last Name']
    fname = dftemp['First Name']

    print(f'{fname} {lname}')
    for key in hwinfo.keys():
        d = hwinfo[key][0]
        p = hwinfo[key][1]

        score = float(dftemp[key])
        grade = score/p

        print(d, key, p, score, grade)




