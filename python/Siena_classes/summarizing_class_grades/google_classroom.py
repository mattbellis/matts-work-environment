import numpy as np
import matplotlib.pylab as plt
import pandas as pd

import sys

infilename = sys.argv[1]

df = pd.read_csv(infilename)

colnames = df.columns

hwandquizinfo = {}
hwinfo = {}
quizinfo = {}
hwinfo = {}
midterminfo = {}
finalinfo = {}

for name in colnames:
    if name.find('QUIZ')>=0:
        quizinfo[name] = []
        hwandquizinfo[name] = []
    elif name.find('HW')>=0 or name.find('HOMEWORK')>=0:
        hwinfo[name] = []
        hwandquizinfo[name] = []
    elif name.find('MID')>=0:
        midterminfo[name] = []
    elif name.find('FINAL')>=0:
        finalinfo[name] = []

'''
print(quizinfo)
print()
print(hwinfo)
print()
print(midterminfo)
print()
print(finalinfo)
print()
'''

for key in hwandquizinfo.keys():
    d = df[key].iloc[0]
    p = float(df[key].iloc[1])
    hwandquizinfo[key] = [d,p]
    #print(key,d,p)
for key in quizinfo.keys():
    d = df[key].iloc[0]
    p = float(df[key].iloc[1])
    quizinfo[key] = [d,p]
    #print(key,d,p)
for key in hwinfo.keys():
    d = df[key].iloc[0]
    p = float(df[key].iloc[1])
    hwinfo[key] = [d,p]
    #print(key,d,p)
for key in midterminfo.keys():
    d = df[key].iloc[0]
    p = float(df[key].iloc[1])
    midterminfo[key] = [d,p]
    #print(key,d,p)
for key in finalinfo.keys():
    d = df[key].iloc[0]
    p = float(df[key].iloc[1])
    finalinfo[key] = [d,p]
    #print(key,d,p)

def summarize(idx,info,df,drop=False):

    dftemp = df.iloc[idx]
    lname = dftemp['Last Name']
    fname = dftemp['First Name']

    #print(f'{fname} {lname}')
    grades = []
    for key in info.keys():
        #print(key)
        d = info[key][0]
        p = info[key][1]

        score = float(dftemp[key])
        grade = score/p
        if grade != grade:
            #print("NAN!")
            1
        else:
            grades.append(100*grade)

        if idx==144:
            print(f"{d:12s} max: {p:5.1f}  score: {score:5.1f}  grade: {100*grade:7.2f}    {key}")
    #print(grades)
    if drop==True:
        grades = np.sort(grades)[1:]
    #print(grades)
    ave = np.mean(grades)
    #print(f"Average: {ave:8.2f}")
    return ave




for i in range(2,len(df)):

    dftemp = df.iloc[i]
    lname = dftemp['Last Name']
    fname = dftemp['First Name']

    hqave = summarize(i,hwandquizinfo,df,True)
    #hqave = summarize(i,hwandquizinfo,df,False)
    have = summarize(i,hwinfo,df)
    qave = summarize(i,quizinfo,df)
    mave = summarize(i,midterminfo,df)
    fave = summarize(i,finalinfo,df)

    ave = 0.5*hqave + 0.25*mave + 0.25*fave

    print(f"{i:2d} {fname:12s} {lname:18s}   {ave:5.1f}     hq: {hqave:5.1f} m: {mave:5.1f} f: {fave:5.1f}     h: {have:5.1f} q: {qave:5.1f}")




