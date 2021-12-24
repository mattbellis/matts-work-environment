import numpy as np
import matplotlib.pylab as plt
import pandas as pd

import sys

infilename = sys.argv[1]

df = pd.read_csv(infilename)

colnames = df.columns

hwandquizinfo = {}
partinfo = {}
hwinfo = {}
quizinfo = {}
hwinfo = {}
finalexaminfo = {}
finalprojectinfo = {}

for name in colnames:
    if name.find('QUIZ')>=0:
        quizinfo[name] = []
    # EDAV
    elif name.find('CHECKPOINT')>=0 or name.find('Reflection')>=0 or name.find('blog')>=0 \
    or name.find('abstracts')>=0 or name.find('Getting')>=0 or name.find('Introduce')>=0:
    # Quantum
    #elif name.find('HW')>=0 or name.find('HOMEWORK')>=0 or name.find('CODING')>=0:
        hwinfo[name] = []
    elif name.find('Final exam')>=0:
        finalexaminfo[name] = []
    elif name.find('Final project')>=0:
        finalprojectinfo[name] = []

'''
print(quizinfo)
print()
print(hwinfo)
print()
print(finalexaminfo)
print()
print(finalprojectinfo)
print()
'''

for key in quizinfo.keys():
    d = df[key].iloc[1]
    p = float(df[key].iloc[1])
    if p<=0:
        continue
    quizinfo[key] = [d,p]
    #print(key,d,p)
for key in hwinfo.keys():
    d = df[key].iloc[1]
    p = df[key].iloc[1].strip()
    #print("there: ",key,"A",p.strip(),"B")
    #print(p.isnumeric())
    if not p.replace('.','',1).isdigit():
        continue
    #print("there: ",key,"A",p.strip(),"B")
    #print(p.isnumeric())
    p = float(p)
    if p<=0:
        continue
    hwinfo[key] = [d,p]
    #print(key,d,p)
for key in finalexaminfo.keys():
    d = df[key].iloc[1]
    p = df[key].iloc[1].strip()
    if not p.replace('.','',1).isdigit():
        continue
    p = float(p)
    if p<=0:
        continue
    finalexaminfo[key] = [d,p]
    #print(key,d,p)
for key in finalprojectinfo.keys():
    d = df[key].iloc[1]
    p = df[key].iloc[1]
    if not p.replace('.','',1).isdigit():
        continue
    p = float(p)
    if p<=0:
        continue
    finalprojectinfo[key] = [d,p]
    #print(key,d,p)

def summarize(idx,info,df,drop=False):

    dftemp = df.iloc[idx]
    lname = dftemp['Student']

    #print(f'{fname} {lname}')
    grades = []
    for key in info.keys():
        #print(key)
        if len(info[key])==0:
            continue
        #print(key,info[key])
        d = info[key][0]
        p = info[key][1]

        if d!=d or p!=p:
            continue

        if p<=0:
            continue

        score = float(dftemp[key])
        grade = score/p
        if grade != grade:
            #print("NAN!")
            1
        else:
            grades.append(100*grade)

        if idx==300:
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
    lname = dftemp['Student']
    #fname = dftemp['First Name']

    #hqave = summarize(i,hwandquizinfo,df,True)
    #hqave = summarize(i,hwandquizinfo,df,False)
    have = summarize(i,hwinfo,df,True)
    qave = summarize(i,quizinfo,df,True)
    feave = summarize(i,finalexaminfo,df)
    fpave = summarize(i,finalprojectinfo,df)

    # EDAV 
    ave = 0.25*have + 0.25*qave + 0.25*fpave + 0.25*feave
    print(f"{i:2d} {lname:22s}   {ave:5.1f}     h: {have:5.1f}  q: {qave:5.1f}  fp: {fpave:5.1f}  fe: {feave:5.1f}")
    # Quantum
    #ave = 0.25*have + 0.10*pave + 0.30*mave + 0.35*fave

    #print(f"{i:2d} {fname:12s} {lname:18s}   {ave:5.1f}     hw: {have:5.1f} p: {pave:5.1f} m: {mave:5.1f} f: {fave:5.1f}")




