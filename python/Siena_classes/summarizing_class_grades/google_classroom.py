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
midterminfo = {}
finalinfo = {}
extracreditinfo = {}

for name in colnames:
    if name.find('QUIZ')>=0:
        quizinfo[name] = []
        hwandquizinfo[name] = []
    # EDAV
    elif name.find('HW')>=0 or name.find('HOMEWORK')>=0 or name.find('CLASS')>=0:
    # Quantum
    #elif name.find('HW')>=0 or name.find('HOMEWORK')>=0 or name.find('CODING')>=0:
        hwinfo[name] = []
        hwandquizinfo[name] = []
    elif name.find('PARTICIPATION')>=0 or name.find('PARTICIPATION')>=0:
        partinfo[name] = []
    elif name.find('EXTRA')>=0:
        extracreditinfo[name] = []
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
#print(finalinfo)

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
for key in partinfo.keys():
    d = df[key].iloc[0]
    p = float(df[key].iloc[1])
    partinfo[key] = [d,p]
    #print(key,d,p)
for key in extracreditinfo.keys():
    d = df[key].iloc[0]
    p = float(df[key].iloc[1])
    extracreditinfo[key] = [d,p]
    #print(key,d,p)
for key in midterminfo.keys():
    d = df[key].iloc[0]
    p = float(df[key].iloc[1])
    midterminfo[key] = [d,p]
    print(key,d,p)
for key in finalinfo.keys():
    #print(key)
    d = df[key].iloc[0]
    p = float(df[key].iloc[1])
    # This is just for PHYS 250 - Spring 2022
    if key.find('PROJECT')>=0 and key.find('FINAL')>=0:
        p = 75
        print(f"here!  {p}")
    finalinfo[key] = [d,p]
    print(key,d,p)

def summarize(idx,info,df,drop=0,verbose=False):

    dftemp = df.iloc[idx]
    lname = dftemp['Last Name']
    fname = dftemp['First Name']

    #print(f'{fname} {lname}')
    grades = []
    for key in info.keys():
        #print(key)
        d = info[key][0]
        p = info[key][1]

        if d!=d or p!=p:
            continue

        score = float(dftemp[key])
        grade = score/p
        if grade != grade:
            #print("NAN!")
            1
        else:
            grades.append(100*grade)

        #if idx==600:
        if verbose:
            print(f"{d:12s} max: {p:5.1f}    score: {score:5.1f}    grade: {100*grade:7.2f}      {key}")
    #print(grades)
    grades = np.sort(grades)[drop:].tolist()
    #print(grades)
    ave = np.mean(grades)
    #print(f"Average: {ave:8.2f}")
    return ave,grades




for i in range(2,len(df)):

    dftemp = df.iloc[i]
    lname = dftemp['Last Name']
    fname = dftemp['First Name']

    #hqave,hqgr = summarize(i,hwandquizinfo,df,drop=1)
    hqave,hqgr = summarize(i,hwandquizinfo,df,drop=0)
    pave,pgr = summarize(i,partinfo,df)
    have,hgr = summarize(i,hwinfo,df)
    qave,qgr = summarize(i,quizinfo,df,drop=0)
    #qave,qgr = summarize(i,quizinfo,df,drop=1)
    mave,mgr = summarize(i,midterminfo,df)
    fave,fgr = summarize(i,finalinfo,df,verbose=False)
    extave,exgr = summarize(i,extracreditinfo,df,verbose=False)

    # Extra credits count for a hw grade
    #print(hgr)
    #print(gr)
    #print(type(hgr), type(gr))
    #newhw = hqgr + exgr
    #print(hqave)
    #hqave = np.mean(newhw)
    #print(hqave)
    #print(midterminfo)

    # Final exam uses quiz grades
    #print("---------")
    #print(fave)
    #fave = (3*fave + qave)/4.0
    #print(fave)
    #print("----")

    # PHYS 260
    ave = (0.20*have + 0.45*qave)/0.65 # For midterm grades
    # EDAV  or PHYS 250
    ave = 0.5*hqave + 0.25*mave + 0.25*fave
    # Midterm
    #ave = (0.5*hqave + 0.25*mave)/0.75
    print(f"{i:2d} {fname:12s} {lname:18s}   {ave:5.1f}     hw: {hqave:5.1f}    m: {mave:5.1f}    f: {fave:5.1f}     h: {have:5.1f} q: {qave:5.1f}  ex: {extave:5.1f}")
    # Quantum
    #ave = 0.25*have + 0.10*pave + 0.30*mave + 0.35*fave

    #print(f"{i:2d} {fname:12s} {lname:18s}   {ave:5.1f}     hw: {have:5.1f} p: {pave:5.1f} m: {mave:5.1f} f: {fave:5.1f}")




