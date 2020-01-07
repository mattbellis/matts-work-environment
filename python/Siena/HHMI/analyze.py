import sys
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

departments = ['MATH', 'BIOL', 'PHYS', 'CSIS', 'ENVA', 'CHEM']
ethnicities = ["UM and MR", 'Asian', 'White', 'Unknown/International']
outcomes = ['Passed','A','B','C','D','P/F','Failed','Withdrew']
letters = ['A','B','C','D','Failed']

grades = {}

infilename = sys.argv[1]

infile = open(infilename)

line = infile.readline()
while 1:

    vals = line.split(',')
    #print(vals)

    if line=='':
        break

    if len(vals)>1:
        for department in departments:
            if vals[1].find(department)>=0:
                course = vals[1].strip()
                #print(course)
                grades[course] = {}
                for ethnicity in ethnicities:
                    grades[course][ethnicity] = {}

                line = infile.readline()
                line = infile.readline()
                line = infile.readline()
                line = infile.readline()
                line = infile.readline()

                for outcome in outcomes:
                    line = infile.readline()
                    vals = line.split(',')
                    for i,ethnicity in enumerate(ethnicities):
                        if len(outcome)>3: # Passed, failed, and withdraw parse different
                            grades[course][ethnicity][outcome] = int(vals[(2*i)+1])
                        else:
                            print(vals)
                            grades[course][ethnicity][outcome] = int(vals[(2*i)+2])

                # To get past the "Total"
                infile.readline()

    line = infile.readline()

print(grades)

################################################################################
e = []
g = []
s = []
c = []
#'''
for coursename in grades.keys():
    course = grades[coursename]
    
    for i,ethnicity in enumerate(ethnicities):
        for grade in letters:
            print(coursename,ethnicity,grade,course[ethnicity][grade])
            c.append(coursename)
            e.append(ethnicity)
            g.append(grade)
            s.append(course[ethnicity][grade])

#'''
#df = pd.DataFrame(data=grades)
#print(df)
data = np.array([c,e,g,s]).transpose()
print(data)
df = pd.DataFrame(data,columns=['course','ethnicity','grade','number'])
df['number'] = df['number'].apply(pd.to_numeric)
df['grade'][df['grade']=='Failed'] = 'F'
print(df)

################################################################################
#tips = sns.load_dataset('tips')
#print(tips)
#print(type(tips))
#exit()

courses = np.unique(df['course'].values)
print(courses)

sns.set(style='whitegrid')

'''
for course in courses:
    plt.figure(figsize=(14,7))
    for i,ethnicity in enumerate(ethnicities):
        plt.subplot(2,4,i+1)
        sns.barplot(x="grade", y="number", data=df[(df['course']==course) & (df['ethnicity']==ethnicity)],ci=None)
        plt.title(ethnicity)
    #plt.tight_layout()

    #plt.figure(figsize=(14,5))
    for i,grade in enumerate(['A','B','C','D','F']):
        plt.subplot(2,5,i+1+5)
        chart = sns.barplot(x="ethnicity", y="number", data=df[(df['course']==course) & (df['grade']==grade)],ci=None)
        plt.title(grade)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, fontsize=8, horizontalalignment='right')

    #plt.tight_layout()
    plt.gcf().suptitle(course,fontsize=18)
    plt.savefig(course+'.png')
'''

'''
for course in courses:
    plt.figure(figsize=(14,3))
    sns.barplot(x="grade", y="number", hue='ethnicity', data=df[(df['course']==course)],ci=None)
    #plt.title(ethnicity)
    #plt.tight_layout()
    plt.gcf().suptitle(course,fontsize=18)
    plt.savefig(course+'_COMBINED_.png')
    plt.tight_layout()
'''


#plt.show()

for course in courses:
    print("------------------")
    print("------------------")
    print (course)
    print("------------------")
    for i,grade in enumerate(['A','B','C','D','F']):
        print("--- " + grade + " ---")
        x = df[(df['course']==course) & (df['grade']==grade)]['number']
        e = df[(df['course']==course) & (df['grade']==grade)]['ethnicity']
        for a,b in zip(x,e):
            print("{0:4.1f}% {1:22} (absolute number {2})".format(100*a/x.sum(),b,a))
        print()

