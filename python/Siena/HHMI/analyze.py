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
sns.set()
#sns.palplot(sns.color_palette("GnBu_d"))
#sns.set_palette(sns.color_palette("GnBu_d"))
#sns.set_palette("PuBuGn_d")
#sns.set_palette("PuBuGn_d")
#sns.set_palette("husl")


#sns.set()
#sns.palplot(sns.color_palette("Paired"))
#sns.palplot(sns.color_palette("hls", 8))



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
# Combine MATH and PHYS
df['course'][df['course']=='PHYS 110'] = 'PHYS'
df['course'][df['course']=='PHYS 130'] = 'PHYS'

df['course'][df['course']=='MATH 110'] = 'MATH'
df['course'][df['course']=='MATH 105'] = 'MATH'

df['course'][df['course']=='BIOL 110'] = 'BIOL'
df['course'][df['course']=='CHEM 110'] = 'CHEM'
df['course'][df['course']=='CSIS 110'] = 'CSIS'

print(df[df['course']=='PHYS'])
print(df[df['course']=='MATH'])
dfmerged = df.groupby(['course','ethnicity','grade']).sum().reset_index()
print(dfmerged[dfmerged['course']=='PHYS'])
print(dfmerged[dfmerged['course']=='MATH'])
#exit()

# Make a data frame out of these
e = [] # Course
g = [] # Grade
s = [] # Fraction
c = [] # Course
n = [] # Numbers
#for course in courses:
for course in dfmerged['course'].values:
    print("------------------")
    print("------------------")
    print (course)
    print("------------------")
    for i,grade in enumerate(['A','B','C','D','F']):
        print("--- " + grade + " ---")
        x = dfmerged[(dfmerged['course']==course) & (dfmerged['grade']==grade)]['number']
        eth = dfmerged[(dfmerged['course']==course) & (dfmerged['grade']==grade)]['ethnicity']

        for a,b in zip(x,eth):
            print("{0:4.1f}% {1:22} (absolute number {2})".format(100*a/x.sum(),b,a))
            e.append(b)
            s.append(a/x.sum())
            c.append(course)
            g.append(grade)
            n.append(a)
        print()
data = np.array([c,e,g,s,n]).transpose()
print(data)
df2 = pd.DataFrame(data,columns=['course','ethnicity','grade','fraction','number'])
df2['number'] = df2['number'].apply(pd.to_numeric)
df2['fraction'] = df2['fraction'].apply(pd.to_numeric)

print(df2[(df2['ethnicity']=='White')])


sns.set_palette("colorblind")
#sns.set_color_codes(palette='deep')

for ethnicity in ethnicities:
    plt.figure(figsize=(14,3))
    sns.barplot(x="grade", y="fraction", hue='course', data=df2[(df2['ethnicity']==ethnicity) & (df2['course'] != 'ENVA 100')],ci=None, palette='Dark2')
    #sns.barplot(x="grade", y="fraction", hue='course', data=df2[(df2['ethnicity']==ethnicity)],ci=None, palette='Set2')
    
    plt.gcf().suptitle(ethnicity,fontsize=18)
    plt.ylim(0,0.5)
    #plt.ylim(0,1.1)
    plt.xlim(-0.5,5.0)
    plt.xlabel('')
    plt.ylabel('Frac. of students receiving grade')
    plt.tight_layout()
    plt.savefig(ethnicity.replace('/','_') +'_SUMMARY_ZOOMED.png')
    #plt.savefig(ethnicity.replace('/','_') +'_SUMMARY.png')

plt.show()
