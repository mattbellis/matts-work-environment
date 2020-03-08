import matplotlib.pylab as plt
import numpy as np
import random
import string

import names
import pandas as pd

################################################################################
nstudents = 20
breakdown = np.array([0.3, 0.35, 0.25, 0.2])
breakdown /= 1.5
nassignments = np.array([8, 4, 2, 0])
assignment_types = np.array(['hw','quizzes','midterms','final exam'])
ncategories = len(breakdown)

# Set colors
cmap = plt.get_cmap("binary")
colors = cmap(np.arange(0,256,int(256/ncategories)))
################################################################################


################################################################################
def gen_students(nstudents,breakdown,nassignments):

    students = []
    for n in range(nstudents):
        students.append([])
        for na in nassignments:
            #print(na)
            r = np.random.randint(50,100,na)
            students[n].append(r)


    return students
################################################################################

ncols = 5
nrows = int(np.ceil(nstudents/ncols))

students = gen_students(nstudents,breakdown,nassignments)
averages = []
for i,student in enumerate(students):
    print(student)
    averages.append([])
    for grades in student:
        avg = np.mean(grades)
        if avg != avg:
            avg = 0.0
        averages[i].append(avg)
print("averages: ")
print(averages)

averages = np.array(averages).transpose()
print(averages)
X = np.arange(ncols)
fig = plt.figure(figsize=(12,2*nrows))
for i in range(nrows):
    #ax = fig.add_axes([0,0,1,1])
    ax = fig.add_subplot(nrows,1,1+i)
    start = i*5
    end = (i+1)*5
    print(start,end)
    if end>=nstudents:
        end = nstudents
    ax.bar(X + 0.00,                      averages[0][start:end], color = colors[0], align='edge', width = breakdown[0], ec='k')
    ax.bar(X + breakdown[0],              averages[1][start:end], color = colors[1], align='edge', width = breakdown[1], ec='k')
    ax.bar(X + breakdown[0]+breakdown[1], averages[2][start:end], color = colors[2], align='edge', width = breakdown[2], ec='k')
    ax.bar(X + breakdown[0]+breakdown[1]+breakdown[2], averages[3][start:end], color = colors[3], align='edge', width = breakdown[3], ec='k')

    plt.ylim(0,120)
plt.tight_layout()


#####################################
# Put it into a data frame
#####################################
name = []
date = []
ass_type = []
ass_name = []
grade = []
fraction = []

temp_names = []
for n in range(nstudents):
    temp_names.append(names.get_full_name())

print(temp_names)

nassignments = np.array([8, 4, 2, 0])
assignment_types = np.array(['hw','quizzes','midterms','final exam'])
breakdown = np.array([0.3, 0.35, 0.25, 0.2])

for n,a,b in zip(nassignments,assignment_types,breakdown):
    for i in range(n):
        an = ''.join(random.choices(string.ascii_uppercase +
                                         string.digits, k = 7)) 
        for tname in temp_names:
            name.append(tname)
            ass_type.append(a)
            ass_name.append(an)
            fraction.append(b)
            grade.append(np.random.randint(50,100))

data = {}
data['name'] = name
data['ass_type'] = ass_type
data['ass_name'] = ass_name
data['grade'] = grade
data['fraction'] = fraction

df = pd.DataFrame(data)

# Histograms
#df.groupby('ass_name').boxplot(column='grade')
df.boxplot(column='grade',by='ass_name')



plt.show()
