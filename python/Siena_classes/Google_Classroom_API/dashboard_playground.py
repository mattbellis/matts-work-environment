import matplotlib.pylab as plt
import numpy as np

nstudents = 5
breakdown = np.array([0.3, 0.35, 0.25, 0.2])
breakdown /= 1.5
nassignments = np.array([8, 4, 2, 0])
ncategories = len(breakdown)

# Set colors
cmap = plt.get_cmap("binary")
colors = cmap(np.arange(0,256,int(256/ncategories)))


def gen_students(nstudents,breakdown,nassignments):

    students = []
    for n in range(nstudents):
        students.append([])
        for na in nassignments:
            #print(na)
            r = np.random.randint(50,100,na)
            students[n].append(r)


    return students


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

averages = np.array(averages).transpose()
print(averages)
X = np.arange(nstudents)
fig = plt.figure(figsize=(12,4))
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00,                      averages[0], color = colors[0], align='edge', width = breakdown[0], ec='k')
ax.bar(X + breakdown[0],              averages[1], color = colors[1], align='edge', width = breakdown[1], ec='k')
ax.bar(X + breakdown[0]+breakdown[1], averages[2], color = colors[2], align='edge', width = breakdown[2], ec='k')

plt.ylim(0,120)

plt.show()
