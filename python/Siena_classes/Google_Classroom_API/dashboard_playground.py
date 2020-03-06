import matplotlib.pylab as plt
import numpy as np

nstudents = 10
breakdown = np.array([0.3, 0.35, 0.25, 0.2])
nassignments = np.array([8, 4, 2, 0])
ncategories = len(breakdown)

def gen_students(nstudents,breakdown,nassignments):

    students = []
    for n in range(nstudents):
        students.append([])
        for na in nassignments:
            print(na)
            r = np.random.randint(50,100,na)
            students[n].append(r)


    return students


students = gen_students(nstudents,breakdown,nassignments)
for student in students:
    print(student)


