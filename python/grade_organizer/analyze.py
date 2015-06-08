import numpy as np
import matplotlib.pylab as plt

hw12 = np.loadtxt('HW_F12.dat')
ex12 = np.loadtxt('exams_F12.dat')

hw14 = np.loadtxt('HW_F14.dat')
ex14 = np.loadtxt('exams_F14.dat')

index = hw12>50
index *= ex12>50

hw12 = hw12[index]
ex12 = ex12[index]

index = hw14>50
index *= ex14>50

hw14 = hw14[index]
ex14 = ex14[index]

plt.hist(ex12,normed=1,bins=10,range=(60,100),label='Quizzes F12',alpha=0.75)
plt.hist(ex14,normed=1,bins=10,range=(60,100),label='Quizzes F14',alpha=0.5)
plt.xlabel('Grade',fontsize=24)
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(hw12,ex12,'o',label='F12')
plt.plot(hw14,ex14,'o',label='F14')
plt.xlabel('HW grade',fontsize=24)
plt.ylabel('Quiz grade',fontsize=24)
plt.ylim(50,120)
plt.xlim(50,120)
plt.legend()
plt.tight_layout()

print np.mean(ex12)
print np.std(ex12)
print np.mean(ex14)
print np.std(ex14)

print np.corrcoef(hw12,ex12)
print np.corrcoef(hw14,ex14)

plt.show()
