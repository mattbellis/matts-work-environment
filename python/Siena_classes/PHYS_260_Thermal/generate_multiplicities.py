import numpy as np

from scipy.misc import factorial

qtot = 20
N_A = 15
N_B = 10

mult_A = []
mult_B = []

for q_A in range(0,qtot+1):

    q_B = qtot - q_A

    mult_A.append(factorial(q_A+N_A)/(factorial(q_A)*factorial(N_A)))
    mult_B.append(factorial(q_B+N_B)/(factorial(q_B)*factorial(N_B)))

    #print "%-10d & %-10d & %-10d & %-10d \\\\" % (q_A,mult_A[q_A],q_B,mult_B[q_A])
    print "%-10d & %-10d & %-10d & %-10d & %-10d \\\\" % (q_A,mult_A[q_A],q_B,mult_B[q_A],mult_A[q_A]*mult_B[q_A])


for i in range(0,len(mult_A)-1):

    if i > 0:
        print i,np.log(mult_A[i+1]) - np.log(mult_A[i-1]) , qtot-i,np.log(mult_B[i-1]) - np.log(mult_B[i+1])
